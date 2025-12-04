# models/graph_gru.py
"""
Graph-GRU 模型

思路：
- 输入 X: (B, T_in, N, F_max)
  其中 N=5 个节点：
    0: BF        -> 包覆机群节点（bf_power + 时间特征）
    1: Cold      -> 冷风机节点（cold_power, cold_freq + 时间特征）
    2: Exhaust   -> 排风机节点（exh_voltage_v, exh_moto_temp + 时间特征）
    3: Env       -> 环境节点（env_temp, env_hum, env_press + 时间特征）
    4: Main      -> 总表节点（main_power + 时间特征）

- 第一步：对每个时间步做一层 GCN（用 ADJ_NORM 聚合邻居信息）：
    X_t: (B, N, F_in)  ->  图卷积后 H_t: (B, N, G)
- 第二步：把每个时间步的 H_t 展平为 (N * G)，得到序列：
    (B, T_in, N*G)
  送入 GRU 做时序建模。
- 第三步：用 GRU 最后时刻的 hidden，映射到未来 T_out=288 点的 main_power 预测序列。
"""

from __future__ import annotations

import torch
import torch.nn as nn  # 神经网络模块

from config.model_config import NUM_NODES, ADJ_NORM  # 节点数 N 和 归一化邻接矩阵 A_norm


class GraphGRU(nn.Module):
    """
    图结构版 GRU：

    输入:
        x: (B, T_in, N, F_in)  其中 F_in = F_max (zero padding 后的统一特征维度)
    输出:
        y_hat: (B, T_out)      只预测 Main 节点的 main_power 序列
    """

    def __init__(
        self,
        input_dim: int,        # F_in = F_max，单个节点的输入特征维度
        gcn_hidden_dim: int,   # G，GCN 输出的节点嵌入维度
        gru_hidden_dim: int,   # H，GRU 隐层维度
        t_out: int,            # T_out，预测序列长度（例如 288）
        num_layers: int = 1,   # GRU 堆叠层数
        dropout: float = 0.0,  # GRU 内部 dropout（只有 num_layers>1 时才有效）
    ) -> None:
        # 调用父类构造函数，初始化 nn.Module 基类
        super().__init__()

        # 记录一些超参数，后面做 shape 检查或保存模型用
        self.input_dim = input_dim          # 节点输入特征维度 F_in
        self.gcn_hidden_dim = gcn_hidden_dim  # GCN 输出维度 G
        self.gru_hidden_dim = gru_hidden_dim  # GRU 隐藏层维度 H
        self.t_out = t_out                  # 预测长度 T_out
        self.num_layers = num_layers        # GRU 层数
        self.num_nodes = NUM_NODES          # 节点数量 N（从静态配置读）

        # ---------- 图结构（归一化邻接矩阵） ----------
        # register_buffer 的作用是：
        #   1) 这个张量会随着模型一起保存 / 加载
        #   2) 不会被当成可训练参数参与优化
        #   3) .to(device) 时会跟着模型一起搬到 GPU/CPU
        self.register_buffer("adj_norm", ADJ_NORM.clone())  # 形状 (N, N)

        # ---------- GCN 线性变换层 ----------
        # 经典 GCN 的公式：H = A_norm * X * W
        # 这里用一个 nn.Linear 来实现 X * W 部分：
        #   输入: X: (..., F_in)
        #   输出: (..., G)
        self.gcn_linear = nn.Linear(
            in_features=self.input_dim,    # 输入特征维度 F_in
            out_features=self.gcn_hidden_dim,  # 输出维度 G
            bias=True,                     # 保留偏置项
        )

        # ---------- GRU 层 ----------
        # GRU 的输入是“所有节点的图嵌入拼起来”的长向量：
        #   每个时间步: (N, G) -> 展平为 (N * G)
        self.gru = nn.GRU(
            input_size=self.num_nodes * self.gcn_hidden_dim,  # 每个时间步输入长度 = N * G
            hidden_size=self.gru_hidden_dim,                  # GRU 隐层维度 H
            num_layers=self.num_layers,                       # 堆叠层数
            batch_first=True,                                 # 输入/输出维度格式为 (B, T, ·)
            dropout=dropout if self.num_layers > 1 else 0.0,  # 单层时禁用 dropout
        )

        # ---------- 输出层 ----------
        # 使用最后一个时间步的 hidden 向量 h_last: (B, H)
        # 一次性映射成 T_out 个预测值（未来 24h 的 main_power）
        self.fc = nn.Linear(
            in_features=self.gru_hidden_dim,  # 输入是 GRU 隐层 H
            out_features=self.t_out,          # 输出长度 T_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数
        ----
        x : torch.Tensor
            输入序列，形状为 (B, T_in, N, F_in)

        返回
        ----
        torch.Tensor
            预测序列 y_hat，形状为 (B, T_out)
        """
        # 解包形状，便于后续使用
        B, T_in, N, F_in = x.shape  # B: batch_size, T_in: 输入长度, N: 节点数, F_in: 特征维

        # 断言检查：节点数量必须和配置一致
        assert N == self.num_nodes, f"节点数不匹配: x.N={N}, NUM_NODES={self.num_nodes}"
        # 断言检查：特征维度必须匹配 input_dim
        assert F_in == self.input_dim, f"特征维度不匹配: x.F={F_in}, input_dim={self.input_dim}"

        # ============================================================
        # 1) 对每个时间步做 GCN：X_t -> H_t
        # ============================================================

        # 先把 batch 维和时间维合并，方便一次性做矩阵运算：
        #   (B, T_in, N, F_in) -> (B*T_in, N, F_in)
        x_bt = x.reshape(B * T_in, N, F_in)

        # 先做线性变换：X * W，得到节点“局部 + 时间特征”映射后的表示
        #   x_lin: (B*T_in, N, G)
        x_lin = self.gcn_linear(x_bt)

        # 使用归一化邻接矩阵做信息聚合：
        #   H = A_norm @ x_lin
        # 这里 torch.matmul 会自动把 (N, N) 广播到 (B*T_in, N, N)
        #   结果 h_gcn: (B*T_in, N, G)
        h_gcn = torch.matmul(self.adj_norm, x_lin)

        # 加一个非线性激活（ReLU），增强表达能力
        h_gcn = torch.relu(h_gcn)

        # 把 (B*T_in, N, G) 再reshape回 (B, T_in, N, G)
        h_gcn = h_gcn.reshape(B, T_in, N, self.gcn_hidden_dim)

        # ============================================================
        # 2) 展平节点维度，送入 GRU 做时序建模
        # ============================================================

        # 展平节点维度：
        #   (B, T_in, N, G) -> (B, T_in, N*G)
        h_seq = h_gcn.reshape(B, T_in, N * self.gcn_hidden_dim)

        # 调用 GRU：
        #   out: (B, T_in, H)  -> 每个时间步的隐藏状态
        #   h_n: (num_layers, B, H) -> 每一层最后一个时间步的 hidden
        out, h_n = self.gru(h_seq)

        # 取最后一层的 hidden（最后一个时间步的表示）
        #   形状: (B, H)
        h_last = h_n[-1]

        # ============================================================
        # 3) 用全连接层预测未来 T_out 个时间步的 main_power
        # ============================================================

        # 从 h_last: (B, H) 映射到 (B, T_out)
        y_hat = self.fc(h_last)

        # 返回预测结果
        return y_hat
