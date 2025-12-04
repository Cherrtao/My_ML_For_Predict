# models/nograph_gru.py
from __future__ import annotations

import torch
import torch.nn as nn # 导入神经网络模块，方便写模型

from config.model_config import NUM_NODES # 从静态配置中导入节点数量（N）


class NoGraphGRU(nn.Module):
    """
    不使用图结构的 GRU baseline：
    - 输入 X: (B, T_in, N, F)
    - 把每个时间步的 (N, F) 展平为 (N*F)，得到 (B, T_in, N*F) 序列
    - 交给 GRU 编码，取最后时刻的 hidden
    - 用一个全连接层一次性预测未来 T_out 个点（main_power 序列）
    """

    def __init__(
        self,
        input_dim: int,   # F_max，单个节点的特征维度（zero padding 后的F_max）
        hidden_dim: int,  # GRU 隐藏层维度H
        t_out: int,       # 预测长度（例如 288）
        num_layers: int = 1,  # GRU堆叠的层数，默认为1层
        dropout: float = 0.0, # GRU内部的dropout比例，大于一层时才会生效
    ) -> None:
        super().__init__() # 调用父类nn.Module的构造函数，初始化基础模块

        self.input_dim = input_dim  # 记录输入特征维度F，用于shape校验
        self.hidden_dim = hidden_dim # 记录GRU隐藏层维度H
        self.t_out = t_out # 记录预测长度T_out
        self.num_layers = num_layers # 记录GRU层数

        self.num_nodes = NUM_NODES # 记录节点数量N，从配置中读取

        # GRU 输入维度 = N * F_max
        # 定义DRU层，输入维度是N*F_amx，因为把所有节点特征平拼在一起
        self.gru = nn.GRU(
            input_size=self.num_nodes * self.input_dim,  # 每个时间步输入向量长度 = N * F
            hidden_size=self.hidden_dim, # 隐藏层状态维度H
            num_layers=self.num_layers, # GRU堆叠层数
            batch_first=True,  # 输入 (B, T, ·) # 输入格式采用(batch,time,feature)
            dropout=dropout if self.num_layers > 1 else 0.0,  # 只有层数>1 时才启用 dropou
        )

        # 从最后时刻 hidden 一次性预测 T_out 个点
        # 定义输出层：从最后的隐藏状态 H 映射到 T_out 个预测值
        self.fc = nn.Linear(self.hidden_dim, self.t_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor
            形状 (B, T_in, N, F_max)

        返回
        ----
        y_hat : torch.Tensor
            形状 (B, T_out)
        """
        B, T_in, N, F = x.shape  # 解包得到batch大小B, 时间步T_in, 节点数N，特征维T
        # 对节点数进行断言检查，防止配置不一致
        assert N == self.num_nodes, f"节点数不匹配: x.N={N}, NUM_NODES={self.num_nodes}"
        # 对特征维度进行断言检查，防止配置不一样
        assert F == self.input_dim, f"特征维度不匹配: x.F={F}, input_dim={self.input_dim}"

        # 展平节点维度: (B, T_in, N, F) -> (B, T_in, N*F)
        # 将节点维度和特征维度展平
        # 这样GRU每个时间步看到的是所有节点拼在一起的一长条特征向量
        x_flat = x.reshape(B, T_in, N * F)

        # GRU 编码
        #   out: (B, T_in, H)
        #   h_n: (num_layers, B, H)
        # 将展平后的序列送入GRU
        # Out : 每个时间步的输出，形状
        # h_n: 每一层最后一个时间步的hidden, 形状(num_Layers,B,H)
        out, h_n = self.gru(x_flat)

        # 取最后一层的 hidden，形状 (B, H)
        # 这相当于把整个过去T_in的信息压缩到一个向量h_last
        h_last = h_n[-1]

        # 预测未来 T_out 个时间步
        # 使用全连接层将隐藏向量映射到T_out维度，得到未来T_out步的预测序列
        y_hat = self.fc(h_last)   # (B, T_out)
        # 返回预测结果
        return y_hat
