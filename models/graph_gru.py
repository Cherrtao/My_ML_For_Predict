# models/graph_gru.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config.model_config import NUM_NODES, NODE_INDEX, ADJ_NORM


class GraphGRU(nn.Module):
    """
    带图结构的 GRU 模型，用于预测 Main 节点的未来序列（或残差）。

    输入:
        x: (B, T_in, N, F_max)

    流程:
        1) 对每个时间步做一次图聚合:
               X_agg[t] = A_norm @ X[t]              # (B, N, F)
        2) 把每个节点当作一条时间序列:
               (B, T_in, N, F) -> (B*N, T_in, F)
           送入 GRU 得到每个节点的最后 hidden。
        3) 取 Main 节点的 hidden，过全连接层，预测 T_out 个点。

    输出:
        y_hat: (B, T_out)      # Main 节点的预测序列（可理解为残差）
    """

    def __init__(
        self,
        input_dim: int,          # F_max
        hidden_dim: int,         # GRU 隐藏维度 H
        t_out: int,              # 预测长度 T_out
        num_layers: int = 1,
        dropout: float = 0.0,    # num_layers > 1 时才生效
        main_idx: Optional[int] = None,
        adj_matrix: Optional[torch.Tensor] = None,  # 默认用 ADJ_NORM
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.t_out = t_out
        self.num_layers = num_layers

        self.num_nodes = NUM_NODES
        self.main_idx = NODE_INDEX["Main"] if main_idx is None else main_idx

        # ====== 邻接矩阵（已经在 config 里归一化过的 ADJ_NORM） ======
        if adj_matrix is None:
            adj = ADJ_NORM.clone().detach()
        else:
            adj = adj_matrix.float()

        if adj.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(
                f"邻接矩阵形状应为 ({self.num_nodes}, {self.num_nodes})，实际为 {adj.shape}"
            )

        # 注册成 buffer，会跟着模型一起存/搬设备
        self.register_buffer("adj_norm", adj)

        # ====== GRU：每个节点单独按时间做编码 ======
        self.gru = nn.GRU(
            input_size=self.input_dim,     # 每个节点每个时间步的特征长度 F_max
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,              # (B*N, T_in, F)
            dropout=dropout if self.num_layers > 1 else 0.0,
        )

        # 只用 Main 节点的 hidden 做预测
        self.fc = nn.Linear(self.hidden_dim, self.t_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, N, F_max)
        return: y_hat : (B, T_out)
        """
        B, T_in, N, F = x.shape
        assert N == self.num_nodes, f"节点数不匹配: x.N={N}, NUM_NODES={self.num_nodes}"
        assert F == self.input_dim, f"特征维度不匹配: x.F={F}, input_dim={self.input_dim}"

        # 1️⃣ 图聚合：对每个时间步做 A_norm @ X[t]
        # adj_norm: (N, N), x: (B, T, N, F) -> (B, T, N, F)
        x_agg = torch.einsum("ij,btjf->btif", self.adj_norm, x)

        # 2️⃣ 按节点展开成 (B*N, T_in, F)，每个节点一条时间序列
        x_seq = x_agg.reshape(B * N, T_in, F)   # (B*N, T_in, F)

        out, h_n = self.gru(x_seq)             # h_n: (num_layers, B*N, H)
        h_last = h_n[-1]                       # (B*N, H)

        # 3️⃣ 还原节点维度 -> (B, N, H)
        h_last = h_last.view(B, N, self.hidden_dim)

        # 4️⃣ 取 Main 节点 hidden -> 预测 T_out 个点
        h_main = h_last[:, self.main_idx, :]   # (B, H)
        y_hat = self.fc(h_main)                # (B, T_out)

        return y_hat
