# models/baseline_lstm.py
"""
Baseline: Total-only LSTM

只使用“总表节点（Main）”的历史特征做预测：
- X_main: (batch, T_in, F_main)
- y_hat : (batch, T_out)

这里的 F_main = 主节点局部特征 + 时间特征
例如:
    NODE_LOCAL_FEATURES["Main"] = ["main_power"]
    TIME_FEATURES = ["dayofweek", "is_weekend", "tod_sin", "tod_cos"]
则 F_main = 1 + 4 = 5
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        out_len: int = 288,
        dropout: float = 0.0,
    ) -> None:
        """
        参数
        ----
        input_dim : int
            输入特征维度 F_main。
        hidden_dim : int
            LSTM 隐层维度。
        num_layers : int
            LSTM 堆叠层数。
        out_len : int
            预测序列长度（T_out），默认为 288（未来 24h）。
        dropout : float
            若 num_layers > 1，在 LSTM 层之间使用的 dropout。
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_len = out_len

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 用最后一个时间步的隐状态 h_T 映射到长度 out_len 的序列预测
        self.fc = nn.Linear(hidden_dim, out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数
        ----
        x : torch.Tensor
            输入序列，形状为 (batch_size, T_in, input_dim)。

        返回
        ----
        torch.Tensor
            预测序列，形状为 (batch_size, out_len)。
        """
        # x: (B, T_in, F_in)
        out, (h_n, c_n) = self.lstm(x)
        # out: (B, T_in, hidden_dim)
        # h_n: (num_layers, B, hidden_dim)

        # 取最后时间步的输出，也可以用 h_n[-1]
        h_last = out[:, -1, :]  # (B, hidden_dim)

        y_hat = self.fc(h_last)  # (B, out_len)
        return y_hat
