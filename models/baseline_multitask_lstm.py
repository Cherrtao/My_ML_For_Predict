# models/multitask_lstm.py
"""
Multi-task Baseline LSTM

共享一套 LSTM 表示：
- 头1：预测未来 24h 的 5min 粒度主功率序列（长度 288）
- 头2：预测未来 24h 的逐小时主功率均值（长度 24）

x: (B, T_in, F_main)
y_5min_hat: (B, 288)
y_1h_hat  : (B, 24)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        out_5min: int = 288,
        out_1h: int = 24,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_5min = out_5min
        self.out_1h = out_1h

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 两个任务头，共享同一个 h_T
        self.fc_5min = nn.Linear(hidden_dim, out_5min)
        self.fc_1h = nn.Linear(hidden_dim, out_1h)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T_in, F_in)

        返回:
        -------
        y_5min_hat: (B, out_5min)
        y_1h_hat  : (B, out_1h)
        """
        out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        h_last = out[:, -1, :]  # (B, hidden_dim)

        y_5min_hat = self.fc_5min(h_last)
        y_1h_hat = self.fc_1h(h_last)

        return y_5min_hat, y_1h_hat
