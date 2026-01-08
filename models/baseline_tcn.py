# models/baseline_tcn.py
"""
Baseline: TCN for main_power forecasting

- 输入: X_main, 形状 (batch, T_in, F_main)
- 输出: y_hat, 形状 (batch, out_len)  -> 未来 out_len 个时间步的 main_power（标准化空间）

和 BaselineLSTM 的接口尽量保持一致，方便在训练脚本里互换对比。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    一个标准的 TCN block:
    Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout + Residual

    这里固定 kernel_size=3, padding=dilation，保证时间长度不变：
    L_out = L_in，当 stride=1 且 padding=dilation、kernel_size=3 时成立。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        assert kernel_size == 3, "当前实现假定 kernel_size=3 以保持长度不变。"
        padding = dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 如果通道数变了，用 1x1 卷积做 residual 对齐
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.init_weights()

    def init_weights(self) -> None:
        for m in [self.conv1, self.conv2, self.downsample]:
            if m is not None:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T)
        """
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class BaselineTCN(nn.Module):
    """
    TCN Baseline，用最后一个时间步的特征映射到整个预测序列。

    输入:
        x: (B, T_in, F_in)
    输出:
        y_hat: (B, out_len)
    """

    def __init__(
        self,
        input_dim: int,
        out_len: int = 288,
        channels: int = 32,
        num_levels: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.out_len = out_len

        layers = []
        in_ch = input_dim
        for i in range(num_levels):
            dilation = 2 ** i  # 1, 2, 4, ...
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = channels

        self.tcn = nn.Sequential(*layers)
        # 用最后一个时间步的通道特征 -> 全序列预测
        self.fc = nn.Linear(channels, out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, F_in)
        return: (B, out_len)
        """
        # 先转成 (B, F_in, T_in) 给 Conv1d 用
        x = x.permute(0, 2, 1)  # (B, F_in, T_in)

        feat = self.tcn(x)      # (B, C, T_in)，每一层都保持 T_in 不变
        last_feat = feat[:, :, -1]  # (B, C) 取最后一个时间步，已包含整个历史的感受野

        y_hat = self.fc(last_feat)  # (B, out_len)
        return y_hat
