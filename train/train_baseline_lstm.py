# train_baseline_lstm.py
"""
训练 Baseline: Total-only LSTM

- 只用 Main 节点的特征（main_power + 时间特征）
- 用 GraphSequenceDataset 生成 X, y，然后在 batch 维度上
  截取 Main 节点对应的特征输入 LSTM
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt  # 用于画 loss 曲线

from config.dataset_config import TRAIN_CSV, VAL_CSV
from config.model_config import NODE_INDEX, NODE_INPUT_DIMS
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from models.baseline_lstm import BaselineLSTM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ========= 1. 拟合 scaler =========
    scaler = fit_scaler_from_csv(str(TRAIN_CSV))
    print("Scaler fitted on:", TRAIN_CSV)

    # ========= 2. 构造 Dataset / DataLoader =========
    T_IN = 288   # 过去 24h
    T_OUT = 288  # 未来 24h

    train_ds = GraphSequenceDataset(
        csv_path=str(TRAIN_CSV),
        t_in=T_IN,
        t_out=T_OUT,
        feature_scaler=scaler,
        fit_scaler=False,
    )

    val_ds = GraphSequenceDataset(
        csv_path=str(VAL_CSV),
        t_in=T_IN,
        t_out=T_OUT,
        feature_scaler=scaler,
        fit_scaler=False,
    )

    print("Train samples:", len(train_ds))
    print("Val   samples:", len(val_ds))

    BATCH_SIZE = 32  # 保持不变
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ========= 3. 构造 Baseline LSTM 模型 =========
    main_idx = NODE_INDEX["Main"]          # Main 节点的索引
    main_dim = NODE_INPUT_DIMS["Main"]     # Main 节点输入维度（局部 + 时间）

    # 这里保持 hidden_dim=64，只动“正则化”这一档旋钮
    model = BaselineLSTM(
        input_dim=main_dim,
        hidden_dim=64,       # 保持 64 不变
        num_layers=1,        # 单层 LSTM，dropout 参数不会真正生效
        out_len=T_OUT,
        dropout=0.0,         # 单层时内部 dropout 本身就无效，写 0 更干脆
    ).to(device)

    print(model)

    # ========= 4. 损失函数 & 优化器 =========
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,   # ★ 这里加了 L2 正则（权重衰减）
    )
    # 提示：如果看起来仍然过拟合，可以尝试调大一点，比如 5e-5 或 1e-4

    # ========= 5. 训练循环 =========
    NUM_EPOCHS = 20
    best_val_loss = float("inf")

    # 记录每个 epoch 的 loss，方便画曲线
    train_losses = []
    val_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ----- 5.1 训练阶段 -----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for X, y in train_loader:
            # X: (B, T_in, N, F_max)
            # y: (B, T_out)

            X = X.to(device)
            y = y.to(device)

            # 只取 Main 节点 & 其实际特征维度（前 main_dim 维）
            X_main = X[:, :, main_idx, :main_dim]  # (B, T_in, main_dim)

            optimizer.zero_grad()
            y_hat = model(X_main)                  # (B, T_out)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * X.size(0)
            train_count += X.size(0)

        train_loss = train_loss_sum / train_count
        train_losses.append(train_loss)

        # ----- 5.2 验证阶段 -----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                X_main = X[:, :, main_idx, :main_dim]
                y_hat = model(X_main)

                loss = criterion(y_hat, y)
                val_loss_sum += loss.item() * X.size(0)
                val_count += X.size(0)

        val_loss = val_loss_sum / val_count
        val_losses.append(val_loss)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )

        # ----- 5.3 保存最优模型 -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = Path("checkpoints")
            ckpt_path.mkdir(exist_ok=True)
            save_file = ckpt_path / "baseline_lstm_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler": scaler,
                    "config": {
                        "T_IN": T_IN,
                        "T_OUT": T_OUT,
                        "main_idx": main_idx,
                        "main_dim": main_dim,
                    },
                },
                save_file,
            )
            print(f"  -> New best model saved to {save_file}")

    print("Training finished. Best Val MSE:", best_val_loss)

    # ========= 6. 画 loss 曲线并保存 =========
    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train MSE")
    plt.plot(epochs, val_losses, marker="s", label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Baseline LSTM Training & Validation Loss (with L2)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # 保存到文件，方便在项目目录下查看
    out_path = Path("checkpoints") / "baseline_lstm_loss_curve_l2.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Loss curve saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
