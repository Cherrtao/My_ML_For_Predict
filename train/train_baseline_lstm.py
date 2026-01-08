# train/train_baseline_lstm_single.py
"""
单任务 Baseline LSTM 训练脚本（5min 主功率预测）

- 输入：GraphSequenceDataset 提供的 (X, y)
  - X: (B, T_in, N, F_max)
  - y: (B, 288)  -> 未来 24h 的 5min 主功率（已标准化）

- 模型：只用 Main 节点的历史特征，LSTM 直接预测 y
- 损失：标准化空间 MSE
- 评估：同时给出反标准化后的 RMSE / MAE / MAPE
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config.dataset_config import TRAIN_CSV, VAL_CSV
from config.model_config import NODE_INDEX, NODE_INPUT_DIMS
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from models.baseline_lstm import BaselineLSTM


# ================== 超参数 ==================
T_IN = 288          # 过去 24h（5min 粒度）
T_OUT = 288         # 未来 24h（5min 序列）
BATCH_SIZE = 32

HIDDEN_DIM = 48
NUM_LAYERS = 2
DROPOUT = 0.2

LR = 5e-4           # 比 1e-3 小一档，更稳
WEIGHT_DECAY = 5e-5
NUM_EPOCHS = 30     # 多给一点训练轮数


def build_datasets(t_in: int, t_out: int):
    """
    构建 train / val 数据集，并在训练集上拟合标准化参数。
    """
    scaler = fit_scaler_from_csv(str(TRAIN_CSV))

    train_ds = GraphSequenceDataset(
        csv_path=str(TRAIN_CSV),
        t_in=t_in,
        t_out=t_out,
        feature_scaler=scaler,
        fit_scaler=False,
    )

    val_ds = GraphSequenceDataset(
        csv_path=str(VAL_CSV),
        t_in=t_in,
        t_out=t_out,
        feature_scaler=scaler,
        fit_scaler=False,
    )

    return train_ds, val_ds, scaler


def train_one_epoch(
    model: BaselineLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    """
    单个 epoch 训练：直接拟合 y（标准化 main_power），不是残差。
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    main_idx = NODE_INDEX["Main"]
    main_dim = NODE_INPUT_DIMS["Main"]

    for X, y in loader:
        # X: (B, T_in, N, F_max)
        # y: (B, T_out) —— 标准化后的 main_power
        X = X.to(device)
        y = y.to(device)

        # 只取 Main 节点特征
        X_main = X[:, :, main_idx, :main_dim]   # (B, T_in, main_dim)

        optimizer.zero_grad()
        y_hat = model(X_main)                  # (B, T_out)

        loss = criterion(y_hat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = X.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def eval_one_epoch(
    model: BaselineLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler,
):
    """
    验证集评估（单任务）：

    返回：
      - mse_norm : 标准化空间 MSE
      - rmse_kw  : 反标准化后的 RMSE（kW）
      - mae_kw   : 反标准化后的 MAE（kW）
      - mape     : 反标准化后的 MAPE（0~1）
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    total_se_kw = 0.0   # Σ e^2  (kW^2)
    total_ae_kw = 0.0   # Σ |e|   (kW)
    total_ape = 0.0     # Σ |e| / |y|
    n_points = 0

    main_idx = NODE_INDEX["Main"]
    main_dim = NODE_INPUT_DIMS["Main"]

    main_mean = float(scaler["mean"]["main_power"])
    main_std = float(scaler["std"]["main_power"])

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        X_main = X[:, :, main_idx, :main_dim]
        y_hat = model(X_main)

        loss = criterion(y_hat, y)

        bs = X.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        # ---- 反标准化到 kW，计算 RMSE / MAE / MAPE ----
        true_y_kw = y * main_std + main_mean
        pred_y_kw = y_hat * main_std + main_mean

        err_kw = pred_y_kw - true_y_kw
        se_kw = (err_kw ** 2)
        ae_kw = err_kw.abs()

        total_se_kw += se_kw.sum().item()
        total_ae_kw += ae_kw.sum().item()
        n_points += se_kw.numel()

        eps = 1e-3
        ape = ae_kw / (true_y_kw.abs() + eps)
        total_ape += ape.sum().item()

    mse_norm = total_loss / max(n_samples, 1)
    rmse_kw = (total_se_kw / max(n_points, 1)) ** 0.5
    mae_kw = total_ae_kw / max(n_points, 1)
    mape = total_ape / max(n_points, 1)

    return mse_norm, rmse_kw, mae_kw, mape


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ========= 1. 数据集 =========
    train_ds, val_ds, scaler = build_datasets(T_IN, T_OUT)
    print("Train samples:", len(train_ds))
    print("Val   samples:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ========= 2. 模型 / 损失 / 优化器 =========
    main_dim = NODE_INPUT_DIMS["Main"]

    model = BaselineLSTM(
        input_dim=main_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        out_len=T_OUT,
        dropout=DROPOUT,
    ).to(device)

    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ========= 3. 训练循环 =========
    best_val_rmse = float("inf")
    best_state = None

    train_mse_hist = []
    val_mse_hist = []
    val_rmse_hist = []
    val_mae_hist = []
    val_mape_hist = []

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        # 手动 LR 衰减：训练到一半 / 后期各压一次
        if epoch in (20, 30):
            for g in optimizer.param_groups:
                g["lr"] *= 0.3
            print(f"[LR] decayed to {optimizer.param_groups[0]['lr']:.1e}")

        train_mse = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_mse, val_rmse_kw, val_mae_kw, val_mape = eval_one_epoch(
            model, val_loader, criterion, device, scaler
        )

        train_mse_hist.append(train_mse)
        val_mse_hist.append(val_mse)
        val_rmse_hist.append(val_rmse_kw)
        val_mae_hist.append(val_mae_kw)
        val_mape_hist.append(val_mape)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train MSE={train_mse:.4f} | "
            f"Val MSE={val_mse:.4f}, "
            f"Val RMSE={val_rmse_kw:.2f} kW, "
            f"Val MAE={val_mae_kw:.2f} kW, "
            f"Val MAPE={val_mape*100:.2f}%, "
            f"LR={optimizer.param_groups[0]['lr']:.1e}"
        )

        # 以 RMSE(kW) 作为主指标保存最优模型
        if val_rmse_kw < best_val_rmse:
            best_val_rmse = val_rmse_kw
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            save_path = ckpt_dir / "baseline_lstm_single_best.pt"
            torch.save(
                {
                    "model_state_dict": best_state,
                    "scaler": scaler,
                    "config": {
                        "T_IN": T_IN,
                        "T_OUT": T_OUT,
                        "main_dim": main_dim,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                    },
                },
                save_path,
            )
            print(f"  -> New best model saved to {save_path}")

    print("Training finished. Best Val RMSE (kW):", best_val_rmse)

    # ========= 4. 画 loss 曲线（标准化 MSE） =========
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_mse_hist, "o-", label="Train MSE (norm)")
    plt.plot(epochs, val_mse_hist, "s-", label="Val MSE (norm)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (normalized)")
    plt.title("Baseline LSTM (5min main_power)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_path = ckpt_dir / "baseline_lstm_single_loss_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Loss curve saved to: {out_path}")
    plt.close()

    # 方便 notebook 画更多图
    return train_mse_hist, val_mse_hist, val_rmse_hist, val_mae_hist, val_mape_hist


if __name__ == "__main__":
    main()
