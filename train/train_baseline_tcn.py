# train/train_baseline_tcn.py
"""
TCN Baseline 训练脚本（单任务：5min 主功率预测）

- 输入：Main 节点的历史特征 (B, T_in, F_main)
- 输出：未来 24h (288 步) 的 main_power（标准化空间）
- 损失：标准化空间 SmoothL1（对异常点更稳）
- 评估：额外计算还原到 kW 后的 RMSE 和 MAPE
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.dataset_config import TRAIN_CSV, VAL_CSV
from config.model_config import NODE_INDEX, NODE_INPUT_DIMS
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from models.baseline_tcn import BaselineTCN


# ================== 超参数 ==================
T_IN = 288          # 过去 24h（5min 粒度）
T_OUT = 288         # 未来 24h（5min 序列）

BATCH_SIZE = 32     # 稍小一点，让梯度更“平滑”

CHANNELS = 32       # 通道数
NUM_LEVELS = 7      # 感受野 ~ 255 步
DROPOUT = 0.2       # 适中正则

# ⭐ 关键：降低基础学习率
LR = 3e-4
WEIGHT_DECAY = 5e-5
NUM_EPOCHS = 40     # 允许更长训练

# 学习率衰减 & 早停（注意：和原来不一样）
LR_DECAY_EPOCHS = (40, 60)   # 先在 3e-4 上跑久一点，再衰减
LR_DECAY_GAMMA = 0.5
PATIENCE = 20                 # 给更大的早停耐心


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
    model: BaselineTCN,
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

        # 取 Main 节点特征
        X_main = X[:, :, main_idx, :main_dim]   # (B, T_in, F_main)

        optimizer.zero_grad()
        y_hat = model(X_main)                   # (B, T_out)

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
    model: BaselineTCN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler,
):
    """
    验证集评估：

    返回：
      - mse_norm : 标准化空间 MSE
      - rmse_kw  : 反标准化后的 RMSE（kW）
      - mape     : 反标准化后的 MAPE（0~1）
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    # 真实空间 RMSE / MAPE 累计
    total_se_kw = 0.0  # Σ e^2 (kW^2)
    total_ape = 0.0    # Σ |e| / |y|
    n_points = 0

    main_idx = NODE_INDEX["Main"]
    main_dim = NODE_INPUT_DIMS["Main"]

    # 从 scaler 中取出 main_power 的 mean/std
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

        # ===== 反标准化到 kW，计算 RMSE + MAPE =====
        # y, y_hat 是标准化空间
        true_y_kw = y * main_std + main_mean      # (B, T_out)
        pred_y_kw = y_hat * main_std + main_mean  # (B, T_out)

        err_kw = pred_y_kw - true_y_kw
        se_kw = (err_kw ** 2)

        total_se_kw += se_kw.sum().item()
        n_points += se_kw.numel()

        eps = 1e-3
        ape = err_kw.abs() / (true_y_kw.abs() + eps)
        total_ape += ape.sum().item()

    mse_norm = total_loss / max(n_samples, 1)
    rmse_kw = (total_se_kw / max(n_points, 1)) ** 0.5
    mape = total_ape / max(n_points, 1)

    return mse_norm, rmse_kw, mape


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

    model = BaselineTCN(
        input_dim=main_dim,
        out_len=T_OUT,
        channels=CHANNELS,
        num_levels=NUM_LEVELS,
        kernel_size=3,
        dropout=DROPOUT,
    ).to(device)

    print(model)

    # ⭐ 损失改为 SmoothL1Loss（对异常点更稳）
    criterion = nn.SmoothL1Loss(beta=1.0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ---- 学习率衰减 & 早停配置 ----
    global LR_DECAY_EPOCHS, LR_DECAY_GAMMA, PATIENCE

    best_val_mse = float("inf")
    best_state = None
    best_epoch = 0

    train_mse_hist = []
    val_mse_hist = []
    val_rmse_hist = []
    val_mape_hist = []

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- 手动学习率衰减 ----
        if epoch in LR_DECAY_EPOCHS:
            for g in optimizer.param_groups:
                g["lr"] *= LR_DECAY_GAMMA
            print(f"[LR] decayed to {optimizer.param_groups[0]['lr']:.1e}")

        # ---- Train / Val ----
        train_mse = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_mse, val_rmse_kw, val_mape = eval_one_epoch(
            model, val_loader, criterion, device, scaler
        )

        train_mse_hist.append(train_mse)
        val_mse_hist.append(val_mse)
        val_rmse_hist.append(val_rmse_kw)
        val_mape_hist.append(val_mape)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train MSE={train_mse:.4f} | "
            f"Val MSE={val_mse:.4f}, "
            f"Val RMSE={val_rmse_kw:.3f} kW, "
            f"Val MAPE={val_mape*100:.2f}%"
        )

        # ---- 保存最优模型（按 Val MSE）----
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            save_path = ckpt_dir / "baseline_tcn_best.pt"
            torch.save(
                {
                    "model_state_dict": best_state,
                    "scaler": scaler,
                    "config": {
                        "T_IN": T_IN,
                        "T_OUT": T_OUT,
                        "main_dim": main_dim,
                        "channels": CHANNELS,
                        "num_levels": NUM_LEVELS,
                        "dropout": DROPOUT,
                    },
                },
                save_path,
            )
            print(f"  -> New best model saved to {save_path}")

        # ---- 早停：val 连续 PATIENCE 轮没变好就停 ----
        if epoch - best_epoch >= PATIENCE:
            print(f"[EarlyStop] epoch={epoch}, best_epoch={best_epoch}")
            break

    print("Training finished. Best Val MSE:", best_val_mse)

    # ✅ 返回给 notebook 画曲线用
    return train_mse_hist, val_mse_hist, val_rmse_hist, val_mape_hist


if __name__ == "__main__":
    main()
