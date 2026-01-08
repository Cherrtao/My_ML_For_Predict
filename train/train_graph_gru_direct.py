# train/train_graph_gru_direct.py
"""
Graph-GRU 直接预测主功率序列（主趋势） 的训练脚本

- 输入：GraphSequenceDataset 提供的 (X, y)
  - X: (B, T_in, N, F_max)
  - y: (B, T_out)  -> 未来 24h 的 5min 主功率（已标准化）

- 模型：使用 GraphGRU，利用整张图的时序特征，直接输出主节点的 y_hat
- 损失：标准化空间 MSE
- 评估：同时给出反标准化后的 RMSE / MAE / MAPE
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.dataset_config import TRAIN_CSV, VAL_CSV
from config.model_config import NODE_INPUT_DIMS, NODE_INDEX
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from models.graph_gru import GraphGRU


# ================== 超参数 ==================
T_IN = 288          # 过去 24h（5min 粒度）
T_OUT = 288         # 未来 24h（5min 序列）

BATCH_SIZE = 32

HIDDEN_DIM = 48
NUM_LAYERS = 2
DROPOUT = 0.2

LR = 5e-4           # 稍微稳一点
WEIGHT_DECAY = 5e-5
NUM_EPOCHS = 20

# 早停相关
PATIENCE = 8        # 连续 PATIENCE 轮没有明显提升就停
MIN_DELTA = 1e-3    # RMSE 至少提升这么多才算“有进步”


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
    model: GraphGRU,
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

    for X, y in loader:
        # X: (B, T_in, N, F_max)
        # y: (B, T_out) —— 标准化后的 main_power
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(X)              # (B, T_out)，GraphGRU 直接输出主节点预测

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
    model: GraphGRU,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler,
):
    """
    验证集评估（直接预测）：返回标准化 MSE + 真实空间 RMSE/MAE/MAPE
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    total_se_kw = 0.0   # Σ e^2 (kW^2)
    total_ae_kw = 0.0   # Σ |e|  (kW)
    total_ape = 0.0     # Σ |e| / |y|
    n_points = 0

    main_mean = float(scaler["mean"]["main_power"])
    main_std = float(scaler["std"]["main_power"])

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)          # 标准化后的 main_power

        y_hat = model(X)          # (B, T_out)

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
    max_feat_dim = max(NODE_INPUT_DIMS.values())
    print("Max feature dim (F_max):", max_feat_dim)

    model = GraphGRU(
        input_dim=max_feat_dim,
        hidden_dim=HIDDEN_DIM,
        t_out=T_OUT,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        main_idx=NODE_INDEX["Main"],
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
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    train_mse_hist = []
    val_mse_hist = []
    val_rmse_hist = []
    val_mae_hist = []
    val_mape_hist = []

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        # 你也可以加一点手动 LR 衰减
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

        # ---- 早停逻辑：按 Val RMSE ----
        if val_rmse_kw < best_val_rmse - MIN_DELTA:
            best_val_rmse = val_rmse_kw
            best_epoch = epoch
            epochs_no_improve = 0

            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            save_path = ckpt_dir / "graph_gru_direct_best.pt"
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "scaler_mean": scaler["mean"],
                    "scaler_std": scaler["std"],
                    "t_in": T_IN,
                    "t_out": T_OUT,
                    "input_dim": max_feat_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "graph_mode": True,
                    "residual_mode": False,
                },
                save_path,
            )
            print(f"  -> New best model saved to {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(
                f"[EarlyStop] epoch={epoch}, best_epoch={best_epoch}, "
                f"best_val_RMSE={best_val_rmse:.3f} kW"
            )
            break

    print("Training finished. Best Val RMSE (kW):", best_val_rmse)

    # 方便 notebook 可视化
    return train_mse_hist, val_mse_hist, val_rmse_hist, val_mae_hist, val_mape_hist


if __name__ == "__main__":
    main()
