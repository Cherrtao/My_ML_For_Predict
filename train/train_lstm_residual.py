# train/train_baseline_lstm_residual.py
"""
残差版 Baseline LSTM 训练脚本（带 RMSE / MAPE 指标）

思路：
- 仍然只用 Main 节点的历史特征做预测；
- 先用一个“naive baseline”：直接把过去 24h 的 main_power
  当作对未来 24h 的预测；
- LSTM 学的是“残差”： r(t) = y_true(t) - y_naive(t)
- 损失函数在残差空间上计算 MSE；
- 评估阶段额外反标准化到 kW，计算 RMSE 和 MAPE。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config.dataset_config import TRAIN_CSV, VAL_CSV
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from config.model_config import NODE_INDEX, NODE_INPUT_DIMS
from models.baseline_lstm import BaselineLSTM


# ================== 一些超参数 ==================
T_IN = 288       # 过去 24h（5min 粒度）
T_OUT = 288      # 预测未来 24h
BATCH_SIZE = 32

HIDDEN_DIM = 32  # 和你现在常用配置对齐：h32 + 2-layer + dropout
NUM_LAYERS = 2
DROPOUT = 0.3

LR = 3e-4
WEIGHT_DECAY = 5e-5
NUM_EPOCHS = 40


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
    t_out: int,
):
    """
    单个 epoch 的训练（残差模式）：
      - naive_y: 复制过去 T_out 个 main_power；
      - target_residual = y_true - naive_y；
      - LSTM 输出 pred_residual，loss 在残差上算 MSE。
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

        # 1️⃣ 只取 Main 节点特征送给 LSTM
        X_main = X[:, :, main_idx, :main_dim]   # (B, T_in, F_main)

        # 2️⃣ naive baseline：复制过去 T_out 个 main_power（标准化空间）
        #    约定 X_main 的第 0 维是 main_power（和 model_config 保持一致）
        naive_y = X_main[:, -t_out:, 0]         # (B, T_out)

        # 3️⃣ 残差标签
        target_residual = y - naive_y           # (B, T_out)

        optimizer.zero_grad()
        pred_residual = model(X_main)           # (B, T_out)
        loss = criterion(pred_residual, target_residual)
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
    t_out: int,
    scaler=None,
):
    """
    验证集评估（残差空间 MSE + 真实空间 RMSE / MAPE）。

    返回：
      - mse_res   : 残差空间 MSE（标准化）
      - rmse_kw   : 还原到 kW 后的 RMSE
      - mape      : 还原到 kW 后的 MAPE（0~1）
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    # 为 MAPE 统计 |e|/|y|
    total_ape = 0.0
    n_points = 0

    main_idx = NODE_INDEX["Main"]
    main_dim = NODE_INPUT_DIMS["Main"]

    main_mean = None
    main_std = None
    if scaler is not None:
        # scaler["mean"] / scaler["std"] 一般是 pandas.Series
        main_mean = float(scaler["mean"]["main_power"])
        main_std = float(scaler["std"]["main_power"])

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)   # 标准化后的 main_power

        X_main = X[:, :, main_idx, :main_dim]
        naive_y = X_main[:, -t_out:, 0]         # baseline（标准化空间）

        target_residual = y - naive_y

        pred_residual = model(X_main)
        loss = criterion(pred_residual, target_residual)

        bs = X.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        # ===== 反标准化，计算真实空间 RMSE / MAPE =====
        if (main_mean is not None) and (main_std is not None):
            # 标准化空间中的真实值和预测值
            true_y_norm = y                      # (B, T_out)
            pred_y_norm = naive_y + pred_residual

            # 反标准化 -> kW
            true_y_kw = true_y_norm * main_std + main_mean
            pred_y_kw = pred_y_norm * main_std + main_mean

            eps = 1e-3
            ape = (pred_y_kw - true_y_kw).abs() / (true_y_kw.abs() + eps)

            total_ape += ape.sum().item()
            n_points += ape.numel()

    mse_res = total_loss / max(n_samples, 1)

    rmse_kw = None
    mape = None
    if main_std is not None:
        # 残差 MSE 等价于 (ŷ_norm - y_norm) 的 MSE
        rmse_kw = (mse_res ** 0.5) * main_std
    if n_points > 0:
        mape = total_ape / n_points

    return mse_res, rmse_kw, mape


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ====== 1. 数据集 & DataLoader ======
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

    # ====== 2. 模型 / 损失 / 优化器 ======
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

    # ====== 3. 训练循环（记录残差 MSE & RMSE/MAPE 曲线） ======
    best_val_loss = float("inf")
    best_state = None

    train_losses = []       # train 残差 MSE
    val_losses = []         # val 残差 MSE
    val_rmse_list = []      # val RMSE (kW)
    val_mape_list = []      # val MAPE (0~1)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_res_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, T_OUT
        )
        val_res_loss, val_rmse_kw, val_mape = eval_one_epoch(
            model, val_loader, criterion, device, T_OUT, scaler=scaler
        )

        train_losses.append(train_res_loss)
        val_losses.append(val_res_loss)
        val_rmse_list.append(val_rmse_kw)
        val_mape_list.append(val_mape)

        mape_pct = val_mape * 100 if val_mape is not None else None

        msg = (
            f"[Epoch {epoch:03d}] "
            f"train_res_MSE={train_res_loss:.4f}  "
            f"val_res_MSE={val_res_loss:.4f}  "
        )
        if val_rmse_kw is not None:
            msg += f"val_RMSE_kW={val_rmse_kw:.3f}  "
        if mape_pct is not None:
            msg += f"val_MAPE={mape_pct:.2f}%  "
        print(msg)

        if val_res_loss < best_val_loss:
            best_val_loss = val_res_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # ====== 4. 保存最优模型 & 画 loss 曲线 ======
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        ckpt_path = ckpt_dir / "baseline_lstm_residual_best.pt"
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
                    "residual_mode": True,
                    "best_val_res_MSE": best_val_loss,
                    # 下面这两个仅供参考（是训练过程中某一 epoch 的值）
                    "last_epoch_val_RMSE_kW": val_rmse_list[-1],
                    "last_epoch_val_MAPE": val_mape_list[-1],
                },
            },
            ckpt_path,
        )
        print(
            f"Saved best residual Baseline LSTM model to {ckpt_path}, "
            f"best val_res_MSE={best_val_loss:.4f}"
        )
    else:
        print("No best model saved.")

    # ====== 5. loss 曲线（残差 MSE） ======
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train residual MSE")
    plt.plot(epochs, val_losses, marker="s", label="Val residual MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (residual space)")
    plt.title(
        f"Residual Baseline LSTM "
        f"(hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, dropout={DROPOUT})"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    curve_path = ckpt_dir / "baseline_lstm_residual_loss_curve.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to: {curve_path}")

    # ✅ 把4条曲线返回给 Notebook（和 Graph / NoGraph 脚本对齐）
    return train_losses, val_losses, val_rmse_list, val_mape_list


if __name__ == "__main__":
    main()
