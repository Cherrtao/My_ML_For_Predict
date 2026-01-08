# train/optuna_lstm_single.py
"""
Optuna 超参搜索：Baseline LSTM（5min 主功率 24h->24h 预测）

- 复用 train_baseline_lstm_single.py 里的数据 & 训练逻辑
- 搜索范围围绕你当前这组超参（稍微收窄一点，提升稳定性）：
    T_IN = 288, T_OUT = 288 固定
    hidden_dim   : [32, 48, 64]
    num_layers   : 1~3
    dropout      : 0.1~0.4
    lr           : [2e-4, 1e-3]（log）
    weight_decay : [1e-6, 5e-4]（log）
    batch_size   : [16, 32]
- 目标：最小化“带轻微过拟合惩罚”的 Val RMSE
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any
import gc

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.dataset_config import TRAIN_CSV, VAL_CSV
from config.model_config import NODE_INDEX, NODE_INPUT_DIMS
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from models.baseline_lstm import BaselineLSTM


# ================== 固定配置 ==================
T_IN = 288          # 过去 24h（5min 粒度）
T_OUT = 288         # 未来 24h（5min 序列）

MAX_EPOCHS = 40     # 每个 trial 最多训练 40 轮

# 如果在 Optuna 里也经常遇到 CUDA 错误，可以把下面改成 False 让它只用 CPU
USE_GPU_FOR_OPTUNA = True
DEVICE = torch.device("cuda" if (USE_GPU_FOR_OPTUNA and torch.cuda.is_available()) else "cpu")
print("[optuna_lstm_single] DEVICE =", DEVICE)


# ================== 数据集构建（与原脚本一致） ==================
def build_datasets(t_in: int, t_out: int):
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


# ================== 复用的 train / eval 函数 ==================
def train_one_epoch(
    model: BaselineLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    main_idx = NODE_INDEX["Main"]
    main_dim = NODE_INPUT_DIMS["Main"]

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        X_main = X[:, :, main_idx, :main_dim]   # (B, T_in, main_dim)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(X_main)                  # (B, T_out)

        loss = criterion(y_hat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = X.size(0)
        total_loss += float(loss.item()) * bs
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
    返回：
      - mse_norm : 标准化空间 MSE
      - rmse_kw  : 反标准化后的 RMSE（kW）
      - mae_kw   : 反标准化后的 MAE（kW）
      - mape     : 反标准化后的 MAPE（0~1）
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    total_se_kw = 0.0
    total_ae_kw = 0.0
    total_ape = 0.0
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
        total_loss += float(loss.item()) * bs
        n_samples += bs

        true_y_kw = y * main_std + main_mean
        pred_y_kw = y_hat * main_std + main_mean

        err_kw = pred_y_kw - true_y_kw
        se_kw = err_kw ** 2
        ae_kw = err_kw.abs()

        total_se_kw += float(se_kw.sum().item())
        total_ae_kw += float(ae_kw.sum().item())
        n_points += se_kw.numel()

        eps = 1e-3
        ape = ae_kw / (true_y_kw.abs() + eps)
        total_ape += float(ape.sum().item())

    mse_norm = total_loss / max(n_samples, 1)
    rmse_kw = (total_se_kw / max(n_points, 1)) ** 0.5
    mae_kw = total_ae_kw / max(n_points, 1)
    mape = total_ape / max(n_points, 1)

    return mse_norm, rmse_kw, mae_kw, mape


# ================== Optuna 目标函数 ==================
_CACHED_DATA = None  # (train_ds, val_ds, scaler)


def create_dataloaders(batch_size: int):
    """为了避免每个 trial 重新读 CSV，把构建逻辑单独放出来。"""
    global _CACHED_DATA
    if _CACHED_DATA is None:
        train_ds, val_ds, scaler = build_datasets(T_IN, T_OUT)
        _CACHED_DATA = (train_ds, val_ds, scaler)
    else:
        train_ds, val_ds, scaler = _CACHED_DATA

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader, scaler


def objective(trial: optuna.Trial) -> float:
    # 每个 trial 开始前清一下 CUDA cache，防止长时间运行后显存碎片化
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # -------- 1) 搜索空间（略收窄）--------
    hidden_dim = trial.suggest_categorical(
        "hidden_dim", [32, 48, 64]
    )
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    lr = trial.suggest_float("lr", 2e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 5e-4, log=True
    )
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32]
    )

    # -------- 2) 数据、模型、优化器 --------
    train_loader, val_loader, scaler = create_dataloaders(batch_size)
    main_dim = NODE_INPUT_DIMS["Main"]

    model = BaselineLSTM(
        input_dim=main_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_len=T_OUT,
        dropout=dropout,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 学习率衰减点：50% 和 75%
    lr_milestones = {
        int(MAX_EPOCHS * 0.5),
        int(MAX_EPOCHS * 0.75),
    }

    best_val_rmse = float("inf")
    best_epoch = 0
    last_train_mse = None
    last_val_mse = None

    # ==== 早停配置 ====
    PATIENCE = 10
    no_improve = 0

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            # 手动 LR 衰减
            if epoch in lr_milestones:
                for g in optimizer.param_groups:
                    g["lr"] *= 0.3

            train_mse = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE
            )
            val_mse, val_rmse_kw, val_mae_kw, val_mape = eval_one_epoch(
                model, val_loader, criterion, DEVICE, scaler
            )

            last_train_mse = train_mse
            last_val_mse = val_mse

            # 追踪最优 RMSE + 早停计数
            if val_rmse_kw < best_val_rmse:
                best_val_rmse = val_rmse_kw
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            # 把当前 val_rmse 报告给 Optuna，用于剪枝
            trial.report(val_rmse_kw, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # ---- 早停：超过 PATIENCE 轮没提升就停 ----
            if no_improve >= PATIENCE:
                break

    except RuntimeError as e:
        msg = str(e)
        # 如果遇到 CUDA 类错误，清理显存并剪枝这个 trial，避免把整个 study 跑挂
        if "CUDA" in msg or "cuda" in msg:
            print(f"[Trial {trial.number}] caught CUDA error, prune this trial:", msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()
        else:
            raise

    # -------- 3) 目标：Val RMSE + 轻微过拟合惩罚 --------
    if last_train_mse is not None and last_val_mse is not None:
        gap = max(0.0, last_val_mse - last_train_mse)
        norm_gap = gap / (last_val_mse + 1e-6)
    else:
        norm_gap = 0.0

    penalty_factor = 1.0 + 0.3 * norm_gap
    objective_value = best_val_rmse * penalty_factor

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("last_train_mse", float(last_train_mse or 0.0))
    trial.set_user_attr("last_val_mse", float(last_val_mse or 0.0))
    trial.set_user_attr("penalty_factor", float(penalty_factor))

    return objective_value


# ================== 入口：运行搜索 ==================
def main():
    N_TRIALS = 30  # 可按时间/算力调整

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # 前几个 trial 不剪枝
        n_warmup_steps=5,    # 每个 trial 至少跑 5 轮
    )

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="baseline_lstm_single_study",
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n===== Optuna Search Finished =====")
    print("Best objective (Val RMSE with penalty):", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    attrs = study.best_trial.user_attrs
    print("Best trial extra info:")
    print(f"  best_epoch     : {attrs.get('best_epoch')}")
    print(f"  last_train_MSE : {attrs.get('last_train_mse')}")
    print(f"  last_val_MSE   : {attrs.get('last_val_mse')}")
    print(f"  penalty_factor : {attrs.get('penalty_factor')}")


if __name__ == "__main__":
    main()
