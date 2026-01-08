# train/optuna_graph_gru_direct.py
"""
Optuna 超参搜索：Graph-GRU 直接预测主功率序列 (24h -> 24h)

- 固定：
    T_IN = 288, T_OUT = 288
- 搜索范围（略收窄，避免显存爆掉）：
    hidden_dim   : [32, 48, 64]        # 不再放到 80/96
    num_layers   : [1, 2]              # 3 层对你这个数据其实意义不大，且更耗显存
    dropout      : [0.0, 0.4]
    lr           : [2e-4, 1.5e-3] (log 采样)
    weight_decay : [1e-6, 5e-4]   (log 采样)
    batch_size   : [16, 32]            # 去掉 48
- 目标：最小化 “Val RMSE * (1 + 0.3 * norm_gap)”，
       其中 norm_gap 反映 Train/Val MSE 的相对差距，抑制过拟合。
"""

from __future__ import annotations

import gc
from typing import Tuple

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.dataset_config import TRAIN_CSV, VAL_CSV
from config.model_config import NODE_INPUT_DIMS, NODE_INDEX
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from models.graph_gru import GraphGRU

# ================== 固定配置 ==================
T_IN = 288      # 过去 24h（5min 粒度）
T_OUT = 288     # 未来 24h（5min 序列）

MAX_EPOCHS = 30     # 每个 trial 最多跑 30 轮

USE_GPU_FOR_OPTUNA = True  # 如果还是老出 CUDA 问题，可以改成 False 强制用 CPU
DEVICE = torch.device("cuda" if (USE_GPU_FOR_OPTUNA and torch.cuda.is_available()) else "cpu")
print("[optuna_graph_gru_direct] DEVICE =", DEVICE)

# 共享缓存，避免每个 trial 重读 CSV
_CACHED_DATA = None  # (train_ds, val_ds, scaler)


# ================== 数据集构建 ==================
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


def create_dataloaders(batch_size: int):
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


# ================== 训练 / 验证 ==================
def train_one_epoch(
    model: GraphGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(X)  # (B, T_out)

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
    model: GraphGRU,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler,
):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    total_se_kw = 0.0
    total_ae_kw = 0.0
    total_ape = 0.0
    n_points = 0

    main_mean = float(scaler["mean"]["main_power"])
    main_std = float(scaler["std"]["main_power"])

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        y_hat = model(X)  # (B, T_out)

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
def objective(trial: optuna.Trial) -> float:
    # 每个 trial 开始前清一下显存（尤其在 Win + 单卡环境有用）
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # ---- 1) 搜索空间（略收窄，避免 OOM）----
    hidden_dim = trial.suggest_categorical(
        "hidden_dim", [32, 48, 64]
    )
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    lr = trial.suggest_float("lr", 2e-4, 1.5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)

    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32]
    )

    # ---- 2) 数据 / 模型 / 优化器 ----
    train_loader, val_loader, scaler = create_dataloaders(batch_size)

    max_feat_dim = max(NODE_INPUT_DIMS.values())

    model = GraphGRU(
        input_dim=max_feat_dim,
        hidden_dim=hidden_dim,
        t_out=T_OUT,
        num_layers=num_layers,
        dropout=dropout,
        main_idx=NODE_INDEX["Main"],
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    lr_milestones = {
        int(MAX_EPOCHS * 0.5),
        int(MAX_EPOCHS * 0.75),
    }

    best_val_rmse = float("inf")
    best_epoch = 0
    last_train_mse = None
    last_val_mse = None

    PATIENCE = 10
    MIN_DELTA = 1e-3
    no_improve = 0

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
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

            if val_rmse_kw < best_val_rmse - MIN_DELTA:
                best_val_rmse = val_rmse_kw
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            trial.report(val_rmse_kw, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if no_improve >= PATIENCE:
                break

    except RuntimeError as e:
        # 若出现 CUDA 相关错误，清空显存并直接剪枝该 trial
        msg = str(e)
        if "CUDA" in msg or "cuda" in msg:
            print(f"[Trial {trial.number}] caught CUDA error, prune this trial:", msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()
        else:
            # 其他错误正常抛出去
            raise

    # ---- 3) 目标：Val RMSE + 轻微过拟合惩罚 ----
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


# ================== 主入口 ==================
def main():
    N_TRIALS = 30

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
    )

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="graph_gru_direct_study",
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
