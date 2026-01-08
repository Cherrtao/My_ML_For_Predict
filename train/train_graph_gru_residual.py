# train_graph_gru_residual.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.dataset_config import TRAIN_CSV, VAL_CSV
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from config.model_config import NODE_INPUT_DIMS, NODE_INDEX
from models.graph_gru import GraphGRU


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
    model,
    loader,
    optimizer,
    criterion,
    device,
    t_out: int,
):
    """
    单个 epoch 的训练（残差模式）：
      - naive_y: 过去 T_out 个 main_power（直接复制）
      - target_residual = y_true - naive_y
      - 模型输出 pred_residual，loss 在残差上算（SmoothL1）
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    main_idx = NODE_INDEX["Main"]

    for X, y in loader:
        # X: (B, T_in, N, F_max)
        # y: (B, T_out) —— 标准化后的 main_power
        X = X.to(device)
        y = y.to(device)

        # naive baseline（标准化空间）
        main_seq = X[:, :, main_idx, :]     # (B, T_in, F_max)
        naive_y = main_seq[:, -t_out:, 0]   # (B, T_out) —— 第 0 维是 main_power

        # 残差标签
        target_residual = y - naive_y       # (B, T_out)

        optimizer.zero_grad()
        pred_residual = model(X)            # (B, T_out)
        loss = criterion(pred_residual, target_residual)
        loss.backward()
        # 梯度裁剪，避免偶发爆炸导致曲线抖得厉害
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = X.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    criterion,
    device,
    t_out: int,
    scaler=None,
):
    """
    验证集上评估一个 epoch。

    返回：
      - loss_res : 残差空间平均 loss（SmoothL1）
      - rmse_kw  : 还原到 kW 后的 RMSE
      - mape     : 还原到 kW 后的 MAPE（0~1）
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    # 真实空间 RMSE / MAPE 相关
    total_se_kw = 0.0
    total_ape = 0.0
    n_points = 0

    main_idx = NODE_INDEX["Main"]

    main_mean = None
    main_std = None
    if scaler is not None:
        main_mean = float(scaler["mean"]["main_power"])
        main_std = float(scaler["std"]["main_power"])

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)   # 标准化后的 main_power

        main_seq = X[:, :, main_idx, :]
        naive_y = main_seq[:, -t_out:, 0]   # 标准化 baseline

        target_residual = y - naive_y

        pred_residual = model(X)
        loss = criterion(pred_residual, target_residual)

        bs = X.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        # ===== 还原到物理空间，计算 RMSE / MAPE =====
        if (main_mean is not None) and (main_std is not None):
            # 标准化空间中的真实值 / 预测值
            true_y_norm = y                          # (B, T_out)
            pred_y_norm = naive_y + pred_residual    # (B, T_out)

            # 反标准化 -> kW
            true_y_kw = true_y_norm * main_std + main_mean
            pred_y_kw = pred_y_norm * main_std + main_mean

            err_kw = pred_y_kw - true_y_kw
            se_kw = (err_kw ** 2)

            total_se_kw += se_kw.sum().item()
            n_points += se_kw.numel()

            eps = 1e-3
            ape = err_kw.abs() / (true_y_kw.abs() + eps)

            total_ape += ape.sum().item()

    # 残差空间平均 loss（SmoothL1）
    loss_res = total_loss / max(n_samples, 1)

    # 真实空间 RMSE / MAPE
    rmse_kw = None
    mape = None
    if n_points > 0:
        rmse_kw = (total_se_kw / n_points) ** 0.5
        mape = total_ape / n_points

    return loss_res, rmse_kw, mape


def main():
    """
    残差版 Graph-GRU 训练入口（方案 A：偏稳的超参）。
    """
    # ====== 超参数（方案 A） ======
    T_IN = 288
    T_OUT = 288

    BATCH_SIZE = 64

    HIDDEN_DIM = 48
    NUM_LAYERS = 1
    DROPOUT = 0.2

    LR = 5e-5
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100

    # 学习率衰减 & 早停
    LR_DECAY_EPOCHS = (50, 80)
    LR_DECAY_GAMMA = 0.5
    PATIENCE = 10

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

    max_feat_dim = max(NODE_INPUT_DIMS.values())
    print("Max feature dim (F_max):", max_feat_dim)

    # ====== 2. 模型 / 损失 / 优化器 ======
    model = GraphGRU(
        input_dim=max_feat_dim,
        hidden_dim=HIDDEN_DIM,
        t_out=T_OUT,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        main_idx=NODE_INDEX["Main"],
    ).to(device)

    # 用 SmoothL1 比 MSE 更抗噪声一些
    criterion = nn.SmoothL1Loss(beta=1.0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0

    # 记录历史曲线
    train_history = []      # 残差 loss（train）
    val_history = []        # 残差 loss（val）
    val_rmse_history = []   # 真实空间 RMSE (kW)
    val_mape_history = []   # 真实空间 MAPE (0~1)

    # ====== 3. 训练循环 ======
    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- 手动学习率衰减 ----
        if epoch in LR_DECAY_EPOCHS:
            for g in optimizer.param_groups:
                g["lr"] *= LR_DECAY_GAMMA
            print(f"[LR] decayed to {optimizer.param_groups[0]['lr']:.1e}")

        train_res_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, T_OUT
        )
        val_res_loss, val_rmse_kw, val_mape = eval_one_epoch(
            model, val_loader, criterion, device, T_OUT, scaler=scaler
        )

        train_history.append(train_res_loss)
        val_history.append(val_res_loss)
        val_rmse_history.append(val_rmse_kw)
        val_mape_history.append(val_mape)

        mape_pct = val_mape * 100 if val_mape is not None else None
        msg = (
            f"[Epoch {epoch:03d}] "
            f"train_res_loss={train_res_loss:.4f}  "
            f"val_res_loss={val_res_loss:.4f}  "
        )
        if val_rmse_kw is not None:
            msg += f"val_RMSE_kW={val_rmse_kw:.3f}  "
        if mape_pct is not None:
            msg += f"val_MAPE={mape_pct:.2f}%  "
        msg += f"lr={optimizer.param_groups[0]['lr']:.1e}"
        print(msg)

        # ---- 记录 best 模型 ----
        if val_res_loss < best_val_loss:
            best_val_loss = val_res_loss
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch

        # ---- 早停：val 一直没变好就停 ----
        if epoch - best_epoch >= PATIENCE:
            print(f"[EarlyStop] epoch={epoch}, best_epoch={best_epoch}")
            break

    # ====== 4. 保存最优模型 ======
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
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
                "residual_mode": True,
                "graph_mode": True,
            },
            "graph_gru_residual_best.pt",
        )
        print(
            f"Saved best residual Graph-GRU model to graph_gru_residual_best.pt, "
            f"best_epoch={best_epoch}, best_val_res_loss={best_val_loss:.4f}"
        )
    else:
        print("No best model saved.")

    # ✅ 把曲线返回给 Notebook：前两个是残差 loss，后两个是真实空间指标
    return train_history, val_history, val_rmse_history, val_mape_history


if __name__ == "__main__":
    main()
