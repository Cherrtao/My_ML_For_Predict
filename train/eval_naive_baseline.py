# train/eval_naive_baseline.py
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from config.dataset_config import TRAIN_CSV, VAL_CSV
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from config.model_config import NODE_INDEX


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


@torch.no_grad()
def eval_naive_on_loader(
    loader: DataLoader,
    scaler,
    device: torch.device,
    t_out: int,
):
    """
    对一个 DataLoader 评估 naive baseline：
      - 预测 = 复制过去 T_OUT 个 main_power（标准化空间）
      - 计算：MSE_norm, RMSE_kW, MAE_kW, MAPE
    """
    main_idx = NODE_INDEX["Main"]

    # 从 scaler 里取出 main_power 的 mean / std（反标准化用）
    main_mean = float(scaler["mean"]["main_power"])
    main_std = float(scaler["std"]["main_power"])

    total_se_norm = 0.0   # 标准化空间平方误差和
    total_se_kw = 0.0     # kW 空间平方误差和
    total_ae_kw = 0.0     # kW 空间绝对误差和
    total_ape = 0.0       # 绝对百分比误差和
    n_points = 0          # 有效点数（B * T_OUT 累加）

    for X, y in loader:
        # X: (B, T_in, N, F_max)   —— 已经是标准化后的特征
        # y: (B, T_out)            —— 标准化后的 main_power
        X = X.to(device)
        y = y.to(device)

        B, T_out_batch = y.shape
        assert T_out_batch == t_out, "y 的长度必须等于 t_out"

        # 取 Main 节点的历史序列，复制最后 T_OUT 个点作为预测
        main_seq = X[:, :, main_idx, :]           # (B, T_in, F_max)
        naive_y_norm = main_seq[:, -t_out:, 0]    # (B, T_out)

        # ---------- 标准化空间 MSE ----------
        err_norm = naive_y_norm - y              # (B, T_out)
        se_norm = err_norm ** 2
        total_se_norm += se_norm.sum().item()

        # ---------- 反标准化到 kW ----------
        true_y_kw = y * main_std + main_mean
        pred_y_kw = naive_y_norm * main_std + main_mean

        err_kw = pred_y_kw - true_y_kw
        se_kw = err_kw ** 2
        ae_kw = err_kw.abs()
        ape = ae_kw / (true_y_kw.abs() + 1e-3)   # 避免除零

        total_se_kw += se_kw.sum().item()
        total_ae_kw += ae_kw.sum().item()
        total_ape += ape.sum().item()
        n_points += B * t_out

    # ===== 聚合指标 =====
    mse_norm = total_se_norm / n_points
    rmse_kw = (total_se_kw / n_points) ** 0.5
    mae_kw = total_ae_kw / n_points
    mape = total_ape / n_points   # 0~1

    return {
        "MSE_norm": mse_norm,
        "RMSE_kW": rmse_kw,
        "MAE_kW": mae_kw,
        "MAPE": mape,
    }


def main():
    """
    评估 naive baseline：
      - 输入：过去 24h 主功率
      - 预测：直接复制作为未来 24h 主功率预测
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    T_IN = 288   # 过去 24h（5min 间隔）
    T_OUT = 288  # 未来 24h
    BATCH_SIZE = 64

    # ===== 1. 数据集 / DataLoader =====
    train_ds, val_ds, scaler = build_datasets(T_IN, T_OUT)
    print("Train samples:", len(train_ds))
    print("Val   samples:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ===== 2. 评估 Train / Val 上的 naive baseline =====
    print("\n=== Evaluate naive baseline (copy last 24h as forecast) ===")

    train_metrics = eval_naive_on_loader(
        train_loader, scaler, device, T_OUT
    )
    val_metrics = eval_naive_on_loader(
        val_loader, scaler, device, T_OUT
    )

    def fmt(metrics):
        return (
            f"MSE_norm={metrics['MSE_norm']:.4f}, "
            f"RMSE_kW={metrics['RMSE_kW']:.2f}, "
            f"MAE_kW={metrics['MAE_kW']:.2f}, "
            f"MAPE={metrics['MAPE'] * 100:.2f}%"
        )

    print("Train:", fmt(train_metrics))
    print("Val  :", fmt(val_metrics))

    return train_metrics, val_metrics


if __name__ == "__main__":
    main()
