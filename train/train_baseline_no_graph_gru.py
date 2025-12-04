# train_nograph_gru.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt  # 用于训练后画 loss 曲线

from config.dataset_config import TRAIN_CSV, VAL_CSV
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv  # 导入数据集类和标准化函数
from models.baseline_no_graph_gru import NoGraphGRU  # 导入 No-Graph GRU 模型
from config.model_config import NODE_INPUT_DIMS  # 导入各节点输入维度，用于计算 F_max


def build_datasets(t_in: int, t_out: int):
    """
    构建 train / val 数据集，并在训练集上拟合标准化参数。
    t_in:  输入窗口长度
    t_out: 输出窗口长度
    """
    # 1. 在训练集上拟合 mean/std,用于后续标准化
    scaler = fit_scaler_from_csv(str(TRAIN_CSV))

    # 2. 构建 Dataset（训练集 / 验证集都用同一个 scaler）
    train_ds = GraphSequenceDataset(
        csv_path=str(TRAIN_CSV),
        t_in=t_in,
        t_out=t_out,
        feature_scaler=scaler,
        fit_scaler=False,
    )

    # 3. 构造验证集 Dataset，同样使用训练集的 scaler，避免数据泄漏
    val_ds = GraphSequenceDataset(
        csv_path=str(VAL_CSV),
        t_in=t_in,
        t_out=t_out,
        feature_scaler=scaler,
        fit_scaler=False,
    )
    # 返回训练集 Dataset， 验证集 Dataset 和该 scaler,用于后续保存
    return train_ds, val_ds, scaler


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx: int, num_epochs: int):
    """
    训练一个epoch（遍历一遍训练集）
    :param model:     要训练的模型
    :param loader:    训练 DataLoader
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param device:    运行设备
    :param epoch_idx: 当前是第几个 epoch（从 1 开始）
    :param num_epochs:总共多少个 epoch
    """
    model.train()        # 将模型切换到训练模式(启用 dropout 等)
    total_loss = 0.0     # 累积总损失
    n_samples = 0        # 累积样本总量

    num_batches = len(loader)  # 本 epoch 一共有多少个 batch

    for batch_idx, (X, y) in enumerate(loader):
        # X: (B, T_in, N, F_max)
        # y: (B, T_out)
        X = X.to(device)  # 将输入数据搬到 GPU/CPU
        y = y.to(device)  # 将标签数据搬到 GPU/CPU

        optimizer.zero_grad()          # 清空上一轮的梯度
        y_hat = model(X)               # (B, T_out)，前向传播
        loss = criterion(y_hat, y)     # 计算损失（例如 MSE）
        loss.backward()                # 反向传播，计算梯度
        optimizer.step()               # 更新模型参数

        batch_size = X.size(0)         # 当前 batch 中的样本数
        total_loss += loss.item() * batch_size  # 累加加权损失(按样本数加权)
        n_samples += batch_size                 # 累加样本数

        # —— 实时打印：每 10 个 batch 打一次当前 batch 的 loss（也可以改成 1，想看得更细的话）——
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or (batch_idx + 1) == num_batches:
            print(
                f"  [Epoch {epoch_idx:03d}/{num_epochs:03d}] "
                f"Batch {batch_idx + 1:03d}/{num_batches:03d} "
                f"loss={loss.item():.4f}"
            )

    # 返回平均损失：总损失 / 样本数（防止除以 0）
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    """
    在验证集上评估一个epoch
    :param model:   要评估的模型
    :param loader:  验证集 DataLoader
    :param criterion: 损失函数
    :param device:  设备（CPU/GPU）
    """
    model.eval()         # 切换到评估模式（禁用 dropout、BN 用 running mean 等）
    total_loss = 0.0     # 累积验证损失
    n_samples = 0        # 累积样本数

    for X, y in loader:
        X = X.to(device)  # 输入搬到设备
        y = y.to(device)  # 标签搬到设备

        y_hat = model(X)            # 前向计算预测
        loss = criterion(y_hat, y)  # 计算损失

        batch_size = X.size(0)            # 当前 batch 的大小
        total_loss += loss.item() * batch_size  # 累积加权损失
        n_samples += batch_size                 # 累积样本数

    # 返回验证集平均损失
    return total_loss / max(n_samples, 1)


def main():
    """
    主函数：配置超参数 -> 构建数据集 -> 构建模型 -> 训练 & 验证 -> 保存最优模型并画 loss 曲线
    """
    # ====== 一些基础超参数（可以之后再调）======
    T_IN = 288      # 过去 24h （288*5min）
    T_OUT = 288     # 预测 24h
    BATCH_SIZE = 32  # 每个 batch 中的样本数
    HIDDEN_DIM = 64  # GRU 隐藏层维度 H
    NUM_LAYERS = 1   # GRU 层数（先从 1 层开始）
    DROPOUT = 0.0    # dropout 比例（单层时设为 0）
    LR = 1e-3        # 学习率
    WEIGHT_DECAY = 1e-5  # L2 正则（权重衰减）
    NUM_EPOCHS = 20  # 训练轮数（可以根据效果再调）

    # 选择设备：优先使用 GPU（cuda），否则退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ====== 1. 构建数据集 & DataLoader ======
    train_ds, val_ds, scaler = build_datasets(T_IN, T_OUT)  # 构建数据集并拿到 scaler
    # 打印训练集和验证集样本数量
    print("Train samples:", len(train_ds))
    print("Val   samples:", len(val_ds))

    # 构建训练集 DataLoader
    train_loader = DataLoader(
        train_ds,              # 训练集 Dataset
        batch_size=BATCH_SIZE, # 每个 batch 大小
        shuffle=True,          # 打乱数据顺序（训练时一般要打乱）
        num_workers=0,         # 数据加载子进程数，Windows 下用 0 最稳妥
        drop_last=True,        # 丢弃最后不足一个 batch 的样本（保证每个 batch 满的）
    )
    # 构建验证集 DataLoader
    val_loader = DataLoader(
        val_ds,                # 验证集 Dataset
        batch_size=BATCH_SIZE, # 每个 batch 大小（验证可以不 drop_last）
        shuffle=False,         # 验证集不需要打乱
        num_workers=0,         # 同样用 0
        drop_last=False,       # 保留最后一个可能不满的 batch
    )

    # 计算 F_max：所有节点输入维度中的最大值
    max_feat_dim = max(NODE_INPUT_DIMS.values())
    print("Max feature dim (F_max):", max_feat_dim)

    # ====== 2. 构建模型 / 损失函数 / 优化器 ======
    model = NoGraphGRU(
        input_dim=max_feat_dim,  # 输入特征维度 F_max
        hidden_dim=HIDDEN_DIM,   # GRU 隐藏维度
        t_out=T_OUT,             # 预测长度 T_out
        num_layers=NUM_LAYERS,   # GRU 层数
        dropout=DROPOUT,         # dropout 比例
    ).to(device)                 # 将模型搬到指定设备（CPU/GPU）

    criterion = nn.MSELoss()    # 使用均方误差作为回归任务的损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),     # 模型参数
        lr=LR,                  # 学习率
        weight_decay=WEIGHT_DECAY,  # L2 正则系数
    )

    best_val_loss = float("inf")  # 记录目前为止最好的验证集损失，初始设为正无穷
    best_state_dict = None        # 记录最优模型参数

    # ====== 记录每个 epoch 的 loss，用于画图 ======
    train_losses = []  # 记录每个 epoch 的训练集平均损失
    val_losses = []    # 记录每个 epoch 的验证集平均损失

    # ====== 3. 训练循环 ======
    for epoch in range(1, NUM_EPOCHS + 1):
        # 在训练集上训练一个 epoch（内部会打印若干 batch 的即时 loss）
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                     epoch_idx=epoch, num_epochs=NUM_EPOCHS)
        # 在验证集上评估一个 epoch
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)  # 记录当前 epoch 的训练损失
        val_losses.append(val_loss)      # 记录当前 epoch 的验证损失

        # 打印当前 epoch 的训练损失和验证损失
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        # 如果当前验证集损失更低，则更新“最佳模型”记录
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 将当前模型参数复制到 CPU 上保存，避免 device 相关问题
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # ====== 4. 保存最优模型 ======
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)  # 把模型权重替换为最优权重
        torch.save(
            {
                "model_state_dict": best_state_dict,  # 保存模型参数
                "scaler_mean": scaler["mean"],        # 保存标准化用的 mean
                "scaler_std": scaler["std"],          # 保存标准化用的 std
                "t_in": T_IN,                         # 保存输入窗口长度
                "t_out": T_OUT,                       # 保存预测窗口长度
                "input_dim": max_feat_dim,            # 保存 F_max
                "hidden_dim": HIDDEN_DIM,             # 保存 GRU 隐藏层维度
            },
            "nograph_gru_best.pt",  # 保存文件名
        )
        print(f"Saved best NoGraph GRU model to nograph_gru_best.pt, val_loss={best_val_loss:.4f}")
    else:
        print("No best model saved (maybe no data?).")

    # ====== 5. 画 loss 曲线 ======
    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss (NoGraph GRU)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve_nograph_gru.png", dpi=150)  # 保存到本地文件
    print("Saved loss curve to loss_curve_nograph_gru.png")
    # 如果你是在本地有图形界面的环境，也可以直接弹出窗口查看：
    plt.show()


if __name__ == "__main__":
    main()
