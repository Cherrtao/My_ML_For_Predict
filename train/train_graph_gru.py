# train/train_graph_gru.py
"""
训练 Graph-GRU 模型

- 使用 GraphSequenceDataset 提供的 X, y：
    X: (B, T_in, N, F_max)
    y: (B, T_out) = future main_power
- Graph-GRU 会：
    1) 每个时间步上做一层 GCN（用 ADJ_NORM 聚合 BF/Cold/Exhaust/Env/Main 之间的信息）
    2) 把每个时间步的节点嵌入 (N, G) 展平为 (N*G)
    3) 用 GRU 做时序编码，最后映射到未来 T_out 点的预测序列
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.dataset_config import TRAIN_CSV, VAL_CSV          # 训练/验证 CSV 路径
from config.model_config import NODE_INPUT_DIMS              # 每个节点输入维度，用来算 F_max
from data.graph_dataset import GraphSequenceDataset, fit_scaler_from_csv
from models.graph_gru import GraphGRU                        # 刚刚实现的 Graph-GRU 模型


def build_datasets(t_in: int, t_out: int):
    """
    构建 train / val 数据集，并在训练集上拟合标准化参数。

    参数
    ----
    t_in : int
        输入窗口长度（过去多少个 5min 点，例如 288）。
    t_out : int
        输出窗口长度（未来多少个 5min 点，例如 288）。

    返回
    ----
    train_ds : GraphSequenceDataset
        训练集 Dataset。
    val_ds   : GraphSequenceDataset
        验证集 Dataset。
    scaler   : Dict[str, pd.Series]
        标准化用的 mean/std。
    """
    # 1. 在训练集上拟合 mean/std（只做一次，避免数据泄漏）
    scaler = fit_scaler_from_csv(str(TRAIN_CSV))

    # 2. 构建训练集 Dataset，直接传入拟合好的 scaler
    train_ds = GraphSequenceDataset(
        csv_path=str(TRAIN_CSV),  # 训练 CSV 路径
        t_in=t_in,                # 输入长度 T_in
        t_out=t_out,              # 输出长度 T_out
        feature_scaler=scaler,    # 标准化参数
        fit_scaler=False,         # 不再在内部拟合
    )

    # 3. 构建验证集 Dataset，使用同一个 scaler
    val_ds = GraphSequenceDataset(
        csv_path=str(VAL_CSV),    # 验证 CSV 路径
        t_in=t_in,
        t_out=t_out,
        feature_scaler=scaler,
        fit_scaler=False,
    )

    # 返回训练集、验证集和 scaler
    return train_ds, val_ds, scaler


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    在训练集上训练一个 epoch。

    参数
    ----
    model : nn.Module
        要训练的 GraphGRU 模型。
    loader : DataLoader
        训练集 DataLoader。
    optimizer : torch.optim.Optimizer
        优化器（例如 Adam）。
    criterion : nn.Module
        损失函数（例如 MSELoss）。
    device : torch.device
        运行设备（CPU 或 GPU）。
    """
    model.train()          # 切换到训练模式（启用 dropout 等）
    total_loss = 0.0       # 累积损失
    n_samples = 0          # 累积样本数量

    # 遍历一个 epoch 中的所有 batch
    for X, y in loader:
        # X: (B, T_in, N, F_max)
        # y: (B, T_out)

        X = X.to(device)   # 把输入搬到 GPU / CPU
        y = y.to(device)   # 把标签搬到 GPU / CPU

        optimizer.zero_grad()   # 清空上一轮的梯度
        y_hat = model(X)        # 前向传播，得到预测 (B, T_out)
        loss = criterion(y_hat, y)  # 计算当前 batch 的损失
        loss.backward()         # 反向传播，计算梯度
        optimizer.step()        # 更新模型参数

        batch_size = X.size(0)  # 当前 batch 的样本数
        total_loss += loss.item() * batch_size  # 按样本数加权累积损失
        n_samples += batch_size                 # 累积样本数

    # 返回平均损失（防止 n_samples=0 时除 0）
    return total_loss / max(n_samples, 1)


@torch.no_grad()  # 表示该函数内部不需要梯度，自动关闭 autograd
def eval_one_epoch(model, loader, criterion, device):
    """
    在验证集上评估一个 epoch（不更新参数）。

    参数
    ----
    model : nn.Module
        已训练/正在训练的 GraphGRU 模型。
    loader : DataLoader
        验证集 DataLoader。
    criterion : nn.Module
        损失函数。
    device : torch.device
        设备。
    """
    model.eval()          # 切换到评估模式（禁用 dropout 等）
    total_loss = 0.0      # 累积验证损失
    n_samples = 0         # 累积样本数

    # 遍历验证集的所有 batch
    for X, y in loader:
        X = X.to(device)  # 输入搬到设备
        y = y.to(device)  # 标签搬到设备

        y_hat = model(X)               # 前向预测
        loss = criterion(y_hat, y)     # 计算损失

        batch_size = X.size(0)        # 当前 batch 的大小
        total_loss += loss.item() * batch_size  # 累积加权损失
        n_samples += batch_size

    # 返回验证集平均损失
    return total_loss / max(n_samples, 1)


def main():
    """
    主函数：配置超参数 -> 构建数据集 -> 构建 Graph-GRU -> 训练 & 验证 -> 保存最优模型。
    """
    # ====== 一些基础超参数（可以后面慢慢调）======
    T_IN = 288         # 过去 24h （288*5min）
    T_OUT = 288        # 预测 24h
    BATCH_SIZE = 32    # 每个 batch 中的样本数
    GCN_HIDDEN = 16    # G，GCN 输出维度（节点嵌入维度）
    GRU_HIDDEN = 64    # H，GRU 隐藏层维度
    NUM_LAYERS = 1     # GRU 层数（先用 1 层）
    DROPOUT = 0.0      # dropout 比例（单层就关掉）
    LR = 1e-3          # 学习率
    WEIGHT_DECAY = 1e-5  # L2 正则
    NUM_EPOCHS = 20    # 训练轮数

    # 选择设备：优先用 GPU，找不到就用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ====== 1. 构建数据集 & DataLoader ======
    train_ds, val_ds, scaler = build_datasets(T_IN, T_OUT)
    print("Train samples:", len(train_ds))
    print("Val   samples:", len(val_ds))

    # DataLoader：训练集
    train_loader = DataLoader(
        dataset=train_ds,        # 训练集 Dataset
        batch_size=BATCH_SIZE,   # 每个 batch 的大小
        shuffle=True,            # 训练时打乱样本顺序
        num_workers=0,           # Windows 下建议 0，避免多进程问题
        drop_last=True,          # 丢掉最后不足一个 batch 的数据
    )

    # DataLoader：验证集
    val_loader = DataLoader(
        dataset=val_ds,          # 验证集 Dataset
        batch_size=BATCH_SIZE,   # batch 大小（验证可以不满）
        shuffle=False,           # 验证不需要打乱顺序
        num_workers=0,           # 同样用 0
        drop_last=False,         # 不丢最后一个 batch
    )

    # 计算 F_max：所有节点输入维度中的最大值
    #   NODE_INPUT_DIMS["BF"] / ["Cold"] / ... 里保存的是“局部特征 + 时间特征”的维度
    max_feat_dim = max(NODE_INPUT_DIMS.values())
    print("Max feature dim (F_max):", max_feat_dim)

    # ====== 2. 构建 Graph-GRU 模型 / 损失函数 / 优化器 ======
    model = GraphGRU(
        input_dim=max_feat_dim,     # F_max
        gcn_hidden_dim=GCN_HIDDEN,  # G，GCN 输出维度
        gru_hidden_dim=GRU_HIDDEN,  # H，GRU 隐层维度
        t_out=T_OUT,                # 预测长度 T_out
        num_layers=NUM_LAYERS,      # GRU 层数
        dropout=DROPOUT,            # dropout 比例
    ).to(device)                    # 把模型搬到 device

    print(model)  # 打印模型结构，确认参数量和层次

    # 使用 MSE 作为回归任务的损失函数
    criterion = nn.MSELoss()

    # Adam 优化器，带一点 L2 正则（weight_decay）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # 初始化“最优验证集损失”和“最优模型权重”
    best_val_loss = float("inf")
    best_state_dict = None

    # ====== 3. 训练循环 ======
    for epoch in range(1, NUM_EPOCHS + 1):
        # 3.1 在训练集上训练一个 epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # 3.2 在验证集上评估一个 epoch
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        # 打印当前 epoch 的训练 / 验证损失
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )

        # 3.3 如果验证损失更低，则记录为“当前最优模型”
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 把模型参数拷贝到 CPU，避免保存时的 device 依赖
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # ====== 4. 保存最优模型 ======
    if best_state_dict is not None:
        # 用最优参数更新当前模型
        model.load_state_dict(best_state_dict)

        # 打包所有需要的信息：模型参数 + 标准化参数 + 一些超参数
        checkpoint = {
            "model_state_dict": best_state_dict,  # Graph-GRU 权重
            "scaler_mean": scaler["mean"],        # 标准化 mean
            "scaler_std": scaler["std"],          # 标准化 std
            "t_in": T_IN,                         # 输入长度
            "t_out": T_OUT,                       # 预测长度
            "input_dim": max_feat_dim,            # F_max
            "gcn_hidden_dim": GCN_HIDDEN,         # G
            "gru_hidden_dim": GRU_HIDDEN,         # H
        }

        # 保存到当前目录
        torch.save(checkpoint, "graph_gru_best.pt")
        print(f"Saved best Graph-GRU model to graph_gru_best.pt, val_loss={best_val_loss:.6f}")
    else:
        print("No best model saved (maybe dataset is empty?).")


if __name__ == "__main__":
    main()
