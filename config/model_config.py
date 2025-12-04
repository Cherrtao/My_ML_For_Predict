"""
model_config.py

图模型相关的“静态配置”：
- 节点定义（谁是第 0、1、2… 个节点）
- 图的邻接矩阵（物理/业务拓扑）
- GCN 用的归一化邻接矩阵 A_tilde
- 各节点对应的特征列名（方便 Dataset 里统一使用）

后续 Graph-GRU / No-Graph-GRU / Baseline LSTM 都可以从这里 import，
保证整个项目中的图结构和特征布局是一致的。
"""

from __future__ import annotations

from typing import Dict, List

import torch

# ============================================================
# 1. 图节点定义
# ============================================================

#: 节点列表，索引顺序固定：
# 0: BF        -> 包覆机群节点
# 1: Cold      -> 冷风机节点
# 2: Exhaust   -> 排风机节点
# 3: Env       -> 环境节点（温湿度、压差）
# 4: Main      -> 车间总表节点（预测目标）
NODE_LIST: List[str] = ["BF", "Cold", "Exhaust", "Env", "Main"]

#: 节点数量 N
NUM_NODES: int = len(NODE_LIST)

#: 从节点名称到索引的映射，方便在代码里用字符串访问
NODE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(NODE_LIST)}


# ============================================================
# 2. 邻接矩阵（物理拓扑）
# ============================================================

# 物理/业务含义：
# - BF / Cold / Exhaust 的功率最终都汇聚到 Main（总表） -> 它们都连 Main；
# - 这三类设备都受车间环境 Env（温湿度、压差）影响 -> 它们都连 Env；
# - Env 与 Main 之间也通过“冷风机&排风机负荷”产生间接耦合 -> 也连一条边；
# - 每个节点保留自环（GCN 标准做法：Ã = D^{-1/2}(A + I)D^{-1/2}）

# 矩阵行列顺序均为 NODE_LIST = ["BF", "Cold", "Exhaust", "Env", "Main"]
#
#           BF  Cold Exhaust Env  Main
# BF        1    0     0      1    1
# Cold      0    1     0      1    1
# Exhaust   0    0     1      1    1
# Env       1    1     1      1    1
# Main      1    1     1      1    1
#
ADJ: torch.Tensor = torch.tensor(
    [
        [1, 0, 0, 1, 1],  # BF
        [0, 1, 0, 1, 1],  # Cold
        [0, 0, 1, 1, 1],  # Exhaust
        [1, 1, 1, 1, 1],  # Env
        [1, 1, 1, 1, 1],  # Main
    ],
    dtype=torch.float32,
)


# ============================================================
# 3. GCN 用归一化邻接矩阵 A_tilde
# ============================================================

def compute_normalized_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    计算 GCN 用的归一化邻接矩阵 A_tilde = D^{-1/2} A D^{-1/2}。

    参数
    ----
    adj : torch.Tensor
        原始邻接矩阵 A，形状为 (N, N)。

    返回
    ----
    torch.Tensor
        归一化后的邻接矩阵 A_tilde，形状同 (N, N)。
    """
    # 每个节点的度 d_i = sum_j A_ij
    deg: torch.Tensor = adj.sum(dim=1)  # (N,)

    # 计算 d_i^{-1/2}
    deg_inv_sqrt: torch.Tensor = torch.pow(deg, -0.5)
    # 避免 0 度节点导致 inf（这里不太会出现，但保险处理）
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # 构造 D^{-1/2}
    D_inv_sqrt: torch.Tensor = torch.diag(deg_inv_sqrt)

    # A_tilde = D^{-1/2} A D^{-1/2}
    return D_inv_sqrt @ adj @ D_inv_sqrt


#: 归一化后的邻接矩阵，直接在 GCN 层使用：
#   H^{(l+1)} = σ(A_NORM @ H^{(l)} @ W^{(l)})
ADJ_NORM: torch.Tensor = compute_normalized_adj(ADJ)


# ============================================================
# 4. 节点特征列定义（与 CSV 列名对齐）
# ============================================================

# 假定你的 workshop_5min_xxx.csv 至少包含这些列：
# ts_5min,
# bf_power,
# cold_power, cold_freq,
# exh_voltage,
# env_temp, env_hum, env_press,
# main_power,
# dayofweek, is_weekend, tod_sin, tod_cos

#: 时间特征统一放在这里（会拼接到每个节点的局部特征后面）
TIME_FEATURES: List[str] = ["dayofweek", "is_weekend", "tod_sin", "tod_cos"]

#: 各节点“局部特征”的列名（不含时间特征）
#: Dataset 在构造 (N, F) 时：
#:   节点特征 = LOCAL_FEATURES[node] + TIME_FEATURES
NODE_LOCAL_FEATURES: Dict[str, List[str]] = {
    # 包覆机群节点：目前只用总有功功率 bf_power
    "BF": ["bf_power"],
    # 冷风机节点：功率 + 频率
    "Cold": ["cold_power", "cold_freq"],
    # 排风机节点：目前只用直流电压 exh_voltage 作为工况代理特征
    "Exhaust": ["exh_voltage_v","exh_moto_temp"],
    # 环境节点：温度、湿度、压差
    "Env": ["env_temp", "env_hum", "env_press"],
    # 总表节点：总有功功率 main_power（也是预测目标）
    "Main": ["main_power"],
}


def get_node_input_dims() -> Dict[str, int]:
    """
    计算每个节点的输入特征维度（局部特征 + 时间特征）。

    返回
    ----
    Dict[str, int]
        例如 {"BF": 1+4, "Cold": 2+4, ...}
    """
    time_dim = len(TIME_FEATURES)
    dims: Dict[str, int] = {}
    for node, feats in NODE_LOCAL_FEATURES.items():
        dims[node] = len(feats) + time_dim
    return dims


#: 方便在模型里直接看到每个节点的输入维数
NODE_INPUT_DIMS: Dict[str, int] = get_node_input_dims()


# ============================================================
# 5. 导出符号（可选）
# ============================================================

__all__ = [
    "NODE_LIST",
    "NODE_INDEX",
    "NUM_NODES",
    "ADJ",
    "ADJ_NORM",
    "TIME_FEATURES",
    "NODE_LOCAL_FEATURES",
    "NODE_INPUT_DIMS",
    "compute_normalized_adj",
]
