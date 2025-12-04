"""
graph_dataset.py

基于车间级 5min 宽表 CSV，构造用于 Graph-GRU / No-Graph-GRU / LSTM 的
滑动窗口时序数据集。

- 支持多节点：BF / Cold / Exhaust / Env / Main
- 每个节点的特征维度可以不同（局部特征数量不同），
  通过“按最大维度 zero-padding”的方式统一到同一个 F 维度。

最终 __getitem__ 返回：
    X: (T_in, N, F)  例如 (288, 5, F_max)
    y: (T_out,)      例如 (288,)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.model_config import (
    NODE_LIST,
    NODE_INDEX,
    NODE_LOCAL_FEATURES,
    TIME_FEATURES,
    NODE_INPUT_DIMS,  # 虽然不必须，用它来算最大特征维度更直观
)


# ============================================================
# 1. 标准化参数拟合
# ============================================================

def fit_scaler_from_csv(csv_path: str) -> Dict[str, pd.Series]:
    """
    在整张 CSV 上拟合特征的 mean / std，用于后续标准化。

    注意：
    - 这里直接对“所有节点的局部特征 + 时间特征”的并集做统计；
    - ts_5min 不参与标准化。
    """
    df = pd.read_csv(csv_path)# 用pandas读取csv文件成为DataFrame

    # 保证时间列存在，但不参与 scaler 计算
    if "ts_5min" in df.columns:
        df["ts_5min"] = pd.to_datetime(df["ts_5min"])# 如果时间列ts_5min存在，就转成pandas的datetime类型，不参与到后面的均值方差的计算

    # 收集所有可能参与标准化的特征列
    local_cols: List[str] = []
    for node in NODE_LIST:
        local_cols.extend(NODE_LOCAL_FEATURES[node])# 手机所有节点的局部特征的列名，也就是对每个节点BF/Cold等把它对应的NODE_LOCAL_FEATURES[node]追加到local_cols列表中

    local_cols = sorted(set(local_cols))  # 去重,去重后排序，得到所有唯一的局部特征
    feature_cols = local_cols + TIME_FEATURES# 去重排序后再加上时间特征TIME_FEATURE,构成最终要参与标准化的全部特征列名

    # 防止 KeyError：如果有列在配置里，但 CSV 里没有，直接报清晰错误
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"[fit_scaler_from_csv] 下列列名在 CSV 中不存在: {missing}")# 安全校验，如果有列再配置里写了但是csv文件中没有，直接报错

    df_feat = df[feature_cols].astype(float)# 选取这些特征列，转换为float类型，方便计算均值方差

    mean = df_feat.mean(axis=0) # asix=0求均值/标准差
    std = df_feat.std(axis=0, ddof=0) # ddof=0用总体标准差
    std[std == 0] = 1.0  # 避免除以 0，如果某一列的标准差为0，统一设成1，避免后续除以0

    return {"mean": mean, "std": std} # 返回一个自带你，后面的GraphSequenceDataset会用这个做(x-mean)/std


# ============================================================
# 2. GraphSequenceDataset 定义
# ============================================================

class GraphSequenceDataset(Dataset): # 一个pytorch的dataset类
    """
    用于车间级 5min 时序预测的图数据集。

    给定一整段按时间排序的宽表（train/val/test 之一），构造滑动窗口：

        输入窗口 T_in：过去 24h（288 个 5min 点）
        输出窗口 T_out：未来 24h（288 个 5min 点）

    每个样本：
        X: (T_in, N, F)
        y: (T_out,)     （这里 y 只取 Main 节点的 main_power）
    """

    def __init__(
        self,
        csv_path: str,# 数据csv路径
        t_in: int,# 输入窗口的长度
        t_out: int,# 输入窗口的长度
        feature_scaler: Optional[Dict[str, pd.Series]] = None, # 标准化参数
        fit_scaler: bool = False, # 如果为True且feature_caler为空，则在当前csv上重新拟合一个scaler
    ) -> None:
        """
        参数
        ----
        csv_path : str
            已经通过 build_dataset / split_dataset 生成的 CSV 文件路径。
        t_in : int
            输入窗口长度（时间步数）。
        t_out : int
            输出窗口长度（时间步数）。
        feature_scaler : Dict[str, pd.Series], optional
            标准化参数 {"mean": Series, "std": Series}。
        fit_scaler : bool
            若为 True 且 feature_scaler 为 None，则在本 CSV 上重新拟合 scaler。
            一般训练集用 fit_scaler=True，其它集用外部传入的 scaler。
        """
        super().__init__() # 调用父类Dataset的狗在函数

        self.csv_path = csv_path
        self.t_in = t_in
        self.t_out = t_out # 把传入的参数存到对象属性里，后面要用

        # ---------- 读入 CSV ----------
        df = pd.read_csv(csv_path) # 将csv格式转为dataframe格式
        if "ts_5min" in df.columns:
            df["ts_5min"] = pd.to_datetime(df["ts_5min"]) #如果有ts_5min列，转成datetime
            df = df.sort_values("ts_5min").reset_index(drop=True)#按照时间进行排序，确保是从早到晚，再去重新编号Index

        self.df = df #最后把整理好的表存到self.df

        # ---------- 准备标准化 ----------
        # 收集所有参与标准化的列名
        local_cols: List[str] = []
        for node in NODE_LIST:
            local_cols.extend(NODE_LOCAL_FEATURES[node])
        local_cols = sorted(set(local_cols))
        self.feature_cols: List[str] = local_cols + TIME_FEATURES #和上一个fit_scaler_from_csv类似

        # 检查是否有缺失列
        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise KeyError(
                f"[GraphSequenceDataset.__init__] CSV 中缺少以下特征列: {missing}"
            )# 安全检查：如果配置里的列，在 CSV 中不存在，就立刻报错，告诉你哪里不配套。

        # 若需要，在本 CSV 上拟合 scaler
        if fit_scaler and feature_scaler is None:
            feature_scaler = fit_scaler_from_csv(csv_path)

        self.scaler = feature_scaler # 如果fit_scaler = True且没窜进来scaler，就调用上面的fit_scaler_from_csv用当前csv来拟合一个scaler

        # 应用标准化
        if self.scaler is not None:
            mean = self.scaler["mean"].reindex(self.feature_cols)
            std = self.scaler["std"].reindex(self.feature_cols).replace(0, 1.0)
            self.df[self.feature_cols] = (self.df[self.feature_cols] - mean) / std #如果self.scaler存在，把mead/std重新对齐到sekf.feature_cols的顺序(reindex)，确保顺序一致
            # std中的0改成1，防止除0
            # 对self.feature_cols这些列做标准化，并直接写回self.df

        # ---------- 计算样本数量 ----------
        self.T = len(self.df) #这张DataFrame的总时间步数，也就是总行数
        self.num_samples = max(0, self.T - (self.t_in + self.t_out) + 1)# 每个样本需要t_in+t_out个时间步，可用的窗口就是T-(t_in+t_out)+1,如果不足返回0

        # ---------- 计算每个节点的特征维度 & 最大维度 ----------
        # NODE_INPUT_DIMS[node] = len(NODE_LOCAL_FEATURES[node]) + len(TIME_FEATURES)
        self.node_input_dims: Dict[str, int] = NODE_INPUT_DIMS #从model_config里拿到每个节点的输入维度
        self.max_feat_dim: int = max(self.node_input_dims.values())#去所有节点维度中的最大值 = F_max

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int): # 取第idx个样本
        """
        返回第 idx 个样本：

        X: (T_in, N, F_max)
        y: (T_out,)
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx) #对非法索引直接抛出错误

        # 确定本样本的滑动窗口区间，输入片段是[idx,idx+t_in)，输出片段是[idx+t_in,idx+t_in+t_out)
        start = idx
        in_end = start + self.t_in
        out_end = in_end + self.t_out
        # 用iloc取两端DataFrame，分别代表过去和未来
        df_in = self.df.iloc[start:in_end]      # 过去 T_in 步
        df_out = self.df.iloc[in_end:out_end]   # 未来 T_out 步

        # 准备一个全为0的numpy数组X_np, 维度为(T_in,N,F_max)，后面会把每个时间不，每个节点的实际特征值天津来，不够的部分保持0
        # ---------- 构造 X: (T_in, N, F_max) ----------
        T_in = self.t_in
        N = len(NODE_LIST)
        F_max = self.max_feat_dim

        X_np = np.zeros((T_in, N, F_max), dtype=np.float32)


        # 逐时间步、逐节点填充
        # 对输入窗口的每一个时间步t，遍历每一个节点(BF/cold...),对于该接待你，取它配置的局部特征+时间特征---feat_list
        # 从当前这一行row中把这些列取出来，转为numpy数组，记为values，长度就是该节点的实际特征位数F_node
        for t, (_, row) in enumerate(df_in.iterrows()):
            for node_idx, node_name in enumerate(NODE_LIST):
                # 当前节点实际使用的特征列表
                feat_list = NODE_LOCAL_FEATURES[node_name] + TIME_FEATURES
                values = row[feat_list].to_numpy(dtype=np.float32)  # shape: (F_node,)

                F_node = values.shape[0]
                if F_node > F_max:
                    # 理论上不会发生，除非配置和 max_feat_dim 不一致
                    raise ValueError(
                        f"节点 {node_name} 特征维度 {F_node} > F_max {F_max}"
                    )

                # 填到前 F_node 维，其余维度保持 0
                X_np[t, node_idx, :F_node] = values

        # ---------- 构造 y: (T_out,) = 未来 main_power ----------
        y_np = df_out["main_power"].to_numpy(dtype=np.float32)  # shape: (T_out,),输出目标y，仅取未来窗口中Main节点的目标变量main_powe的序列，长度为T_out

        X = torch.from_numpy(X_np)  # (T_in, N, F_max)
        y = torch.from_numpy(y_np)  # (T_out,)

        return X, y
        #把numpy转成pytroch tensor，返回(x,y)
