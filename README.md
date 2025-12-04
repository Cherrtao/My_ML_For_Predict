# 车间级能耗预测数据集构建脚本

## 配置
- 编辑 `config.py`：填写 `POSTGRES_DSN`，按需调整时间范围与 `DEVICE_CONFIG`（所有业务字段统一维护在此）。

## 生成全量宽表
- 执行：`python -m data.build_dataset`
- 输出：`workshop_5min_clean_all.csv`
  - 列：ts_5min、bf_power、cold_power、cold_freq、exh_power、exh_freq、env_temp、env_hum、env_press、main_power、dayofweek、is_weekend、tod_sin、tod_cos

## 按 14-3-3 划分
- 执行：`python -m data.split_dataset`
- 输出：`workshop_5min_train.csv`、`workshop_5min_val.csv`、`workshop_5min_test.csv`
  - 仍包含 ts_5min、全部设备特征、main_power 和时间特征

## 探索
- `notebooks/explore_dataset.ipynb` 可用于可视化与质量检查（占位）。
