"""Global config: DB connection, time range, device/point mapping."""

# PostgreSQL (TimescaleDB) connection
POSTGRES_DSN = "postgresql+psycopg2://postgres:postgres@192.168.31.224:5432/HuaZhiBackUp"

# Time range for this experiment (2025-10-23 ~ 2025-11-12, end exclusive)
START_TS = "2025-10-23 00:00:00"
END_TS = "2025-11-12 00:00:00"

WORKSHOP = "Workshop4"

# Device/point config — filled with current devids/propnames
DEVICE_CONFIG = {
    "bf": {
        "devids": [51295, 51297, 51299, 51301, 51303, 51305, 51307, 51309, 51311, 51313, 51315, 51317, 51319, 51321, 51323, 51325, 51327],
        "power_points": ["ALL_power_1"],  # adjust if you need other props
    },
    "cold_fan": {
        "devids": [90001, 90002],
        "power_point": "OutPower",
        "freq_point": "Freq",
    },
    "exhaust_fan": {
        "devid": 89999,
        "power_point": "vfd_power_pct",
        "freq_point": "vfd_freq_hz",
        "voltage_point": "vfd_voltage_v",
        "moto_temp_point": "MotoTemp",
        "ambient_temp_point": "vfd_ambient_temp_c",
    },
    "env": {
        "devid": 90000,
        "temp_point": "Temp",
        "hum_point": "Aname",
        "press_point": "Pression",
    },
    "main_meter": {
        "devid": 90004,
        "power_points": ["All-Power"],
    },
}


# config/dataset_config.py
from pathlib import Path

# 项目根目录 = 当前文件的上两级目录
ROOT_DIR = Path(__file__).resolve().parents[1]

# 三个数据集 CSV 的路径（后面训练、调试都用它们）
TRAIN_CSV = ROOT_DIR / "workshop_5min_train.csv"
VAL_CSV   = ROOT_DIR / "workshop_5min_val.csv"
TEST_CSV  = ROOT_DIR / "workshop_5min_test.csv"
