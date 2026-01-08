"""基于 clean_all 按 14-3-3 做 train/val/test 划分。"""
import pandas as pd

INPUT_FILE = "workshop_5min_clean_all.csv"


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["ts_5min"])
    df["date"] = df["ts_5min"].dt.date

    counts = df.groupby("date").size()
    full_days = sorted(counts[counts == 288].index)  # 每天 288 条为完整日
    df_full = df[df["date"].isin(full_days)].copy()

    train_days = full_days[:25]
    val_days = full_days[25:31]
    test_days = full_days[31:]

    df_train = df_full[df_full["date"].isin(train_days)].drop(columns=["date"])
    df_val = df_full[df_full["date"].isin(val_days)].drop(columns=["date"])
    df_test = df_full[df_full["date"].isin(test_days)].drop(columns=["date"])

    df_train.to_csv("workshop_5min_train.csv", index=False)
    df_val.to_csv("workshop_5min_val.csv", index=False)
    df_test.to_csv("workshop_5min_test.csv", index=False)

    print("Saved: workshop_5min_train.csv, workshop_5min_val.csv, workshop_5min_test.csv")


if __name__ == "__main__":
    main()
