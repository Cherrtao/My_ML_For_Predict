"""从 TimescaleDB 构建 5min 宽表并添加时间特征。"""
import numpy as np
import pandas as pd
from sqlalchemy import text

from config.dataset_config import START_TS, END_TS, DEVICE_CONFIG, TIME_RANGES
from db.connection import get_engine

SQL = text(
    """
    SELECT
        c.ts_5min,
        d.devid,
        p.propname AS propname,
        c.v_avg
    FROM public.cagg_measure_5min AS c
    JOIN public.dim_point  AS p ON c.point_sk  = p.point_sk
    JOIN public.dim_device AS d ON p.devid = d.devid
    WHERE c.ts_5min >= :start_ts
      AND c.ts_5min <  :end_ts;
    """
)


def build_wide_df(df_long: pd.DataFrame) -> pd.DataFrame:
    """根据配置聚合长表，生成宽表特征。"""
    cfg = DEVICE_CONFIG

    def agg_bf_power(df: pd.DataFrame) -> pd.Series:
        devids = cfg["bf"]["devids"]
        df_bf = df[df["devid"].isin(devids) & df["propname"].isin(cfg["bf"]["power_points"])]
        if df_bf.empty:
            return pd.Series(dtype=float, name="bf_power")
        return df_bf.groupby("ts_5min")["v_avg"].sum().rename("bf_power")

    def agg_cold(df: pd.DataFrame) -> pd.DataFrame:
        cold = df[df["devid"].isin(cfg["cold_fan"]["devids"])]
        power = cold[cold["propname"] == cfg["cold_fan"]["power_point"]].groupby("ts_5min")["v_avg"].mean().rename(
            "cold_power"
        )
        freq = cold[cold["propname"] == cfg["cold_fan"]["freq_point"]].groupby("ts_5min")["v_avg"].mean().rename(
            "cold_freq"
        )
        return pd.concat([power, freq], axis=1)

    def agg_exh(df: pd.DataFrame) -> pd.DataFrame:
        cond = df["devid"] == cfg["exhaust_fan"]["devid"]
        exh = df[cond]
        pieces = []
        for prop, col in [
            (cfg["exhaust_fan"]["power_point"], "exh_power"),
            (cfg["exhaust_fan"]["freq_point"], "exh_freq"),
            (cfg["exhaust_fan"].get("voltage_point"), "exh_voltage_v"),
            (cfg["exhaust_fan"].get("moto_temp_point"), "exh_moto_temp"),
            (cfg["exhaust_fan"].get("ambient_temp_point"), "exh_ambient_temp"),
        ]:
            if prop:
                series = exh[exh["propname"] == prop].set_index("ts_5min")["v_avg"].rename(col)
                pieces.append(series)
        if not pieces:
            return pd.DataFrame(index=exh["ts_5min"].unique())
        return pd.concat(pieces, axis=1)

    def agg_env(df: pd.DataFrame) -> pd.DataFrame:
        cond = df["devid"] == cfg["env"]["devid"]
        env = df[cond]
        temp = env[env["propname"] == cfg["env"]["temp_point"]].set_index("ts_5min")["v_avg"].rename("env_temp")
        hum = env[env["propname"] == cfg["env"]["hum_point"]].set_index("ts_5min")["v_avg"].rename("env_hum")
        press = env[env["propname"] == cfg["env"]["press_point"]].set_index("ts_5min")["v_avg"].rename("env_press")
        return pd.concat([temp, hum, press], axis=1)

    def agg_main(df: pd.DataFrame) -> pd.Series:
        cond_dev = df["devid"] == cfg["main_meter"]["devid"]
        df_main = df[cond_dev & df["propname"].isin(cfg["main_meter"]["power_points"])]
        if df_main.empty:
            return pd.Series(dtype=float, name="main_power")
        return df_main.groupby("ts_5min")["v_avg"].sum().rename("main_power")

    parts = [
        agg_bf_power(df_long),
        agg_cold(df_long),
        agg_exh(df_long),
        agg_env(df_long),
        agg_main(df_long),
    ]
    wide = pd.concat(parts, axis=1).reset_index()
    return wide


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间特征 dayofweek / is_weekend / tod_sin / tod_cos。"""
    df = df.copy()
    df["ts_5min"] = pd.to_datetime(df["ts_5min"])
    df["dayofweek"] = df["ts_5min"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    minutes = df["ts_5min"].dt.hour * 60 + df["ts_5min"].dt.minute
    df["tod_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    df["tod_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    return df


def get_time_ranges():
    return TIME_RANGES if TIME_RANGES else [(START_TS, END_TS)]


def run_diagnostics(conn, start_ts, end_ts) -> None:
    diag_sqls = {
        "cagg_range": text(
            """
            SELECT count(*) FROM public.cagg_measure_5min c
            WHERE c.ts_5min >= :start_ts AND c.ts_5min < :end_ts
            """
        ),
        "cagg_join_point": text(
            """
            SELECT count(*) FROM public.cagg_measure_5min c
            JOIN public.dim_point p ON c.point_sk = p.point_sk
            WHERE c.ts_5min >= :start_ts AND c.ts_5min < :end_ts
            """
        ),
        "cagg_join_point_devname": text(
            """
            SELECT count(*) FROM public.cagg_measure_5min c
            JOIN public.dim_point p ON c.point_sk = p.point_sk
            JOIN public.dim_device d ON p.devname = d.devname
            WHERE c.ts_5min >= :start_ts AND c.ts_5min < :end_ts
            """
        ),
        "cagg_join_point_devid": text(
            """
            SELECT count(*) FROM public.cagg_measure_5min c
            JOIN public.dim_point p ON c.point_sk = p.point_sk
            JOIN public.dim_device d ON p.devid = d.devid
            WHERE c.ts_5min >= :start_ts AND c.ts_5min < :end_ts
            """
        ),
    }
    params = {"start_ts": start_ts, "end_ts": end_ts}
    for name, sql_stmt in diag_sqls.items():
        try:
            res = conn.execute(sql_stmt, params).fetchall()
            print(f"[diag] {name} ({start_ts} ~ {end_ts}): {res}")
        except Exception as e:
            print(f"[diag] {name} failed ({start_ts} ~ {end_ts}): {e}")


def load_range(engine, start_ts, end_ts) -> pd.DataFrame:
    params = {"start_ts": start_ts, "end_ts": end_ts}
    print("Executing SQL for long table:", SQL.text.strip(), "params:", params)
    df_long = pd.read_sql(SQL, engine, params=params)
    df_long["ts_5min"] = pd.to_datetime(df_long["ts_5min"]).dt.tz_localize(None)
    print(f"Fetched rows: {len(df_long)}")

    wide = build_wide_df(df_long)
    full_index = pd.date_range(start=start_ts, end=end_ts, freq="5min", inclusive="left")
    wide = wide.set_index("ts_5min").reindex(full_index)
    wide.index.name = "ts_5min"
    wide = wide.ffill().bfill().reset_index()

    # Optional day slicing per range.
    # start_date = pd.to_datetime(start_ts).date()
    # end_date = (pd.to_datetime(end_ts) - pd.Timedelta(days=1)).date()
    # wide["date"] = wide["ts_5min"].dt.date
    # mask = (wide["date"] >= start_date) & (wide["date"] <= end_date)
    # wide = wide.loc[mask].drop(columns=["date"])

    return add_time_features(wide)



def main():
    engine = get_engine()
    ranges = get_time_ranges()

    with engine.connect() as conn:
        for start_ts, end_ts in ranges:
            run_diagnostics(conn, start_ts, end_ts)

    wide_parts = [load_range(engine, start_ts, end_ts) for start_ts, end_ts in ranges]
    if wide_parts:
        wide = pd.concat(wide_parts, ignore_index=True).sort_values("ts_5min")
    else:
        wide = pd.DataFrame()

    wide.to_csv("workshop_5min_clean_all.csv", index=False)
    print("Saved: workshop_5min_clean_all.csv")



if __name__ == "__main__":
    main()
