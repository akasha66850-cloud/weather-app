"""
train_hourly_models.py
======================
Trains delta regressors for temperature and humidity on data/hourly_global.csv.
- Samples: hour sin/cos, doy sin/cos, lat, lon, val_t, val_t-1, val_t-2, val_t-3, trend_3h (val_t - val_t-3)
- Target: delta_3h = val(t+3h) - val(t)
- Time split: Train <= 2023-12-31, Test >= 2024-01-01
- Held-out cities: 3 cities held out completely to evaluate spatial generalization
- Baselines: Persistence (delta=0) and Same-hour-yesterday (t-24h)
- Evaluates 3h, 12h, 24h horizons (fully vectorized chaining)
- Saves models to models/*.joblib and metrics to models/metrics_hourly.json
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import math
import json
import time
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_hourly_models")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "hourly_global.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HELDOUT_CITIES = ["Reykjavik", "Cairo", "Sydney"]
FEATURES = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "Lat", "Lon",
    "val_t", "val_t_1", "val_t_2", "val_t_3", "trend_3h"
]


def prepare_features(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    logger.info(f"Computing lag features and 3-hour delta target for {col_name}...")
    city_dfs = []
    
    for city, grp in df.groupby("City"):
        g = grp.copy().sort_values("dt").reset_index(drop=True)
        vals = g[col_name].values
        
        # Lags
        val_t = vals
        val_t_1 = np.roll(vals, 1)
        val_t_2 = np.roll(vals, 2)
        val_t_3 = np.roll(vals, 3)
        val_t_plus_3 = np.roll(vals, -3)
        
        # Targets for multi-horizon evaluation
        val_t_plus_12 = np.roll(vals, -12)
        val_t_plus_24 = np.roll(vals, -24)
        
        # Same hour yesterday (t-24h relative to future target)
        val_yesterday_3h = np.roll(vals, 24 - 3)
        val_yesterday_12h = np.roll(vals, 24 - 12)
        val_yesterday_24h = np.roll(vals, 24 - 24)
        
        g["val_t"] = val_t
        g["val_t_1"] = val_t_1
        g["val_t_2"] = val_t_2
        g["val_t_3"] = val_t_3
        g["trend_3h"] = val_t - val_t_3
        g["target_delta"] = val_t_plus_3 - val_t
        
        g["actual_3h"] = val_t_plus_3
        g["actual_12h"] = val_t_plus_12
        g["actual_24h"] = val_t_plus_24
        
        g["yesterday_3h"] = val_yesterday_3h
        g["yesterday_12h"] = val_yesterday_12h
        g["yesterday_24h"] = val_yesterday_24h
        
        # Drop boundary rows (first 24 hours and last 24 hours of each city)
        g = g.iloc[24:-24].copy()
        city_dfs.append(g)
        
    return pd.concat(city_dfs, ignore_index=True)


def evaluate_chained_horizons(model, test_df, col_name, is_humidity=False):
    """
    Evaluates model at 3h, 12h, and 24h horizons across all test sequences.
    Also evaluates persistence baseline and same-hour-yesterday baseline.
    Uses vectorized batch evaluation for instant execution.
    """
    logger.info(f"Evaluating chained multi-horizon predictions for {col_name}...")
    
    # 1. 3h horizon (single-step delta across ALL test rows)
    preds_delta_3h = model.predict(test_df[FEATURES])
    pred_3h = test_df["val_t"].values + preds_delta_3h
    if is_humidity:
        pred_3h = np.clip(pred_3h, 0.0, 100.0)
        
    actual_3h = test_df["actual_3h"].values
    persist_3h = test_df["val_t"].values
    yesterday_3h = test_df["yesterday_3h"].values
    
    mae_3h = float(mean_absolute_error(actual_3h, pred_3h))
    rmse_3h = float(root_mean_squared_error(actual_3h, pred_3h))
    persist_mae_3h = float(mean_absolute_error(actual_3h, persist_3h))
    persist_rmse_3h = float(root_mean_squared_error(actual_3h, persist_3h))
    yesterday_mae_3h = float(mean_absolute_error(actual_3h, yesterday_3h))
    yesterday_rmse_3h = float(root_mean_squared_error(actual_3h, yesterday_3h))
    
    # 2. Multi-step chaining: evaluate across regular sample sequences
    # Sampling every 24 hours (e.g. 12:00 daily per city) provides thousands of complete chains
    sample_mask = (test_df["dt"].dt.hour == 12)
    sample_df = test_df[sample_mask].copy().reset_index(drop=True)
    logger.info(f"Vectorized chaining on {len(sample_df):,} sequences...")
    
    buf_0 = sample_df["val_t_3"].values.copy()
    buf_1 = sample_df["val_t_2"].values.copy()
    buf_2 = sample_df["val_t_1"].values.copy()
    buf_3 = sample_df["val_t"].values.copy()
    cur_val = buf_3.copy()
    
    lats = sample_df["Lat"].values
    lons = sample_df["Lon"].values
    base_dt = sample_df["dt"]
    
    pred_12h = None
    pred_24h = None
    
    for step in range(1, 9):
        step_dt = base_dt + pd.Timedelta(hours=3 * step)
        h_sin = np.sin(2 * np.pi * step_dt.dt.hour / 24).values
        h_cos = np.cos(2 * np.pi * step_dt.dt.hour / 24).values
        d_sin = np.sin(2 * np.pi * step_dt.dt.dayofyear / 365.25).values
        d_cos = np.cos(2 * np.pi * step_dt.dt.dayofyear / 365.25).values
        trend = buf_3 - buf_0
        
        feat_df = pd.DataFrame({
            "hour_sin": h_sin, "hour_cos": h_cos,
            "doy_sin": d_sin, "doy_cos": d_cos,
            "Lat": lats, "Lon": lons,
            "val_t": buf_3, "val_t_1": buf_2, "val_t_2": buf_1, "val_t_3": buf_0,
            "trend_3h": trend
        })
        
        delta = model.predict(feat_df) * (0.97 ** (step - 1))
        cur_val = cur_val + delta
        if is_humidity:
            cur_val = np.clip(cur_val, 0.0, 100.0)
            
        buf_0 = buf_1
        buf_1 = buf_2
        buf_2 = buf_3
        buf_3 = cur_val.copy()
        
        if step == 4:
            pred_12h = cur_val.copy()
        elif step == 8:
            pred_24h = cur_val.copy()
            
    actual_12h = sample_df["actual_12h"].values
    actual_24h = sample_df["actual_24h"].values
    persist_12h = sample_df["val_t"].values
    persist_24h = sample_df["val_t"].values
    yesterday_12h = sample_df["yesterday_12h"].values
    yesterday_24h = sample_df["yesterday_24h"].values
    
    mae_12h = float(mean_absolute_error(actual_12h, pred_12h))
    rmse_12h = float(root_mean_squared_error(actual_12h, pred_12h))
    persist_mae_12h = float(mean_absolute_error(actual_12h, persist_12h))
    persist_rmse_12h = float(root_mean_squared_error(actual_12h, persist_12h))
    yesterday_mae_12h = float(mean_absolute_error(actual_12h, yesterday_12h))
    yesterday_rmse_12h = float(root_mean_squared_error(actual_12h, yesterday_12h))
    
    mae_24h = float(mean_absolute_error(actual_24h, pred_24h))
    rmse_24h = float(root_mean_squared_error(actual_24h, pred_24h))
    persist_mae_24h = float(mean_absolute_error(actual_24h, persist_24h))
    persist_rmse_24h = float(root_mean_squared_error(actual_24h, persist_24h))
    yesterday_mae_24h = float(mean_absolute_error(actual_24h, yesterday_24h))
    yesterday_rmse_24h = float(root_mean_squared_error(actual_24h, yesterday_24h))
    
    logger.info(f"[{col_name}] 3h MAE: {mae_3h:.3f}° (Persistence: {persist_mae_3h:.3f}°, Yesterday: {yesterday_mae_3h:.3f}°)")
    logger.info(f"[{col_name}] 12h MAE: {mae_12h:.3f}° (Persistence: {persist_mae_12h:.3f}°, Yesterday: {yesterday_mae_12h:.3f}°)")
    logger.info(f"[{col_name}] 24h MAE: {mae_24h:.3f}° (Persistence: {persist_mae_24h:.3f}°, Yesterday: {yesterday_mae_24h:.3f}°)")
    
    res = {
        "3h": {
            "model_mae": round(mae_3h, 3), "model_rmse": round(rmse_3h, 3),
            "persistence_mae": round(persist_mae_3h, 3), "persistence_rmse": round(persist_rmse_3h, 3),
            "yesterday_mae": round(yesterday_mae_3h, 3), "yesterday_rmse": round(yesterday_rmse_3h, 3),
            "beats_persistence": mae_3h < persist_mae_3h,
            "beats_yesterday": mae_3h < yesterday_mae_3h
        },
        "12h": {
            "model_mae": round(mae_12h, 3), "model_rmse": round(rmse_12h, 3),
            "persistence_mae": round(persist_mae_12h, 3), "persistence_rmse": round(persist_rmse_12h, 3),
            "yesterday_mae": round(yesterday_mae_12h, 3), "yesterday_rmse": round(yesterday_rmse_12h, 3),
            "beats_persistence": mae_12h < persist_mae_12h,
            "beats_yesterday": mae_12h < yesterday_mae_12h
        },
        "24h": {
            "model_mae": round(mae_24h, 3), "model_rmse": round(rmse_24h, 3),
            "persistence_mae": round(persist_mae_24h, 3), "persistence_rmse": round(persist_rmse_24h, 3),
            "yesterday_mae": round(yesterday_mae_24h, 3), "yesterday_rmse": round(yesterday_rmse_24h, 3),
            "beats_persistence": mae_24h < persist_mae_24h,
            "beats_yesterday": mae_24h < yesterday_mae_24h
        }
    }
    return res


def train_pipeline():
    logger.info(f"Loading {DATA_FILE}...")
    t0 = time.time()
    df = pd.read_csv(DATA_FILE)
    df["dt"] = pd.to_datetime(df["time"])
    logger.info(f"Loaded {len(df):,} rows across {df['City'].nunique()} cities.")

    df["hour_sin"] = np.sin(2 * np.pi * df["dt"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["dt"].dt.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dt"].dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dt"].dt.dayofyear / 365.25)

    city_coords = df.groupby("City")[["Lat", "Lon"]].first().to_dict(orient="index")

    # ═════════════════════════════════════════════════════════════════════════
    # 1. TEMPERATURE MODEL
    # ═════════════════════════════════════════════════════════════════════════
    df_temp = prepare_features(df, "temperature_2m")

    train_mask = (df_temp["dt"] <= "2023-12-31 23:00") & (~df_temp["City"].isin(HELDOUT_CITIES))
    test_mask = (df_temp["dt"] >= "2024-01-01 00:00") & (~df_temp["City"].isin(HELDOUT_CITIES))
    heldout_mask = (df_temp["City"].isin(HELDOUT_CITIES)) & (df_temp["dt"] >= "2024-01-01 00:00")

    X_tr_t, y_tr_t = df_temp.loc[train_mask, FEATURES], df_temp.loc[train_mask, "target_delta"]
    test_temp_df = df_temp[test_mask]
    heldout_temp_df = df_temp[heldout_mask]

    logger.info(f"Temperature Train set: {len(X_tr_t):,} | Test set: {len(test_temp_df):,} | Held-out set: {len(heldout_temp_df):,}")

    logger.info("Training HistGradientBoostingRegressor for Temperature delta...")
    hgb_temp = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
    t_start = time.time()
    hgb_temp.fit(X_tr_t, y_tr_t)
    logger.info(f"Temperature HistGBM trained in {time.time() - t_start:.2f}s")

    temp_metrics = evaluate_chained_horizons(hgb_temp, test_temp_df, "Temperature")
    temp_heldout_metrics = evaluate_chained_horizons(hgb_temp, heldout_temp_df, "Temperature (Held-out Cities)")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. HUMIDITY MODEL
    # ═════════════════════════════════════════════════════════════════════════
    df_hum = prepare_features(df, "relative_humidity_2m")

    train_mask_h = (df_hum["dt"] <= "2023-12-31 23:00") & (~df_hum["City"].isin(HELDOUT_CITIES))
    test_mask_h = (df_hum["dt"] >= "2024-01-01 00:00") & (~df_hum["City"].isin(HELDOUT_CITIES))
    heldout_mask_h = (df_hum["City"].isin(HELDOUT_CITIES)) & (df_hum["dt"] >= "2024-01-01 00:00")

    X_tr_h, y_tr_h = df_hum.loc[train_mask_h, FEATURES], df_hum.loc[train_mask_h, "target_delta"]
    test_hum_df = df_hum[test_mask_h]
    heldout_hum_df = df_hum[heldout_mask_h]

    logger.info(f"Humidity Train set: {len(X_tr_h):,} | Test set: {len(test_hum_df):,} | Held-out set: {len(heldout_hum_df):,}")

    logger.info("Training HistGradientBoostingRegressor for Humidity delta...")
    hgb_hum = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
    t_start = time.time()
    hgb_hum.fit(X_tr_h, y_tr_h)
    logger.info(f"Humidity HistGBM trained in {time.time() - t_start:.2f}s")

    hum_metrics = evaluate_chained_horizons(hgb_hum, test_hum_df, "Humidity", is_humidity=True)
    hum_heldout_metrics = evaluate_chained_horizons(hgb_hum, heldout_hum_df, "Humidity (Held-out Cities)", is_humidity=True)

    # ═════════════════════════════════════════════════════════════════════════
    # 3. SAVE MODELS & METRICS
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("Saving trained models and metrics...")
    temp_model_path = os.path.join(MODELS_DIR, "temp_model.joblib")
    hum_model_path = os.path.join(MODELS_DIR, "hum_model.joblib")
    cities_path = os.path.join(MODELS_DIR, "training_cities.joblib")

    joblib.dump(hgb_temp, temp_model_path, compress=3)
    joblib.dump(hgb_hum, hum_model_path, compress=3)
    joblib.dump(city_coords, cities_path)

    metrics_payload = {
        "dataset": {
            "source": "Open-Meteo Historical Archive API (ERA5 reanalysis)",
            "total_rows": len(df),
            "cities_count": len(city_coords),
            "held_out_cities": HELDOUT_CITIES,
            "train_cutoff": "2023-12-31",
            "test_start": "2024-01-01"
        },
        "temperature_model": {
            "model_type": "HistGradientBoostingRegressor(delta_3h)",
            "test_metrics": temp_metrics,
            "held_out_cities_metrics": temp_heldout_metrics
        },
        "humidity_model": {
            "model_type": "HistGradientBoostingRegressor(delta_3h)",
            "test_metrics": hum_metrics,
            "held_out_cities_metrics": hum_heldout_metrics
        }
    }

    metrics_file = os.path.join(MODELS_DIR, "metrics_hourly.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(f"Saved metrics to {metrics_file}")
    logger.info(f"Temperature model size: {os.path.getsize(temp_model_path) / 1024 / 1024:.2f} MB")
    logger.info(f"Humidity model size: {os.path.getsize(hum_model_path) / 1024 / 1024:.2f} MB")
    logger.info(f"Entire training pipeline completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    train_pipeline()
