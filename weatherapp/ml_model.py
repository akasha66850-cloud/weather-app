"""
ml_model.py  –  India Weather ML Pipeline
==========================================
Rain classifier  : RandomForest or HistGradientBoosting (best rain-class F1)
Regressors       : RandomForest with 3-lag hourly features (temp & humidity)
                   Falls back to chained single-lag if hourly data is missing
                   or if the model does not beat the persistence baseline.
Persistence      : joblib files in <BASE_DIR>/models/
Training trigger : python manage.py train_models (manual/offline only)
"""

import os
import json
import math
import logging
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
# Resolved lazily from settings so this module imports cleanly before Django is
# fully configured.
def _base_dir():
    from django.conf import settings
    return settings.BASE_DIR


def _models_dir():
    d = os.path.join(_base_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def _daily_csv():
    return os.path.join(_base_dir(), "weather_india.csv")


def _hourly_csv():
    return os.path.join(_base_dir(), "data", "hourly_india.csv")


# ── Compass ────────────────────────────────────────────────────────────────────
COMPASS_POINTS = [
    ("N", 0, 11.25), ("NNE", 11.25, 33.75),
    ("NE", 33.75, 56.25), ("ENE", 56.25, 78.75),
    ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
    ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75),
    ("S", 168.75, 191.25), ("SSW", 191.25, 213.75),
    ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
    ("W", 258.75, 281.25), ("WNW", 281.25, 303.75),
    ("NW", 303.75, 326.25), ("NNW", 326.25, 348.75),
    ("N", 348.75, 360),
]

KNOWN_COMPASS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

# Fallback for unknown wind directions (explicit, not silent index-0)
WIND_FALLBACK = "N"


def get_compass_direction(wind_deg: float) -> str:
    """Convert wind degrees to 16-point compass string."""
    wind_deg = wind_deg % 360
    for point, start, end in COMPASS_POINTS:
        if start <= wind_deg < end:
            return point
    return "N"


# ── In-memory model cache ──────────────────────────────────────────────────────
_model_cache: dict = {}


def clear_model_cache():
    """Force reload on next request (useful after train_models)."""
    _model_cache.clear()
    logger.info("ml_model: cache cleared")


# ── Feature helpers ────────────────────────────────────────────────────────────
def _month_features(month: int):
    """Return (sin, cos) cyclical encoding for month 1-12."""
    rad = 2 * math.pi * month / 12
    return math.sin(rad), math.cos(rad)


def _hour_features(hour: int):
    """Return (sin, cos) cyclical encoding for hour 0-23."""
    rad = 2 * math.pi * hour / 24
    return math.sin(rad), math.cos(rad)


def _encode_wind(wd: str, le: LabelEncoder) -> int:
    """Encode wind direction string; falls back to WIND_FALLBACK with a warning."""
    if wd in le.classes_:
        return int(le.transform([wd])[0])
    logger.warning(
        "ml_model: unknown wind direction '%s', falling back to '%s'",
        wd,
        WIND_FALLBACK,
    )
    return int(le.transform([WIND_FALLBACK])[0])


# ── Training ───────────────────────────────────────────────────────────────────

def train_rain_classifier(daily_csv: str) -> dict:
    """
    Train rain classifier on weather_india.csv.
    Returns dict with model, wind_le, rain_le, threshold, and metrics.
    """
    logger.info("ml_model: loading daily dataset from %s", daily_csv)
    df = pd.read_csv(daily_csv, parse_dates=["Date"])
    df = df.dropna(subset=[
        "MinTemp", "MaxTemp", "Temp", "Humidity", "Pressure",
        "WindGustSpeed", "WindGustDir", "Month", "Lat", "Lon",
        "RainTomorrow",
    ])

    # ── Encode wind direction ──────────────────────────────────────────────
    wind_le = LabelEncoder()
    wind_le.fit(KNOWN_COMPASS)
    # Map any unknowns to fallback before encoding
    df["WindGustDir"] = df["WindGustDir"].apply(
        lambda x: x if x in KNOWN_COMPASS else WIND_FALLBACK
    )
    df["WindGustDir_enc"] = wind_le.transform(df["WindGustDir"])

    rain_le = LabelEncoder()
    df["RainTomorrow_enc"] = rain_le.fit_transform(df["RainTomorrow"])

    # ── Cyclical month ─────────────────────────────────────────────────────
    df["month_sin"] = df["Month"].apply(lambda m: math.sin(2 * math.pi * m / 12))
    df["month_cos"] = df["Month"].apply(lambda m: math.cos(2 * math.pi * m / 12))

    FEATURES = [
        "MinTemp", "MaxTemp", "Temp", "Humidity", "Pressure",
        "WindGustSpeed", "WindGustDir_enc", "month_sin", "month_cos",
        "Lat", "Lon",
    ]
    TARGET = "RainTomorrow_enc"

    # ── Time-based split ───────────────────────────────────────────────────
    train_mask = df["Date"] <= pd.Timestamp("2022-12-31")
    val_mask   = (df["Date"] >= pd.Timestamp("2022-01-01")) & (df["Date"] <= pd.Timestamp("2022-12-31"))
    test_mask  = df["Date"] >= pd.Timestamp("2023-01-01")

    X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET]
    X_val,   y_val   = df.loc[val_mask,   FEATURES], df.loc[val_mask,   TARGET]
    X_test,  y_test  = df.loc[test_mask,  FEATURES], df.loc[test_mask,  TARGET]

    logger.info(
        "ml_model: train=%d  val=%d  test=%d  (rain class balance: %.1f%%)",
        len(X_train), len(X_val), len(X_test),
        100 * y_train.mean(),
    )

    # ── Train RF & HistGBM, pick best rain-F1 on val ──────────────────────
    rf = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=40,
        max_depth=12,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    hgb = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=300,
        random_state=42,
    )
    hgb.fit(X_train, y_train)

    rain_class_idx = list(rain_le.classes_).index("Yes")

    rf_f1  = f1_score(y_val, rf.predict(X_val),  pos_label=rain_class_idx, zero_division=0)
    hgb_f1 = f1_score(y_val, hgb.predict(X_val), pos_label=rain_class_idx, zero_division=0)

    if hgb_f1 > rf_f1:
        model = hgb
        model_type = "HistGradientBoostingClassifier"
        logger.info("ml_model: HistGBM wins (val rain-F1 %.3f vs RF %.3f)", hgb_f1, rf_f1)
    else:
        model = rf
        model_type = "RandomForestClassifier"
        logger.info("ml_model: RF wins (val rain-F1 %.3f vs HistGBM %.3f)", rf_f1, hgb_f1)

    # ── Threshold tuning on validation slice ──────────────────────────────
    val_proba = model.predict_proba(X_val)[:, rain_class_idx]
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.20, 0.81, 0.01):
        preds = (val_proba >= thr).astype(int)
        score = f1_score(y_val, preds, pos_label=rain_class_idx, zero_division=0)
        if score > best_f1:
            best_f1, best_thr = score, float(thr)
    logger.info("ml_model: tuned threshold=%.2f  val rain-F1=%.3f", best_thr, best_f1)

    # ── Evaluate on test set ───────────────────────────────────────────────
    test_proba = model.predict_proba(X_test)[:, rain_class_idx]
    test_preds = (test_proba >= best_thr).astype(int)

    acc  = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, pos_label=rain_class_idx, zero_division=0)
    rec  = recall_score(y_test, test_preds,    pos_label=rain_class_idx, zero_division=0)
    f1   = f1_score(y_test, test_preds,        pos_label=rain_class_idx, zero_division=0)
    auc  = roc_auc_score(y_test, test_proba)
    cm   = confusion_matrix(y_test, test_preds).tolist()

    # Feature importances (RF only; skip for HistGBM which uses different API)
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(FEATURES, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        top_features = [{"feature": k, "importance": round(float(v), 4)} for k, v in importances[:8]]
    else:
        top_features = []

    metrics = {
        "model_type":    model_type,
        "accuracy":      round(acc,  4),
        "precision_rain": round(prec, 4),
        "recall_rain":   round(rec,  4),
        "f1_rain":       round(f1,   4),
        "roc_auc":       round(auc,  4),
        "threshold":     round(best_thr, 4),
        "confusion_matrix": cm,
        "top_features":  top_features,
    }
    logger.info("ml_model: test rain metrics: %s", metrics)

    return {
        "model":    model,
        "wind_le":  wind_le,
        "rain_le":  rain_le,
        "threshold": best_thr,
        "metrics":  metrics,
    }


def train_regressors(hourly_csv: str) -> dict:
    """
    Train temp & humidity regressors using 3-lag hourly features on hourly_india.csv.
    Returns dict with temp_model, hum_model, and metrics.
    Falls back to None,None if hourly data is unavailable or model is worse
    than persistence baseline.
    """
    if not os.path.exists(hourly_csv):
        logger.warning("ml_model: hourly CSV not found at %s — skipping regressor training", hourly_csv)
        return {"temp_model": None, "hum_model": None, "metrics": {}}

    logger.info("ml_model: loading hourly dataset from %s", hourly_csv)
    df = pd.read_csv(hourly_csv, parse_dates=["time"])
    df = df.dropna(subset=["temperature_2m", "relative_humidity_2m", "City", "Lat", "Lon"])

    df["hour"]      = df["time"].dt.hour
    df["month"]     = df["time"].dt.month
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    all_dfs = []
    for city, grp in df.sort_values("time").groupby("City"):
        g = grp.copy().reset_index(drop=True)
        for col in ("temperature_2m", "relative_humidity_2m"):
            g[f"{col}_lag1"] = g[col].shift(1)
            g[f"{col}_lag2"] = g[col].shift(2)
            g[f"{col}_target"] = g[col].shift(-3)   # predict 3 h ahead
        all_dfs.append(g)

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna(subset=[
        "temperature_2m", "temperature_2m_lag1", "temperature_2m_lag2",
        "relative_humidity_2m", "relative_humidity_2m_lag1", "relative_humidity_2m_lag2",
        "temperature_2m_target", "relative_humidity_2m_target",
    ])

    FEAT_TEMP = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "Lat", "Lon",
        "temperature_2m", "temperature_2m_lag1", "temperature_2m_lag2",
    ]
    FEAT_HUM = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "Lat", "Lon",
        "relative_humidity_2m", "relative_humidity_2m_lag1", "relative_humidity_2m_lag2",
    ]

    train_mask = df["time"] <= pd.Timestamp("2022-12-31 23:00:00")
    test_mask  = df["time"] >= pd.Timestamp("2023-01-01 00:00:00")

    X_tr_t, y_tr_t = df.loc[train_mask, FEAT_TEMP], df.loc[train_mask, "temperature_2m_target"]
    X_te_t, y_te_t = df.loc[test_mask,  FEAT_TEMP], df.loc[test_mask,  "temperature_2m_target"]
    X_tr_h, y_tr_h = df.loc[train_mask, FEAT_HUM],  df.loc[train_mask, "relative_humidity_2m_target"]
    X_te_h, y_te_h = df.loc[test_mask,  FEAT_HUM],  df.loc[test_mask,  "relative_humidity_2m_target"]

    logger.info("ml_model: regressor train=%d  test=%d", len(X_tr_t), len(X_te_t))

    temp_model = RandomForestRegressor(n_estimators=25, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=42)
    temp_model.fit(X_tr_t, y_tr_t)

    hum_model = RandomForestRegressor(n_estimators=25, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=42)
    hum_model.fit(X_tr_h, y_tr_h)

    # Evaluate vs persistence baseline
    t_pred  = temp_model.predict(X_te_t)
    h_pred  = hum_model.predict(X_te_h)
    t_base  = df.loc[test_mask, "temperature_2m"].values
    h_base  = df.loc[test_mask, "relative_humidity_2m"].values

    t_mae   = mean_absolute_error(y_te_t, t_pred)
    t_rmse  = math.sqrt(mean_squared_error(y_te_t, t_pred))
    t_b_mae = mean_absolute_error(y_te_t, t_base)
    t_b_rmse= math.sqrt(mean_squared_error(y_te_t, t_base))

    h_mae   = mean_absolute_error(y_te_h, h_pred)
    h_rmse  = math.sqrt(mean_squared_error(y_te_h, h_pred))
    h_b_mae = mean_absolute_error(y_te_h, h_base)
    h_b_rmse= math.sqrt(mean_squared_error(y_te_h, h_base))

    t_beats = t_mae < t_b_mae
    h_beats = h_mae < h_b_mae

    logger.info(
        "ml_model: temp MAE=%.3f (baseline=%.3f) beats=%s | "
        "hum MAE=%.3f (baseline=%.3f) beats=%s",
        t_mae, t_b_mae, t_beats, h_mae, h_b_mae, h_beats,
    )

    metrics = {
        "temp_regressor": {
            "mae": round(t_mae, 4), "rmse": round(t_rmse, 4),
            "baseline_mae": round(t_b_mae, 4), "baseline_rmse": round(t_b_rmse, 4),
            "beats_baseline": t_beats,
        },
        "humidity_regressor": {
            "mae": round(h_mae, 4), "rmse": round(h_rmse, 4),
            "baseline_mae": round(h_b_mae, 4), "baseline_rmse": round(h_b_rmse, 4),
            "beats_baseline": h_beats,
        },
    }

    final_temp = temp_model if t_beats else None
    final_hum  = hum_model  if h_beats else None
    if not t_beats:
        logger.warning("ml_model: temp regressor did NOT beat baseline; falling back to chained approach")
    if not h_beats:
        logger.warning("ml_model: hum regressor did NOT beat baseline; falling back to chained approach")

    return {"temp_model": final_temp, "hum_model": final_hum, "metrics": metrics}


# ── Lazy load / save ───────────────────────────────────────────────────────────

def get_trained_models():
    """
    Lazily load models from models/*.joblib. Never train during a request or
    application startup; missing artifacts disable ML predictions with a warning.
    Returns (rain_model, temp_model, hum_model, wind_le, rain_le, threshold).
    temp_model / hum_model may be None if regressors didn't beat the baseline.
    """
    if "models" in _model_cache:
        return _model_cache["models"]

    md = _models_dir()
    paths = {
        "rain":      os.path.join(md, "rain_model.joblib"),
        "temp":      os.path.join(md, "temp_model.joblib"),
        "hum":       os.path.join(md, "hum_model.joblib"),
        "wind_le":   os.path.join(md, "wind_le.joblib"),
        "rain_le":   os.path.join(md, "rain_le.joblib"),
        "threshold": os.path.join(md, "threshold.joblib"),
    }

    all_exist = all(os.path.exists(p) for p in paths.values())

    if all_exist:
        logger.info("ml_model: loading pre-trained models from %s", md)
        rain_model = joblib.load(paths["rain"])
        temp_model = joblib.load(paths["temp"])
        hum_model  = joblib.load(paths["hum"])
        wind_le    = joblib.load(paths["wind_le"])
        rain_le    = joblib.load(paths["rain_le"])
        threshold  = joblib.load(paths["threshold"])
    else:
        logger.warning(
            "ml_model: pre-trained model files are missing in %s. "
            "ML predictions are disabled. Run 'python manage.py train_models' to train.",
            md,
        )
        rain_model = None
        temp_model = None
        hum_model  = None
        wind_le    = None
        rain_le    = None
        threshold  = 0.5

    packed = (rain_model, temp_model, hum_model, wind_le, rain_le, threshold)
    _model_cache["models"] = packed
    return packed


def run_training_pipeline(save: bool = True) -> dict:
    """
    Run the full training pipeline and optionally save to models/*.joblib.
    Returns a dict with all models and metrics.
    """
    daily_csv  = _daily_csv()
    hourly_csv = _hourly_csv()
    md         = _models_dir()

    if not os.path.exists(daily_csv):
        raise FileNotFoundError(f"Daily CSV not found: {daily_csv}. Run build_dataset.py first.")

    # ── Train classifier ───────────────────────────────────────────────────
    clf_result = train_rain_classifier(daily_csv)
    rain_model = clf_result["model"]
    wind_le    = clf_result["wind_le"]
    rain_le    = clf_result["rain_le"]
    threshold  = clf_result["threshold"]
    rain_metrics = clf_result["metrics"]

    # ── Train regressors ───────────────────────────────────────────────────
    reg_result = train_regressors(hourly_csv)
    temp_model = reg_result["temp_model"]
    hum_model  = reg_result["hum_model"]
    reg_metrics = reg_result["metrics"]

    # ── Save models ────────────────────────────────────────────────────────
    if save:
        joblib.dump(rain_model, os.path.join(md, "rain_model.joblib"), compress=3)
        joblib.dump(temp_model, os.path.join(md, "temp_model.joblib"), compress=3)
        joblib.dump(hum_model,  os.path.join(md, "hum_model.joblib"), compress=3)
        joblib.dump(wind_le,    os.path.join(md, "wind_le.joblib"))
        joblib.dump(rain_le,    os.path.join(md, "rain_le.joblib"))
        joblib.dump(threshold,  os.path.join(md, "threshold.joblib"))

        row_count = sum(1 for _ in open(daily_csv)) - 1

        full_metrics = {
            "rain_classifier": rain_metrics,
            **reg_metrics,
            "dataset_rows":    row_count,
            "train_cutoff":    "2022-12-31",
            "test_start":      "2023-01-01",
        }
        metrics_path = os.path.join(md, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(full_metrics, f, indent=2)
        logger.info("ml_model: saved metrics to %s", metrics_path)
        logger.info("ml_model: all model files saved to %s", md)

    return {
        "rain_model": rain_model,
        "temp_model": temp_model,
        "hum_model":  hum_model,
        "wind_le":    wind_le,
        "rain_le":    rain_le,
        "threshold":  threshold,
    }


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_future(
    current_temp: float,
    current_humidity: float,
    lat: float,
    lon: float,
    start_datetime: datetime.datetime,
    steps: int = 40,
):
    """
    Predict temperature and humidity for `steps` future 3-hour intervals.

    Returns:
        (ml_temps: list[float], ml_humidities: list[float])  — each of length `steps`
    """
    _, temp_model, hum_model, _, _, _ = get_trained_models()

    ml_temps, ml_humidities = [], []

    if temp_model is not None and hum_model is not None:
        t_buf = [current_temp,     current_temp,     current_temp]
        h_buf = [current_humidity, current_humidity, current_humidity]

        t_cols = [
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "Lat", "Lon",
            "temperature_2m", "temperature_2m_lag1", "temperature_2m_lag2",
        ]
        h_cols = [
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "Lat", "Lon",
            "relative_humidity_2m", "relative_humidity_2m_lag1", "relative_humidity_2m_lag2",
        ]

        for i in range(steps):
            ts = start_datetime + datetime.timedelta(hours=3 * i)
            h_sin, h_cos = _hour_features(ts.hour)
            m_sin, m_cos = _month_features(ts.month)

            t_feat = pd.DataFrame([[h_sin, h_cos, m_sin, m_cos, lat, lon, t_buf[-1], t_buf[-2], t_buf[-3]]], columns=t_cols)
            h_feat = pd.DataFrame([[h_sin, h_cos, m_sin, m_cos, lat, lon, h_buf[-1], h_buf[-2], h_buf[-3]]], columns=h_cols)

            t_pred = float(temp_model.predict(t_feat)[0])
            h_pred = float(hum_model.predict(h_feat)[0])

            ml_temps.append(round(t_pred, 1))
            ml_humidities.append(round(h_pred, 1))

            t_buf.append(t_pred)
            h_buf.append(h_pred)
    else:
        logger.debug("ml_model: predict_future using chained fallback (no lag model)")
        t, h = current_temp, current_humidity
        for _ in range(steps):
            ml_temps.append(round(t, 1))
            ml_humidities.append(round(h, 1))

    return ml_temps, ml_humidities
