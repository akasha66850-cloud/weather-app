"""
build_dataset.py
Builds India-specific daily weather dataset (weather_india.csv) and hourly weather dataset
(data/hourly_india.csv) using Open-Meteo Historical Archive API (2015-2025).
Combines daily and hourly variables in 1 request per city and caches raw responses.
"""
import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np

# 12 Target Indian Cities with coordinates
CITIES = {
    "Chennai": (13.08, 80.27),
    "Mumbai": (19.08, 72.88),
    "Delhi": (28.61, 77.21),
    "Coimbatore": (11.02, 76.96),
    "Kolkata": (22.57, 88.36),
    "Bengaluru": (12.97, 77.59),
    "Hyderabad": (17.39, 78.49),
    "Ahmedabad": (23.02, 72.57),
    "Jaipur": (26.91, 75.79),
    "Kochi": (9.93, 76.27),
    "Guwahati": (26.14, 91.74),
    "Lucknow": (26.85, 80.95),
}

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Asia/Kolkata"
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_DIR = os.path.join("data", "raw_cache")

COMPASS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]


def degrees_to_compass(degrees):
    """Convert wind direction in degrees (0-360) to 16-point compass direction."""
    if pd.isna(degrees):
        return "N"
    val = (float(degrees) + 11.25) % 360
    idx = int(val // 22.5)
    return COMPASS_16[idx % 16]


def fetch_city_data(city, lat, lon):
    """Fetch combined daily + hourly data for a city, using disk cache if available."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{city.lower()}_2015_2025.json")

    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "daily" in data and "hourly" in data:
                print(f"  [Cache Hit] Loaded {city} from disk.")
                return data
        except Exception:
            pass

    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "surface_pressure_mean",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "precipitation_sum"
    ]
    hourly_vars = ["temperature_2m", "relative_humidity_2m"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(daily_vars),
        "hourly": ",".join(hourly_vars),
        "timezone": TIMEZONE
    }

    max_retries = 6
    for attempt in range(max_retries):
        try:
            print(f"  [API Request] Fetching {city} (Attempt {attempt+1}/{max_retries})...")
            resp = requests.get(BASE_URL, params=params, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
            elif resp.status_code == 429:
                wait_time = 25 + attempt * 10
                print(f"  [429 Rate Limit] Cooldown sleeping {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [HTTP {resp.status_code}] Sleeping 5s before retry...")
                time.sleep(5)
        except Exception as e:
            print(f"  [Network error: {e}] Sleeping 5s before retry...")
            time.sleep(5)

    raise RuntimeError(f"Failed to fetch {city} after {max_retries} attempts.")


def build_all_datasets():
    print("=" * 65)
    print("BUILDING INDIA WEATHER DATASETS (Open-Meteo Historical 2015-2025)")
    print("=" * 65)

    daily_frames = []
    hourly_frames = []

    for idx, (city, (lat, lon)) in enumerate(CITIES.items(), 1):
        print(f"\n[{idx}/{len(CITIES)}] Processing {city} ({lat}, {lon})...")
        data = fetch_city_data(city, lat, lon)

        # ── 1. Process Daily Data ──────────────────────────────────────────
        daily = data.get("daily", {})
        if daily and "time" in daily:
            df_city = pd.DataFrame({
                "City": city,
                "Lat": lat,
                "Lon": lon,
                "Date": daily["time"],
                "MaxTemp": daily["temperature_2m_max"],
                "MinTemp": daily["temperature_2m_min"],
                "Temp": daily["temperature_2m_mean"],
                "Humidity": daily["relative_humidity_2m_mean"],
                "Pressure": daily["surface_pressure_mean"],
                "WindGustSpeed": daily["wind_gusts_10m_max"],
                "wind_direction_deg": daily["wind_direction_10m_dominant"],
                "precipitation_sum": daily["precipitation_sum"],
            })

            # Add Month column
            df_city["Month"] = pd.to_datetime(df_city["Date"]).dt.month

            # Convert wind direction to 16-point compass
            df_city["WindGustDir"] = df_city["wind_direction_deg"].apply(degrees_to_compass)
            df_city.drop(columns=["wind_direction_deg"], inplace=True)

            # RainTomorrow: 'Yes' if next day's precipitation_sum >= 1.0 mm else 'No'
            next_day_precip = df_city["precipitation_sum"].shift(-1)
            df_city["RainTomorrow"] = np.where(next_day_precip >= 1.0, "Yes", "No")

            # Drop each city's last row (shift is NaN)
            df_city = df_city.iloc[:-1].copy()
            df_city.dropna(inplace=True)

            print(f"   Daily: {len(df_city):,} rows")
            daily_frames.append(df_city)

        # ── 2. Process Hourly Data ─────────────────────────────────────────
        hourly = data.get("hourly", {})
        if hourly and "time" in hourly:
            df_h = pd.DataFrame({
                "City": city,
                "Lat": lat,
                "Lon": lon,
                "time": hourly["time"],
                "temperature_2m": hourly["temperature_2m"],
                "relative_humidity_2m": hourly["relative_humidity_2m"]
            })
            df_h.dropna(inplace=True)
            print(f"   Hourly: {len(df_h):,} rows")
            hourly_frames.append(df_h)

        # Courtesy sleep between calls
        time.sleep(2.5)

    # ── Save Daily Dataset ─────────────────────────────────────────────────
    df_daily_total = pd.concat(daily_frames, ignore_index=True)
    daily_out = "weather_india.csv"
    df_daily_total.to_csv(daily_out, index=False)
    daily_mb = os.path.getsize(daily_out) / (1024 * 1024)

    # ── Save Hourly Dataset ────────────────────────────────────────────────
    df_hourly_total = pd.concat(hourly_frames, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    hourly_out = os.path.join("data", "hourly_india.csv")
    df_hourly_total.to_csv(hourly_out, index=False)
    hourly_mb = os.path.getsize(hourly_out) / (1024 * 1024)

    # ── Summary Report ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DATASET BUILD COMPLETE")
    print("=" * 65)
    print(f"1. Daily Dataset: {daily_out}")
    print(f"   Total rows: {len(df_daily_total):,}")
    print(f"   File size:  {daily_mb:.2f} MB")
    print(f"   Date range: {df_daily_total['Date'].min()} to {df_daily_total['Date'].max()}")
    print("\n   Class balance (RainTomorrow):")
    counts = df_daily_total["RainTomorrow"].value_counts()
    pcts = df_daily_total["RainTomorrow"].value_counts(normalize=True) * 100
    for label in counts.index:
        print(f"     {label}: {counts[label]:,} ({pcts[label]:.2f}%)")

    print(f"\n2. Hourly Dataset: {hourly_out}")
    print(f"   Total rows: {len(df_hourly_total):,}")
    print(f"   File size:  {hourly_mb:.2f} MB")
    print("=" * 65)

    return df_daily_total, df_hourly_total


if __name__ == "__main__":
    t0 = time.time()
    build_all_datasets()
    print(f"Finished in {round(time.time() - t0, 1)} seconds.")
