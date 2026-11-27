"""
build_hourly_dataset.py
=======================
Fetches global hourly weather data (2018-01-01 to 2025-12-31) for ~30 cities
across diverse global climates from Open-Meteo Historical Archive API.

Variables: temperature_2m, relative_humidity_2m
Timezone : auto (local time for each city; captures local diurnal cycle)
Saves to : data/hourly_global.csv
Cache    : data/raw_cache_global/*.json
"""

import os
import time
import json
import logging
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_hourly_dataset")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "raw_cache_global")
OUTPUT_CSV = os.path.join(DATA_DIR, "hourly_global.csv")

os.makedirs(CACHE_DIR, exist_ok=True)

# 30 cities spanning tropical, subtropical, temperate, arid, continental, and subpolar climates
GLOBAL_CITIES = {
    "Chennai": (13.08, 80.27),
    "Mumbai": (19.08, 72.88),
    "Delhi": (28.61, 77.21),
    "Coimbatore": (11.02, 76.96),
    "Kolkata": (22.57, 88.36),
    "Bengaluru": (12.97, 77.59),
    "Hyderabad": (17.39, 78.49),
    "Kochi": (9.93, 76.27),
    "London": (51.51, -0.13),
    "Dubai": (25.20, 55.27),
    "New York": (40.71, -74.01),
    "Sydney": (-33.87, 151.21),
    "Singapore": (1.35, 103.82),
    "Tokyo": (35.68, 139.69),
    "Paris": (48.86, 2.35),
    "Berlin": (52.52, 13.40),
    "Moscow": (55.75, 37.62),
    "Cairo": (30.04, 31.24),
    "Cape Town": (-33.92, 18.42),
    "Sao Paulo": (-23.55, -46.63),
    "Los Angeles": (34.05, -118.24),
    "Toronto": (43.65, -79.38),
    "Bangkok": (13.75, 100.50),
    "Nairobi": (-1.29, 36.82),
    "Reykjavik": (64.15, -21.94),
    "Mexico City": (19.43, -99.13),
    "Istanbul": (41.01, 28.98),
    "Seoul": (37.57, 126.98),
    "Dhaka": (23.81, 90.41),
    "Colombo": (6.93, 79.86),
}

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_city_hourly(city: str, lat: float, lon: float) -> dict:
    cache_file = os.path.join(CACHE_DIR, f"{city.lower().replace(' ', '_')}.json")
    if os.path.exists(cache_file):
        logger.info(f"Loading {city} from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "auto",
    }

    max_retries = 6
    backoff = [5, 15, 25, 35, 45, 60]

    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {city} ({lat}, {lon}) [attempt {attempt + 1}/{max_retries}]...")
            resp = requests.get(API_URL, params=params, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                logger.info(f"Successfully fetched & cached {city}")
                return data
            elif resp.status_code == 429:
                wait_time = backoff[min(attempt, len(backoff) - 1)]
                logger.warning(f"Rate limited (429) for {city}. Backing off {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"HTTP {resp.status_code} for {city}: {resp.text[:200]}")
                time.sleep(5)
        except Exception as e:
            wait_time = backoff[min(attempt, len(backoff) - 1)]
            logger.warning(f"Error fetching {city}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to fetch data for {city} after {max_retries} attempts.")


def build_dataset():
    logger.info(f"Building global hourly dataset for {len(GLOBAL_CITIES)} cities from {START_DATE} to {END_DATE}")
    all_dfs = []

    for i, (city, (lat, lon)) in enumerate(GLOBAL_CITIES.items()):
        raw = fetch_city_hourly(city, lat, lon)
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        hums = hourly.get("relative_humidity_2m", [])
        tz = raw.get("timezone", "UTC")
        utc_offset = raw.get("utc_offset_seconds", 0)

        df = pd.DataFrame({
            "City": city,
            "Lat": lat,
            "Lon": lon,
            "timezone": tz,
            "utc_offset_seconds": utc_offset,
            "time": times,
            "temperature_2m": temps,
            "relative_humidity_2m": hums,
        })
        all_dfs.append(df)
        time.sleep(1.2)  # Respect Open-Meteo per-minute rate limits

    logger.info("Concatenating all city records...")
    combined = pd.concat(all_dfs, ignore_index=True)
    initial_len = len(combined)
    combined = combined.dropna(subset=["time", "temperature_2m", "relative_humidity_2m"])
    logger.info(f"Dropped {initial_len - len(combined)} rows with missing values.")

    logger.info(f"Saving {len(combined)} rows to {OUTPUT_CSV}...")
    combined.to_csv(OUTPUT_CSV, index=False)
    file_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    logger.info(f"Dataset successfully saved ({file_size_mb:.2f} MB, {len(combined):,} rows).")


if __name__ == "__main__":
    t0 = time.time()
    build_dataset()
    logger.info(f"Dataset build completed in {time.time() - t0:.1f}s")
