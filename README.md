# 🌤️ Real-Time Weather Forecasting App

[![Django CI](https://github.com/akasha66850-cloud/weather-app/actions/workflows/ci.yml/badge.svg)](https://github.com/akasha66850-cloud/weather-app/actions/workflows/ci.yml)
[![Live on Render](https://img.shields.io/badge/Render-Live_App-46e3b7?logo=render&logoColor=white)](https://weather-app-gck2.onrender.com)

A modern, full-stack Django 5 weather application that combines **live OpenWeatherMap data** with an optimized **Machine Learning pipeline** (RandomForest) for rain prediction and temperature/humidity forecasting — featuring an interactive Chart.js forecast graph, live animated weather particle effects, dynamic city background photography, Google News weather headlines, Twilio SMS alerts, and an **interactive India Weather Map built with Folium**.

👉 **Live Production URL**: [https://weather-app-gck2.onrender.com](https://weather-app-gck2.onrender.com)

All data loads via **AJAX (no page reloads)** — search a city or use geolocation and the entire UI updates smoothly in place.

---

## 🔸 Features at a Glance

### Public vs Members Access

| Feature | Public (Logged Out) | Members (Logged In) |
|---------|:---:|:---:|
| 🔍 City Search & Autocomplete | ✅ Included | ✅ Included |
| 🖼️ Dynamic City Background Photography | ✅ Included | ✅ Included |
| 🌡️ Live Temperature, Feels-like & Stats | ✅ Included | ✅ Included |
| 🔘 °C / °F Instant Toggle | ✅ Included | ✅ Included |
| 🌦️ Live Animated Weather Particle Effects | ✅ Included | ✅ Included |
| 📈 5-Day / 24-Hour Forecast Chart | 🔒 Locked (Glass Card) | ✅ Full Chart.js access |
| 🤖 ML Regressor Comparison Line | 🔒 Locked (Glass Card) | ✅ Real vs ML overlay |
| 🌧️ ML Rain Prediction Badge & Confidence | 🔒 Locked (Sign-in pill) | ✅ Full badge & % |
| 📰 Live Google Weather News | 🔒 Locked (Glass Card) | ✅ Full feed & alerts |
| 🗺️ Interactive India Weather Map (Folium) | 🔒 Locked (401 HTML prompt) | ✅ Full interactive map |
| 📱 Twilio SMS Weather Alerts & OTP | 🔒 Locked (Glass Card) | ✅ Full subscription flow |
| ⭐ Favourites & Bookmarks | 🔒 Disabled (prompt) | ✅ Server-synced favorites |
| 📍 GPS Geolocation Detection | 🔒 Disabled (prompt) | ✅ Reverse geocoding |
| ☁️ Profile Sync & Default City | 🔒 Disabled | ✅ Cloud saved & restored |

> 🛡️ **Server-side Security:** Locked endpoints (`/api/forecast/`, `/api/news/`, `/api/reverse/`, `/api/map/`, `/api/favorites/`, `/api/profile/`, `/api/sms/*`) return HTTP 401 with JSON `{"error": "Login required", "login_url": "..."}` or a 401 HTML prompt for iframe requests. `/api/weather/` and `/autocomplete/` enforce rate limits (60 requests per 10 minutes for anonymous users) to protect external API quotas.

---

| Feature | Description |
|---------|-------------|
| 🔐 User Authentication | Case-insensitive email login, signup, remember-me (30-day session), brute-force lockout, password reset |
| 🌐 Google OAuth | One-click Sign in with Google via `django-allauth` with automatic graceful fallback |
| ☁️ Cloud Sync & Profile | Server-backed favourite cities and °C/°F unit preferences synced to user profile, with local-storage auto-import |
| 🖼️ Dynamic Background | City photograph fetched automatically via Wikipedia REST API → Commons → Picsum |
| 🌡️ Live Weather | Current temp, feels-like, humidity, pressure, wind (km/h & direction), visibility, clouds |
| 🌧️ ML Rain Prediction | RandomForest predicts rain tomorrow (Yes/No) with confidence % |
| 📊 Interactive Forecast Chart | Chart.js 4.4 line chart — toggle Temp / Humidity / Rain % and 24h / 5d views |
| 🤖 ML vs Real Comparison | Dashed orange "ML Prediction" line overlaid on real OWM temperature data |
| 🌦️ Weather Particle Effects | Animated rain, snow, sun, clouds, mist, thunder in the dark glass panel |
| 🔍 City Autocomplete | Live dropdown with 280 ms debounce powered by OpenWeather Geocoding API |
| 📍 Geolocation | One-click GPS location detection using browser Geolocation + reverse geocoding |
| ⭐ Favourites | Save favourite cities to `localStorage`; render as glass chips with quick-load and removal |
| 🔘 °C / °F Toggle | Instant temperature unit conversion across current stats and Chart.js forecast |
| 📰 Weather News | Live local weather headlines via Google News RSS with shimmer loading skeletons |
| 📱 Twilio SMS Alerts | Automated severe weather warnings & daily morning weather briefings via SMS |
| 🔒 OTP Verification | Built-in 6-digit OTP verification powered by Twilio SMS and cache storage |
| 🗺️ Interactive India Map | Folium live weather map with 15 city temperature bubbles, tooltips & modal iframe |
| 💾 Server-side Cache | Forecasts (10 min), news (15 min), and map data (10 min) cached with Django LocMemCache |
| ⚡ Full AJAX | City search updates the UI without page reloads; URL synchronizes via History API |

---

## 🗂️ Project Structure

```
weather_app/
├── manage.py
├── requirements.txt
├── .env.example                               ← Template configuration for environment variables
├── .env                                       ← Environment variables (API keys & credentials)
├── build_dataset.py                           ← Fetches India historical data from Open-Meteo
├── weather.csv                                ← Original Australian dataset (kept for reference)
├── weather_india.csv                          ← India dataset — 48,204 rows, 2015–2025 (ML training)
├── db.sqlite3                                 ← SQLite database (auth, subscribers, profiles & migrations)
│
├── accounts/                                  ← User authentication & profile management app
│   ├── __init__.py
│   ├── apps.py
│   ├── admin.py                               ← UserProfile & FavoriteCity admin models
│   ├── models.py                              ← UserProfile & FavoriteCity models
│   ├── backends.py                            ← Case-insensitive EmailBackend for authentication
│   ├── security.py                            ← IP+Email brute-force rate limiter (cache-backed)
│   ├── urls.py                                ← Login, signup, logout, password-reset routes
│   ├── views.py                               ← Custom authentication & session views
│   ├── tests.py                               ← 11 unit tests (auth, brute force, profiles, APIs)
│   └── migrations/                            ← Accounts migrations
│       └── 0001_initial.py
│
├── data/
│   ├── hourly_india.csv                       ← 1.1M-row hourly dataset for temp/humidity regressors
│   ├── raw_cache/                             ← Cached Open-Meteo JSON responses (12 city files)
│   └── legacy/
│       └── weather_australia.csv              ← Original Australian dataset backup
│
├── models/                                    ← Trained model files (generated by train_models)
│   ├── rain_model.joblib
│   ├── temp_model.joblib
│   ├── hum_model.joblib
│   ├── wind_le.joblib
│   ├── rain_le.joblib
│   ├── threshold.joblib
│   └── metrics.json                           ← Real training metrics
│
├── weatherproject/                            ← Django project configuration
│   ├── __init__.py
│   ├── settings.py                            ← allauth, auth backends, LocMemCache, .env loader
│   ├── urls.py                                ← Root URL configuration (includes accounts & allauth)
│   └── wsgi.py
│
└── weatherapp/                                ← Main Django application
    ├── __init__.py
    ├── apps.py
    ├── admin.py                               ← SubscriberAdmin registration
    ├── models.py                              ← Subscriber model (phone, city, alert settings)
    ├── sms.py                                 ← Twilio SMS & OTP verification service
    ├── urls.py                                ← API routes (weather, forecast, profile, favorites, SMS, map)
    ├── views.py                               ← Weather API endpoints, profile/favorites APIs & renderers
    ├── ml_model.py                            ← ML pipeline (classifier + 3-lag regressors + joblib)
    ├── tests.py                               ← 18 unit tests (weather & SMS features)
    ├── migrations/                            ← Database migrations
    │   └── 0001_initial.py
    ├── management/                            ← Custom Django management commands
    │   └── commands/
    │       ├── train_models.py                ← Train & save all ML models to models/
    │       ├── check_weather_alerts.py        ← Severe weather warning scanner & SMS dispatcher
    │       └── send_morning_updates.py        ← Scheduled daily morning weather SMS dispatcher
    └── templates/
        ├── registration/                      ← InsideBox styled standalone auth templates
        │   ├── base_auth.html
        │   ├── login.html
        │   ├── signup.html
        │   ├── password_reset_form.html
        │   ├── password_reset_done.html
        │   ├── password_reset_confirm.html
        │   ├── password_reset_complete.html
        │   ├── password_reset_email.txt
        │   └── password_reset_subject.txt
        └── weatherapp/
            └── index.html                     ← Glassmorphism UI, Chart.js, Modal & AJAX logic
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.13, Django 5.1 |
| **Machine Learning** | scikit-learn — `RandomForestClassifier`, `RandomForestRegressor` |
| **Data Processing** | pandas, numpy |
| **Mapping Engine** | Folium 0.20, Esri ArcGIS World Topographic Map |
| **Messaging & SMS** | Twilio REST API 9.x (Phone + Messaging Service + Verify) |
| **Weather Data** | OpenWeatherMap Current Weather (`/weather`), 5-day/3-hour Forecast (`/forecast`), Geocoding (`/reverse`) |
| **News Feed** | Google News RSS search (parsed via `xml.etree.ElementTree`) |
| **City Images** | Wikipedia REST API → Wikimedia Commons → picsum.photos |
| **Interactive Chart** | Chart.js 4.4 (CDN) |
| **Caching** | Django `LocMemCache` (10-min forecast & map cache, 15-min news cache) |
| **Frontend** | Vanilla HTML5 / CSS3 / JavaScript — glassmorphism design, Inter font, full AJAX |

---

## 🚀 Setup & Installation

### 1. Clone or Navigate to the Project

```bash
cd "C:\Users\Akash P\Downloads\weather_app"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` pins the currently installed application dependency versions.**

### 3. Environment Variables Configuration (`.env`)

Create or edit the `.env` file in the project root:

```env
# Django Configuration
DJANGO_SECRET_KEY=your_secret_key_here
DEBUG=True

# OpenWeather API Key
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Google OAuth Credentials
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# Twilio SMS Credentials
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_FROM_NUMBER=your_twilio_phone_number_here
TWILIO_MESSAGING_SERVICE_SID=your_twilio_messaging_service_sid_here

# Demo User Setup (Optional)
DEMO_PASSWORD=your_demo_password_here
```

> Local development loads `.env` from `settings.py`. Render uses its configured environment variables and does not load the local `.env` file.

### 4. Run Migrations

```bash
python manage.py migrate
```

### 5. Start the Development Server

```bash
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

---

## 🚀 Deployment on Render

This repository includes `render.yaml`, `build.sh`, and a `Procfile`. In Render, choose **New → Blueprint**, connect the GitHub repository, and apply the `render.yaml` blueprint. The build runs `pip install -r requirements.txt`, `collectstatic`, and migrations; the web process runs Gunicorn with one worker, four threads, and a 120-second timeout. Configure the environment values below in the Render service. Use placeholders as a guide and enter real credentials only in Render's environment settings.

| Variable | Render value |
| --- | --- |
| `DJANGO_SECRET_KEY` | `<generate a secret in Render>` |
| `DEBUG` | `False` |
| `ALLOWED_HOSTS` | `<service-name>.onrender.com` |
| `CSRF_TRUSTED_ORIGINS` | `https://<service-name>.onrender.com` |
| `DATABASE_URL` | `<Render internal PostgreSQL URL>` |
| `OPENWEATHER_API_KEY` | `<OpenWeather API key>` |
| `GOOGLE_CLIENT_ID` | `<Google OAuth client ID>` |
| `GOOGLE_CLIENT_SECRET` | `<Google OAuth client secret>` |
| `EMAIL_BACKEND` | `<SMTP email backend>` |
| `EMAIL_HOST` | `<SMTP host>` |
| `EMAIL_PORT` | `<SMTP port>` |
| `EMAIL_USE_TLS` | `<True or False>` |
| `EMAIL_HOST_USER` | `<SMTP username>` |
| `EMAIL_HOST_PASSWORD` | `<SMTP password>` |
| `TWILIO_ACCOUNT_SID` | `<Twilio account SID>` |
| `TWILIO_AUTH_TOKEN` | `<Twilio auth token>` |
| `TWILIO_FROM_NUMBER` | `<Twilio sender number>` |
| `TWILIO_MESSAGING_SERVICE_SID` | `<Twilio Messaging Service SID>` |
| `TWILIO_VERIFY_SID` | `<Twilio Verify Service SID, optional>` |

Google OAuth: add `https://<service-name>.onrender.com/accounts/google/login/callback/` as an **Authorized redirect URI** in the Google OAuth client. In Django Admin, open **Sites** and set the site with ID `1` to domain `<service-name>.onrender.com` (no scheme or path); this is the allauth Site domain. The local callback is `http://127.0.0.1:8000/accounts/google/login/callback/`.

For SMS jobs, create Render Cron Jobs using these commands and provide the same database, OpenWeather, and Twilio environment variables to each job:

```sh
python manage.py send_morning_updates
python manage.py check_weather_alerts
```

Run `send_morning_updates` every 15 minutes so each subscriber's configured local send time is checked. Run `check_weather_alerts` on the interval you want alerts checked; the command has a six-hour duplicate-alert cooldown. Render Cron Jobs need an appropriate paid plan.

Render Free web services spin down after 15 minutes without inbound traffic and usually take about a minute to wake on the next request. Their local filesystem is ephemeral. Free Render Postgres databases expire after 30 days, with a 14-day grace period to upgrade before deletion; use a paid database for durable production data. See [Render free instance limits](https://render.com/docs/free).

Twilio trial accounts currently expire after 30 days, can send only to verified recipients, and are limited to 50 SMS messages per day; trial messages also have trial-specific content restrictions. See [Twilio trial account restrictions](https://www.twilio.com/docs/usage/trials).

After deployment, check `https://<service-name>.onrender.com/healthz/` for a JSON health response. `models/*.joblib` are loaded on first ML use; the server logs a warning and skips ML predictions if an artifact is missing. The server never trains models at startup.

---

## 🌐 URL Routes & APIs

| Endpoint | Method | Params / Payload | Description |
|---|---|---|---|
| `/` | `GET` | `?city=<name>` | Main application shell; auto-loads city if provided |
| `/healthz/` | `GET` | — | Lightweight health check returning HTTP 200 JSON `{"status": "ok"}` for uptime monitoring |
| `/autocomplete/` | `GET` | `?q=<prefix>` | City search suggestions from OpenWeather Geocoding |
| `/api/weather/` | `GET` | `?city=<name>` | Current live weather, feels-like, stats, ML prediction & city photo |
| `/api/forecast/` | `GET` | `?city=<name>` | 5-day / 3-hour forecast entries + chained ML predictions |
| `/api/reverse/` | `GET` | `?lat=<lat>&lon=<lon>` | Reverse geocodes coordinates to `{city, country}` |
| `/api/news/` | `GET` | `?city=<name>` | Scrapes & parses up to 8 recent weather news articles (Google News RSS) |
| `/api/map/` | `GET` | `?city=<name>` | Returns interactive Folium HTML map with 15 Indian city weather bubbles |
| `/api/sms/subscribe/` | `POST` | `{"phone", "city", "morning_time"}` | Rate-limited subscription initiation & OTP dispatch |
| `/api/sms/verify/` | `POST` | `{"phone", "code"}` | Verifies OTP and activates subscription |
| `/api/sms/unsubscribe/` | `POST` | `{"phone"}` | Unsubscribes number from automated alerts |

---

## 🗺️ Interactive India Weather Map (Folium)

The application includes an interactive live weather map of India powered by **Folium** and **Esri ArcGIS World Topographic** tiles.

### Key Capabilities:
- **15 Major Indian Cities Covered**: Delhi, Mumbai, Chennai, Kolkata, Bengaluru, Hyderabad, Ahmedabad, Pune, Jaipur, Lucknow, Coimbatore, Kochi, Bhopal, Guwahati, and Chandigarh.
- **Parallel Fetching**: Weather data for all 15 cities is gathered concurrently using Python's `ThreadPoolExecutor(max_workers=8)` so the map renders in ~1.5 seconds.
- **10-Minute In-Memory Cache**: The combined weather dataset is cached using Django's `LocMemCache` (`map_weather`), speeding up subsequent loads to < 0.04 seconds.
- **Dynamic Temperature Bubbles (`folium.DivIcon`)**: 36 px circular badges color-coded by temperature:
  - 🔴 **Red (`#dc2626`)**: $\ge 38^\circ\text{C}$ (extreme heat)
  - 🟠 **Orange (`#f97316`)**: $\ge 30^\circ\text{C}$ (warm)
  - 🟡 **Yellow (`#eab308`)**: $\ge 22^\circ\text{C}$ (mild)
  - 🔵 **Blue (`#3b82f6`)**: $< 22^\circ\text{C}$ (cool)
- **Sticky Hover Tooltips (`folium.Tooltip`)**: Hovering over any bubble reveals city name, large colored temperature, feels-like, weather condition description, humidity %, and wind speed (km/h).
- **Click Popups (`folium.Popup`)**: Clicking a bubble displays an interactive card with a button **"Open full forecast &rarr;"** linking to `/?city=<name>` with `target="_top"`, reloading the main app to that city.
- **City Highlighting (`?city=<name>`)**: When passing a city parameter, the map automatically centers on that city at **zoom 8** with an amber halo glow.
- **Glassmorphism Modal & Lazy Loading**: Opened via the **"🗺️ India Weather Map"** button. The iframe source is lazy-loaded on the first click and reloaded only if the active city has changed. Supports `Esc` key dismissal, backdrop click, and full-screen mobile responsiveness.

---

## 📱 Twilio SMS Notifications & Alerts

Subscribers can sign up for morning daily forecasts and severe weather warnings directly from the UI.

### Automated Management Commands

#### 1. Severe Weather Alert Checker (`check_weather_alerts`)
Monitors verified subscribers with alerts enabled. Scans the upcoming 6-hour forecast window for:
- Thunderstorms (OWM weather IDs 200–232)
- Rain probability $\ge 70\%$ or volume $\ge 7\,\text{mm}$
- Extreme temperature ($\ge 40^\circ\text{C}$ or $\le 5^\circ\text{C}$)
- High wind speed ($\ge 50\,\text{km/h}$)
- Extreme weather / tornado / squall codes (771, 781, 600–622)

Includes a **6-hour deduplication cooldown** using `last_alert_key` and `last_alert_at`.

```powershell
# Dry run (prints alerts to console without sending SMS)
python manage.py check_weather_alerts --dry-run

# Live run
python manage.py check_weather_alerts
```

#### 2. Morning Weather Dispatcher (`send_morning_updates`)
Checks subscribers whose scheduled `morning_time` matches the current 15-minute window in `Asia/Kolkata`. Dispatches a concise, single-segment GSM-7 SMS with current temperature, today's high/low, rain chance %, and ML prediction.

```powershell
# Dry run
python manage.py send_morning_updates --dry-run

# Force send immediately to all active verified subscribers
python manage.py send_morning_updates --force
```

### High-Deliverability Routing (Indian Telecom / TRAI Compliance)
- Uses **Twilio Messaging Service** (`MG...`) to ensure compliant carrier routing.
- Formats messages in pure **GSM-7 clean ASCII** (< 160 characters) to prevent multi-segment UCS-2 fragmentation and DLT filtering.

---

## 🤖 Machine Learning Pipeline

The application incorporates a machine learning forecasting engine retrained on **India-specific historical weather data** from the [Open-Meteo Historical Archive API](https://open-meteo.com/) (CC BY 4.0):
1. **Rain Tomorrow Classifier**: Predicts rain occurrence (`Yes`/`No`) with confidence % using a tuned decision threshold.
2. **3-Lag Hourly Regressors**: Predicts temperature and humidity trajectory across 40 three-hour intervals (5-day forecast), chaining iteratively from current live weather readings.

---

### 1. Dataset

| Property | Value |
|---|---|
| **Source** | Open-Meteo Historical Archive API (ERA5 reanalysis, CC BY 4.0, zero API key) |
| **Cities (12 Major Indian Cities)** | Chennai, Mumbai, Delhi, Coimbatore, Kolkata, Bengaluru, Hyderabad, Ahmedabad, Jaipur, Kochi, Guwahati, Lucknow |
| **Date Range** | 2015-01-01 → 2025-12-30 (11 full years) |
| **Daily Rows** | **48,204 rows** (`weather_india.csv`, ≈ 4,017 rows per city) |
| **Hourly Rows** | **1,157,184 rows** (`data/hourly_india.csv`) |
| **Class Balance** | No rain: 63.1% · Rain: 36.9% |
| **Attribution** | Historical weather data © Open-Meteo.com |

---

### 2. Training Strategy & Architecture

- **Time-Based Split**: Train on dates $\le$ 2022-12-31, test on dates $\ge$ 2023-01-01 (prevents time-leakage and simulates real-world deployment).
- **Rain Classifier**: `RandomForestClassifier(class_weight="balanced", n_estimators=40, max_depth=12, min_samples_leaf=4, random_state=42)`.
  - Tuned decision threshold ($0.51$) optimized on the 2022 validation slice to maximize rain-class F1 score.
  - Compressed with `joblib.dump(..., compress=3)` to **2.17 MB** (RAM footprint: ~12 MB), preventing free-tier memory exhaustion.
- **Hourly Regressors**: `RandomForestRegressor(n_estimators=25, max_depth=12, min_samples_leaf=4, random_state=42)` with 3-lag history features:
  - Features (9): `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `Lat`, `Lon`, and 3 hourly lags (`t0`, `t-1`, `t-2`).
  - Iteratively chains predictions across 40 three-hour steps ($120\text{ hours}$ / $5\text{ days}$) anchored to real-time OWM input.
  - Compressed with `joblib.dump(..., compress=3)` to **4.34 MB** (Temp) and **4.13 MB** (Humidity).

---

### 3. Real Test-Set Metrics (`models/metrics.json`)

Evaluated on unseen test data ($\ge 2023$):

#### Rain Tomorrow Classifier
| Metric | Production Model (India, 48,204 rows, time-based split) |
|---|---|
| **Model** | `RandomForestClassifier` (balanced class weights, depth 12) |
| **Accuracy** | **0.8364** (83.6%) |
| **Precision (rain)** | **0.7776** (77.8%) |
| **Recall (rain)** | **0.7976** (79.8%) |
| **F1 Score (rain)** | **0.7875** |
| **ROC-AUC** | **0.9141** |
| **Decision Threshold** | **0.51** (tuned) |
| **Confusion Matrix** | TN=7,007 · FP=1,139 · FN=1,011 · TP=3,983 |

**Top Predictive Features**:
1. `Humidity` (0.4110)
2. `MinTemp` (0.1161)
3. `month_sin` (0.0955)
4. `month_cos` (0.0865)
5. `Pressure` (0.0576)
6. `Temp` (0.0550)

#### Temperature & Humidity 3-Lag Regressors
| Metric | Temperature Regressor | Humidity Regressor |
|---|---|---|
| **Model MAE** | **0.8362 °C** | **4.3094 %** |
| **Model RMSE** | 1.1508 °C | 5.9976 % |
| **Persistence Baseline MAE** | 2.1894 °C | 9.0061 % |
| **Beats Persistence?** | ✅ **Yes (nearly 3× better)** | ✅ **Yes (over 2× better)** |

---

### 4. Model Files & Commands

```bash
# Build India dataset from Open-Meteo Historical API (run once)
python build_dataset.py

# Train both rain classifier and 3-lag regressors and save to models/
python manage.py train_models

# Force retrain models
python manage.py train_models --force
```

Model artifacts stored in `models/`:
- `rain_model.joblib`: RandomForestClassifier
- `temp_model.joblib`: 3-lag hourly RandomForestRegressor
- `hum_model.joblib`: 3-lag hourly RandomForestRegressor
- `wind_le.joblib`: 16-point compass direction LabelEncoder
- `rain_le.joblib`: RainTomorrow LabelEncoder
- `threshold.joblib`: Tuned decision threshold (0.56)
- `metrics.json`: Evaluated test metrics and confusion matrix

---

## 📊 Interactive Forecast Chart

Powered by **Chart.js 4.4** (loaded via CDN):

| Control | Options |
|---|---|
| **Metric Toggle** | 🌡️ Temperature · 💧 Humidity · 🌧️ Rain % |
| **Time Range** | 24 h (next 8 entries) · 5 d (all 40 entries) |
| **Unit Conversion** | Seamlessly converts between °C and °F across both axes and stat cards |
| **Hover Tooltips** | Displays exact timestamps, metrics, and comparisons |
| **ML Overlay** | Dashed orange "ML Prediction" line overlaid directly on real OpenWeather temperature data |

---

## 🛠️ Testing & Verification

### Automated Unit Tests (29 tests)

Run the full Django test suite (both `weatherapp` and `accounts`):

```bash
python manage.py test
```

Or test each app individually:

```bash
# Test weather app (ML, SMS, OTP, API endpoints)
python manage.py test weatherapp --verbosity=2

# Test accounts app (email auth, session expiry, brute-force, profile/favorites APIs)
python manage.py test accounts --verbosity=2
```

| App | Test Class | Tests | What it covers |
|---|---|---|---|
| **accounts** | `AccountsAuthTests` | 12 | Signup creates user/profile, duplicate email rejection, case-insensitive email login, password validation, 30-day remember-me session, brute-force lockout (5 attempts / 15 min), open redirect protection, unauthenticated 401 guards, user-isolated favourites, profile GET/POST update, reverse-proxy `X-Forwarded-For` client IP rate limiting |
| **weatherapp** | `CompassDirectionTests` | 7 | 0°→N, 90°→E, 180°→S, 270°→W, boundary angles |
| **weatherapp** | `WindFallbackTests` | 2 | Unknown direction → fallback to `'N'` with warning log |
| **weatherapp** | `CyclicalFeaturesTests` | 2 | `month_sin/cos` and `hour_sin/cos` values |
| **weatherapp** | `PredictFutureTests` | 2 | Returns exactly `steps` values; lag model & fallback paths |
| **weatherapp** | `PredictRainTests` | 3 | Returns `'Yes'/'No'`, confidence 0–100, exception → `None` |
| **weatherapp** | `WeatherApiKeysTests` | 7 | JSON keys `rain_result`, `rain_confidence`, `ml_available` present; 400 on missing city; anonymous vs authenticated feature gating |
| **weatherapp** | `HealthzEndpointTests` | 1 | `/healthz/` returns HTTP 200 JSON `{"status": "ok"}` without calling external APIs |
| **Total** | **38 Tests** | **38** | **100% Pass Rate in ~7 seconds** |

### Manual Verification

1. **Test Weather Search & Autocomplete:**
   Type `Mumbai` or `London` into the search bar. Autocomplete suggestions will appear; selecting one loads the weather and city background photograph instantly without reloading the page.
2. **Test India Weather Map:**
   Click **"🗺️ India Weather Map"** under the search bar. The modal opens showing 15 city temperature bubbles on the ArcGIS topographic map. Hover to see stats, click to open the forecast.
3. **Test °C / °F Unit Toggle:**
   Click **°F**; observe current temperature, feels-like, min/max, and chart y-axis values convert instantly without re-fetching API data.
4. **Test Favourites:**
   Click the ⭐ star button to bookmark the current city. A glass chip appears below the search bar; clicking it switches to that city, and clicking `✕` deletes it.
5. **Test SMS Dispatch:**
   Run `python manage.py send_morning_updates --force` to send a live test weather SMS to active subscribers.
6. **Test ML Cache:** Reload `/api/weather/?city=Chennai` within 10 minutes — the second load returns instantly (models cached in memory). Restart the server and reload — model loads from `models/*.joblib` in < 1 second.

---

## 🔐 User Authentication & Personalization

The app features full user authentication powered by `django.contrib.auth` and `django-allauth`:

### Key Features
- **Email + Password Authentication:** Sign up and log in using case-insensitive email address.
- **InsideBox Clean Card Design:** Modern, accessible login/signup card with show/hide password toggle, loading spinner, and responsive layout.
- **30-Day Remember Me:** Extends session cookie duration to 30 days or clears on browser exit.
- **Brute-Force Protection:** Automatically locks out any IP + email combination exceeding 5 consecutive failed attempts for 15 minutes.
- **Password Reset:** Secure token-based password reset workflow with development console output and production SMTP configuration.
- **Google OAuth Integration:** Sign in with Google with CSRF protection and seamless account linking via `django-allauth`.
- **User Profile & Cloud Sync:** Saves favourite cities and preferred temperature unit (`°C` / `°F`) to the database. Automatically prompts to import local device favourites upon first sign-in.
- **Public Dashboard:** The weather app remains fully usable without logging in.

### Routes

| Route | Description |
|-------|-------------|
| `/accounts/login/` | Sign in card with email, password, and Google OAuth |
| `/accounts/signup/` | Account creation card |
| `/accounts/logout/` | Secure POST logout |
| `/accounts/password-reset/` | Password reset request |
| `/accounts/password-reset/confirm/<uidb64>/<token>/` | Password reset confirmation |
| `/api/profile/` | GET/POST user profile (default city, unit preference) |
| `/api/favorites/` | GET/POST/DELETE user favorite cities |

### Google OAuth Setup
1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project, then configure the **OAuth consent screen**.
3. Create **Credentials** → **OAuth client ID** → Application type: **Web application**.
4. Under **Authorized redirect URIs**, add:
   - Development: `http://127.0.0.1:8000/accounts/google/login/callback/`
   - Render: `https://<service-name>.onrender.com/accounts/google/login/callback/`
5. Copy the Client ID and Client Secret into your `.env` file:
   ```env
   GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your-google-client-secret
   ```
   *(Note: If left empty, the Google Sign-In button is automatically hidden cleanly.)*

---

## 📄 License

This project is for **educational purposes**. Weather data © OpenWeatherMap. Historical training data © [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). Map tiles © Esri & GIS User Community. City images © respective Wikipedia/Wikimedia contributors under Creative Commons licenses.
