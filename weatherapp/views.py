import os
import re
import json
import html
import logging
import datetime
import pytz
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import email.utils
from urllib.parse import quote as url_quote
from concurrent.futures import ThreadPoolExecutor
import folium

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.core.cache import cache
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import ensure_csrf_cookie

logger = logging.getLogger(__name__)

from .ml_model import get_trained_models, get_compass_direction, predict_future, _encode_wind
from .models import Subscriber
from .sms import start_verification, check_verification, send_sms
from accounts.security import api_login_required, check_rate_limit

# ── API key & base URL ────────────────────────────────────────────────────────
# Loaded from environment variable or .env file
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# ── Module-level image cache (avoids repeated Wikipedia calls) ────────────────
_image_cache: dict = {}


# ═══════════════════════════════════════════════════════════════════════════════
#  get_city_image  –  3-tier city photograph fetcher
# ═══════════════════════════════════════════════════════════════════════════════
def get_city_image(city):
    """
    Return a dict with keys: image_url, author, license, source_url.
    Strategy:
      1. Wikipedia REST summary (best quality, real city photos)
      2. Wikimedia Commons generator search (skyline / landscape fallback)
      3. picsum.photos (always works, deterministic per city seed)
    """
    cache_key = city.strip().lower()
    if cache_key in _image_cache:
        return _image_cache[cache_key]

    result = {'image_url': None, 'author': None, 'license': None, 'source_url': None}
    _HEADERS = {'User-Agent': 'WeatherDjangoApp/1.0 (education project)'}
    _EXCLUDE = {'svg', 'logo', 'map', 'flag', 'icon', 'coat', 'seal',
                'emblem', 'symbol', 'diagram', 'sign', 'chart', 'graph'}

    # ── Tier 1: Wikipedia REST API ──────────────────────────────────────────
    for query in [city.split(',')[0].strip(), city.strip()]:
        try:
            wiki_url = (
                f"https://en.wikipedia.org/api/rest_v1/page/summary/"
                f"{url_quote(query, safe='')}"
            )
            resp = requests.get(wiki_url, timeout=7, headers=_HEADERS)
            if resp.ok:
                data = resp.json()
                img_obj = data.get('originalimage') or data.get('thumbnail') or {}
                img_url = img_obj.get('source', '')
                if img_url and not img_url.lower().endswith('.svg'):
                    source_url = (
                        data.get('content_urls', {})
                            .get('desktop', {})
                            .get('page', '')
                    )
                    result = {
                        'image_url': img_url,
                        'author': 'Wikipedia contributors',
                        'license': 'CC BY-SA',
                        'source_url': source_url,
                    }
                    _image_cache[cache_key] = result
                    return result
        except Exception:
            pass

    # ── Tier 2: Wikimedia Commons generator search ──────────────────────────
    for term in [f"{city} skyline", f"{city} city", f"{city} landscape"]:
        try:
            resp = requests.get(
                'https://commons.wikimedia.org/w/api.php',
                params={
                    'action': 'query', 'generator': 'search',
                    'gsrsearch': term, 'gsrnamespace': '6', 'gsrlimit': '20',
                    'prop': 'imageinfo', 'iiprop': 'url|extmetadata|mime',
                    'iiurlwidth': '1600', 'format': 'json',
                },
                timeout=8, headers=_HEADERS,
            )
            pages = resp.json().get('query', {}).get('pages', {})
            for page in sorted(pages.values(), key=lambda p: p.get('index', 999)):
                title = page.get('title', '').lower()
                if any(kw in title for kw in _EXCLUDE):
                    continue
                if not any(title.endswith(ext) for ext in ('.jpg', '.jpeg', '.png')):
                    continue
                for info in page.get('imageinfo', []):
                    mime = info.get('mime', '')
                    if not mime.startswith('image/') or 'svg' in mime:
                        continue
                    thumb = info.get('thumburl') or info.get('url')
                    if not thumb:
                        continue
                    extmeta = info.get('extmetadata', {})
                    author = re.sub(
                        r'<[^>]+>', '',
                        extmeta.get('Artist', {}).get('value', 'Unknown')
                    ).strip() or 'Unknown'
                    lic = (
                        extmeta.get('LicenseShortName', {}).get('value')
                        or extmeta.get('License', {}).get('value', '')
                    )
                    result = {
                        'image_url': thumb,
                        'author': author,
                        'license': lic,
                        'source_url': info.get('descriptionurl', ''),
                    }
                    _image_cache[cache_key] = result
                    return result
        except Exception:
            pass

    # ── Tier 3: picsum.photos (guaranteed fallback) ─────────────────────────
    seed = re.sub(r'\W+', '', city.lower())[:20] or 'city'
    result = {
        'image_url': f'https://picsum.photos/seed/{seed}/1920/1080',
        'author': 'picsum.photos', 'license': 'CC0', 'source_url': '',
    }
    _image_cache[cache_key] = result
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  _current_weather_json  –  internal helper: fetch OWM current weather
# ═══════════════════════════════════════════════════════════════════════════════
def _current_weather_json(city, api_key):
    """
    Call OWM /weather endpoint and return a normalised dict.
    Raises ValueError with a human-readable message on 404 / 401.
    Raises requests.Timeout on timeout.
    """
    url = f"{BASE_URL}weather?q={city}&appid={api_key}&units=metric"
    resp = requests.get(url, timeout=8)

    if resp.status_code == 404:
        raise ValueError(f"City '{city}' not found. Please check the spelling.")
    if resp.status_code == 401:
        raise ValueError("Invalid API key. Please set OPENWEATHER_API_KEY correctly.")
    resp.raise_for_status()

    data = resp.json()

    # Local time via UTC offset supplied by OWM
    tz_offset = data.get('timezone', 0)      # seconds east of UTC
    utc_now   = datetime.datetime.utcnow()
    now       = utc_now + datetime.timedelta(seconds=tz_offset)

    _h    = now.strftime('%I').lstrip('0') or '12'
    _m    = now.strftime('%M')
    _ampm = 'a.m.' if now.strftime('%p').lower() == 'am' else 'p.m.'
    current_datetime = f"{now.strftime('%b.')} {now.day}, {now.year}, {_h}:{_m} {_ampm}"

    temp          = round(data['main']['temp'])
    feels_like    = round(data['main']['feels_like'])
    temp_min      = round(data['main']['temp_min'])
    temp_max      = round(data['main']['temp_max'])
    humidity      = data['main']['humidity']
    pressure      = data['main']['pressure']
    visibility    = data.get('visibility', 'N/A')
    clouds        = data.get('clouds', {}).get('all', 0)
    description   = data['weather'][0]['description']
    icon          = data['weather'][0]['icon']
    wind_speed    = data.get('wind', {}).get('speed', 0)
    wind_kph      = round(wind_speed * 3.6, 2)
    country       = data.get('sys', {}).get('country', '')
    wind_gust_dir = data.get('wind', {}).get('deg', 0)
    wind_direction = get_compass_direction(wind_gust_dir)

    return {
        'city':             data.get('name', city),
        'country':          country,
        'temp':             temp,
        'feels_like':       feels_like,
        'temp_min':         temp_min,
        'temp_max':         temp_max,
        'humidity':         humidity,
        'pressure':         pressure,
        'visibility':       visibility,
        'clouds':           clouds,
        'description':      description,
        'icon':             icon,
        'wind_gust_dir':    wind_gust_dir,
        'wind_gust_speed':  wind_speed,
        'wind_kph':         wind_kph,
        'wind_direction':   wind_direction,
        'current_datetime': current_datetime,
        'tz_offset':        tz_offset,
        'lat':              data.get('coord', {}).get('lat', 20.0),
        'lon':              data.get('coord', {}).get('lon', 78.0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  _run_ml_rain  –  RandomForest rain prediction
# ═══════════════════════════════════════════════════════════════════════════════
def _run_ml_rain(weather):
    """
    Given a weather dict (from _current_weather_json), run the ML rain
    classifier and return (rain_result, rain_confidence, ml_available).

    Uses the new India model features:
      MinTemp, MaxTemp, Temp, Humidity, Pressure, WindGustSpeed,
      WindGustDir_enc (LabelEncoder), Month_sin/cos, Lat, Lon
    """
    import math as _math

    try:
        rain_model, temp_model, hum_model, wind_le, rain_le, threshold = get_trained_models()

        wd = weather.get('wind_direction', 'N')
        wind_enc = _encode_wind(wd, wind_le)

        # Month from local time (use tz_offset baked into weather dict if available)
        now_month = datetime.datetime.utcnow().month
        month_sin = _math.sin(2 * _math.pi * now_month / 12)
        month_cos = _math.cos(2 * _math.pi * now_month / 12)

        lat = weather.get('lat', 20.0)
        lon = weather.get('lon', 78.0)

        current_df = pd.DataFrame({
            'MinTemp':       [weather['temp_min']],
            'MaxTemp':       [weather['temp_max']],
            'Temp':          [weather['temp']],
            'Humidity':      [weather['humidity']],
            'Pressure':      [weather['pressure']],
            'WindGustSpeed': [weather['wind_gust_speed']],
            'WindGustDir_enc': [wind_enc],
            'month_sin':     [month_sin],
            'month_cos':     [month_cos],
            'Lat':           [lat],
            'Lon':           [lon],
        })

        # Identify the "Yes" class index
        rain_class_idx = list(rain_le.classes_).index('Yes')
        proba = rain_model.predict_proba(current_df)[0]
        rain_prob = float(proba[rain_class_idx])

        # Apply tuned threshold
        rain_result = 'Yes' if rain_prob >= threshold else 'No'
        # Confidence = probability of the predicted class
        rain_conf = round(rain_prob * 100 if rain_result == 'Yes' else (1 - rain_prob) * 100, 1)

        return rain_result, rain_conf, True
    except Exception:
        return None, None, False


from django.middleware.csrf import get_token

# ═══════════════════════════════════════════════════════════════════════════════
#  home  –  returns empty shell; JS does all the work
# ═══════════════════════════════════════════════════════════════════════════════
@ensure_csrf_cookie
def home(request):
    """
    Serve the app shell. Any ?city= param is forwarded to JS as INITIAL_CITY.
    If no ?city= param and user is logged in with a default_city, use that.
    Also injects window.APP context via json_script.
    """
    initial_city = request.GET.get('city', '').strip()
    is_auth = request.user.is_authenticated
    if not initial_city and is_auth:
        profile = getattr(request.user, 'profile', None)
        if profile and profile.default_city:
            initial_city = profile.default_city

    profile_city = ''
    if is_auth:
        profile = getattr(request.user, 'profile', None)
        if profile and profile.default_city:
            profile_city = profile.default_city

    app_context = {
        'authenticated': is_auth,
        'user': {
            'name': request.user.get_full_name() or request.user.email,
            'email': request.user.email,
        } if is_auth else None,
        'profileDefaultCity': profile_city,
        'loginUrl': f"{settings.LOGIN_URL}?next={url_quote(request.get_full_path())}",
        'signupUrl': f"/accounts/signup/?next={url_quote(request.get_full_path())}",
        'csrfToken': get_token(request),
    }

    return render(request, 'weatherapp/index.html', {
        'initial_city': initial_city,
        'app_context': app_context,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  weather_api  –  GET /api/weather/?city=<name>
# ═══════════════════════════════════════════════════════════════════════════════
@require_GET
def weather_api(request):
    """
    Return current weather + ML rain prediction (if authenticated) + city image as JSON.
    Used by the frontend fetch() on every city search.
    Publicly accessible with rate limiting: 60/10m for anonymous, 300/10m for authenticated.
    """
    is_auth = request.user.is_authenticated
    limit = 300 if is_auth else 60
    if not check_rate_limit(request, 'weather', limit=limit, timeout_seconds=600):
        return JsonResponse({'error': 'Rate limit exceeded. Please wait a few minutes.'}, status=429)

    city = request.GET.get('city', '').strip()
    if not city:
        return JsonResponse({'error': 'city parameter is required'}, status=400)

    try:
        weather = _current_weather_json(city, OPENWEATHER_API_KEY)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=404)
    except requests.Timeout:
        return JsonResponse({'error': 'Weather service timed out. Try again.'}, status=503)
    except Exception as e:
        return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

    # City background image (non-fatal, public)
    try:
        city_image = get_city_image(weather['city'])
    except Exception:
        city_image = {'image_url': None, 'author': None, 'license': None, 'source_url': None}

    # ML rain prediction: only for authenticated users
    if is_auth:
        rain_result, rain_conf, ml_available = _run_ml_rain(weather)
        return JsonResponse({
            **weather,
            'rain_result':      rain_result,
            'rain_confidence':  rain_conf,
            'ml_available':     ml_available,
            'city_image':       city_image,
            'authenticated':    True,
        })
    else:
        # Anonymous users get basic weather without ML prediction fields
        return JsonResponse({
            **weather,
            'ml_available':     False,
            'city_image':       city_image,
            'authenticated':    False,
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  forecast_api  –  GET /api/forecast/?city=<name>
# ═══════════════════════════════════════════════════════════════════════════════
@require_GET
@api_login_required
def forecast_api(request):
    """
    Return OWM 5-day/3-hour forecast as JSON, plus ML-chained temperature &
    humidity predictions for comparison. Responses are cached 10 minutes.
    """
    city = request.GET.get('city', '').strip()
    if not city:
        return JsonResponse({'error': 'city parameter is required'}, status=400)

    cache_key = 'forecast_' + re.sub(r'\W+', '_', city.lower())
    cached = cache.get(cache_key)
    if cached:
        return JsonResponse(cached)

    # ── Fetch from OWM ───────────────────────────────────────────────────────
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=10)
    except requests.Timeout:
        return JsonResponse({'error': 'Forecast service timed out. Try again.'}, status=503)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    if resp.status_code == 404:
        return JsonResponse({'error': f"City '{city}' not found."}, status=404)
    if resp.status_code == 401:
        return JsonResponse({'error': 'Invalid API key.'}, status=401)
    if not resp.ok:
        return JsonResponse({'error': f'OWM error {resp.status_code}'}, status=502)

    data = resp.json()
    tz_offset = data.get('city', {}).get('timezone', 0)   # seconds east of UTC
    country   = data.get('city', {}).get('country', '')
    city_name = data.get('city', {}).get('name', city)

    # ── Build entries list ───────────────────────────────────────────────────
    entries = []
    for item in data.get('list', []):
        utc_dt = datetime.datetime.utcfromtimestamp(item['dt'])
        local_dt = utc_dt + datetime.timedelta(seconds=tz_offset)
        timestamp = local_dt.strftime('%Y-%m-%d %H:%M')

        rain_3h = item.get('rain', {}).get('3h', 0.0)
        entries.append({
            'timestamp':        timestamp,
            'dt':               item['dt'],
            'temp':             round(item['main']['temp'], 1),
            'feels_like':       round(item['main']['feels_like'], 1),
            'humidity':         item['main']['humidity'],
            'rain_probability': round(item.get('pop', 0) * 100, 1),
            'rain_3h':          rain_3h,
            'wind_speed':       round(item.get('wind', {}).get('speed', 0) * 3.6, 1),
            'description':      item['weather'][0]['description'],
            'icon':             item['weather'][0]['icon'],
        })

    # ── ML prediction (3-lag hourly model) ──────────────────────────────────
    ml_temps      = []
    ml_humidities = []
    if entries:
        try:
            city_coord = data.get('city', {}).get('coord', {})
            lat = float(city_coord.get('lat', 20.0))
            lon = float(city_coord.get('lon', 78.0))
            current_temp = float(entries[0]['temp'])
            current_hum  = float(entries[0]['humidity'])
            start_dt = datetime.datetime.utcfromtimestamp(entries[0]['dt'])
            ml_temps, ml_humidities = predict_future(
                current_temp, current_hum, lat, lon, start_dt, steps=len(entries)
            )
        except Exception as e:
            logger.exception("forecast_api: ML prediction failed: %s", e)
            ml_temps      = []
            ml_humidities = []

    result = {
        'city':           city_name,
        'country':        country,
        'entries':        entries,
        'ml_temps':       ml_temps,
        'ml_humidities':  ml_humidities,
    }

    cache.set(cache_key, result, 600)   # cache for 10 minutes
    return JsonResponse(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  city_autocomplete  –  GET /autocomplete/?q=<query>
# ═══════════════════════════════════════════════════════════════════════════════
def city_autocomplete(request):
    """Return JSON list of city name suggestions via OWM Geocoding API."""
    is_auth = request.user.is_authenticated
    limit = 300 if is_auth else 60
    if not check_rate_limit(request, 'autocomplete', limit=limit, timeout_seconds=600):
        return JsonResponse({'error': 'Rate limit exceeded. Please wait a few minutes.'}, status=429)

    q = request.GET.get('q', '').strip()
    if len(q) < 2:
        return JsonResponse([], safe=False)
    try:
        url = (
            f"https://api.openweathermap.org/geo/1.0/direct"
            f"?q={q}&limit=6&appid={OPENWEATHER_API_KEY}"
        )
        resp = requests.get(url, timeout=4)
        data = resp.json()
        seen = set()
        suggestions = []
        for item in data:
            name    = item.get('name', '')
            country = item.get('country', '')
            state   = item.get('state', '')
            label   = f"{name}, {state}, {country}" if state else f"{name}, {country}"
            if label not in seen:
                seen.add(label)
                suggestions.append(label)
        return JsonResponse(suggestions, safe=False)
    except Exception:
        return JsonResponse([], safe=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  reverse_geocode  –  GET /api/reverse/?lat=&lon=
# ═══════════════════════════════════════════════════════════════════════════════
@require_GET
@api_login_required
def reverse_geocode(request):
    """
    Convert GPS coordinates to a city name using OWM reverse geocoding.
    Returns JSON {city, country} on success.
    Called by the frontend geolocation button.
    """
    lat_str = request.GET.get('lat', '').strip()
    lon_str = request.GET.get('lon', '').strip()

    # ── Validate parameters ──────────────────────────────────────────────────
    if not lat_str or not lon_str:
        return JsonResponse({'error': 'lat and lon parameters are required.'}, status=400)
    try:
        lat = float(lat_str)
        lon = float(lon_str)
    except ValueError:
        return JsonResponse({'error': 'lat and lon must be valid numbers.'}, status=400)
    if not (-90 <= lat <= 90):
        return JsonResponse({'error': 'lat must be between -90 and 90.'}, status=400)
    if not (-180 <= lon <= 180):
        return JsonResponse({'error': 'lon must be between -180 and 180.'}, status=400)

    # ── Call OWM reverse geocoding API ───────────────────────────────────────
    url = (
        f"http://api.openweathermap.org/geo/1.0/reverse"
        f"?lat={lat}&lon={lon}&limit=1&appid={OPENWEATHER_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=6)
    except requests.Timeout:
        return JsonResponse({'error': 'Geocoding service timed out. Try again.'}, status=503)
    except Exception as e:
        return JsonResponse({'error': f'Network error: {str(e)}'}, status=500)

    if resp.status_code == 401:
        return JsonResponse({'error': 'Invalid API key.'}, status=401)
    if not resp.ok:
        return JsonResponse({'error': f'Geocoding error {resp.status_code}.'}, status=502)

    data = resp.json()
    if not data:
        return JsonResponse({'error': 'No city found for these coordinates.'}, status=404)

    place = data[0]
    return JsonResponse({
        'city':    place.get('name', ''),
        'country': place.get('country', ''),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  weather_news  –  GET /api/news/?city=<name>
# ═══════════════════════════════════════════════════════════════════════════════
@require_GET
@api_login_required
def weather_news(request):
    """
    Fetch recent weather headlines for a city from Google News RSS.
    Returns JSON: {city, articles: [{title, source, link, published}]}
    Caches results for 15 minutes.
    """
    city = request.GET.get('city', '').strip()
    if not city:
        return JsonResponse({'error': 'city parameter is required.'}, status=400)

    cache_key = 'news_' + re.sub(r'\W+', '_', city.lower())
    cached = cache.get(cache_key)
    if cached:
        return JsonResponse(cached)

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        )
    }

    def _fetch_rss(query_str):
        url = (
            f"https://news.google.com/rss/search"
            f"?q={url_quote(query_str)}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        resp = requests.get(url, headers=headers, timeout=6)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        return root.findall('.//item')

    try:
        primary_query = f"{city} (weather OR rain OR cyclone OR forecast) when:3d"
        items = _fetch_rss(primary_query)

        # Fallback if fewer than 3 articles come back (e.g. small towns)
        if len(items) < 3:
            fallback_query = "weather alert rain forecast when:2d"
            fallback_items = _fetch_rss(fallback_query)
            seen_links = set((it.findtext('link') or '').strip() for it in items)
            for it in fallback_items:
                lnk = (it.findtext('link') or '').strip()
                if lnk not in seen_links:
                    items.append(it)
                    seen_links.add(lnk)
    except Exception:
        return JsonResponse({'error': 'Failed to fetch or parse news feed.'}, status=502)

    articles = []
    for it in items[:8]:
        raw_title = (it.findtext('title') or '').strip()
        source_el = it.find('source')
        source = (source_el.text if source_el is not None and source_el.text else '').strip()
        link = (it.findtext('link') or '').strip()
        pub_str = it.findtext('pubDate')

        # Convert published RFC date to ISO string
        iso_pub = ''
        if pub_str:
            try:
                iso_pub = email.utils.parsedate_to_datetime(pub_str).isoformat()
            except Exception:
                iso_pub = ''

        # Strip trailing " - <Source>" that Google appends to titles
        if source and raw_title.endswith(f" - {source}"):
            clean_title = raw_title[:-len(f" - {source}")].strip()
        else:
            m = re.search(r'\s+-\s+([^-]+)$', raw_title)
            if m:
                if not source:
                    source = m.group(1).strip()
                clean_title = raw_title[:m.start()].strip()
            else:
                clean_title = raw_title

        articles.append({
            'title': clean_title or raw_title,
            'source': source or 'News',
            'link': link,
            'published': iso_pub,
        })

    result = {
        'city': city,
        'articles': articles,
    }

    cache.set(cache_key, result, 900)  # 15 minutes cache
    return JsonResponse(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  TWILIO SMS SUBSCRIBER APIS (Subscribe, Verify OTP, Unsubscribe)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_request_data(request):
    """Helper to parse JSON or form-encoded POST body."""
    if request.content_type == 'application/json':
        try:
            return json.loads(request.body.decode('utf-8'))
        except Exception:
            return {}
    return request.POST


@require_POST
@api_login_required
def sms_subscribe(request):
    """
    POST /api/sms/subscribe/
    Body: {phone, city, morning_time, morning_enabled, alerts_enabled}
    Validates E.164 phone, validates city exists via OWM, rate limits 5/hr/IP,
    registers subscriber (unverified), and dispatches Twilio Verify OTP.
    """
    # ── 1. Rate Limiting by IP (max 5 attempts per hour) ─────────────────────
    ip = request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR', '')).split(',')[0].strip()
    rate_key = f"rl_sms_sub_{ip}"
    attempts = cache.get(rate_key, 0)
    if attempts >= 5:
        return JsonResponse(
            {'error': 'Too many subscription attempts from this IP. Please try again in 1 hour.'},
            status=429
        )

    data = _parse_request_data(request)
    phone = data.get('phone', '').strip().replace(' ', '').replace('-', '')
    city = data.get('city', '').strip()
    morning_time_str = data.get('morning_time', '07:00').strip()
    morning_enabled = bool(data.get('morning_enabled', True))
    alerts_enabled = bool(data.get('alerts_enabled', True))

    # ── 2. Validate E.164 Phone Format ───────────────────────────────────────
    # E.164: + followed by 7 to 15 digits
    if not re.match(r'^\+[1-9]\d{6,14}$', phone):
        return JsonResponse(
            {'error': 'Invalid phone number format. Please provide full international format with country code (e.g. +919876543210).'},
            status=400
        )

    # ── 3. Validate City Exists via OpenWeatherMap ────────────────────────────
    if not city:
        return JsonResponse({'error': 'City is required for weather alerts.'}, status=400)
    try:
        _current_weather_json(city, OPENWEATHER_API_KEY)
    except ValueError as e:
        return JsonResponse({'error': f"Invalid city '{city}': could not find weather station."}, status=400)
    except Exception as e:
        return JsonResponse({'error': f"Could not verify city with weather service: {str(e)}"}, status=503)

    # ── 4. Parse Morning Preferred Time ──────────────────────────────────────
    parsed_time = datetime.time(7, 0)
    if morning_time_str:
        try:
            parts = morning_time_str.split(':')
            parsed_time = datetime.time(int(parts[0]), int(parts[1]))
        except Exception:
            parsed_time = datetime.time(7, 0)

    # ── 5. Record/Update Unverified Subscriber ───────────────────────────────
    subscriber, _ = Subscriber.objects.update_or_create(
        phone=phone,
        defaults={
            'city': city,
            'morning_time': parsed_time,
            'morning_enabled': morning_enabled,
            'alerts_enabled': alerts_enabled,
            'verified': False,
        }
    )

    # Increment rate limit counter
    cache.set(rate_key, attempts + 1, 3600)

    # ── 6. Trigger Twilio Verify OTP ─────────────────────────────────────────
    success, err = start_verification(phone)
    if not success:
        return JsonResponse(
            {'error': f"Failed to send verification code via Twilio: {err}"},
            status=502
        )

    return JsonResponse({
        'status': 'pending',
        'phone': phone,
        'city': city,
        'message': f"Verification code sent to {phone}. Please enter the OTP to confirm your subscription.",
    })


@require_POST
@api_login_required
def sms_verify(request):
    """
    POST /api/sms/verify/
    Body: {phone, code}
    Verifies OTP via Twilio Verify. If approved, marks subscriber.verified = True.
    """
    data = _parse_request_data(request)
    phone = data.get('phone', '').strip().replace(' ', '').replace('-', '')
    code = data.get('code', '').strip()

    if not phone or not code:
        return JsonResponse({'error': 'Phone number and verification code are required.'}, status=400)

    subscriber = Subscriber.objects.filter(phone=phone).first()
    if not subscriber:
        return JsonResponse({'error': 'No pending subscription found for this phone number. Please sign up first.'}, status=404)

    # Verify code with Twilio Verify
    approved, err = check_verification(phone, code)
    if not approved:
        return JsonResponse({'error': err or 'Invalid or expired verification code.'}, status=400)

    # Mark verified
    subscriber.verified = True
    subscriber.save(update_fields=['verified'])

    # Optional welcome confirmation SMS
    welcome_text = (
        f"✅ Subscribed to Weather alerts for {subscriber.city}! "
        f"{'Morning update at ' + subscriber.morning_time.strftime('%H:%M') + '. ' if subscriber.morning_enabled else ''}"
        f"{'Severe weather warnings active. ' if subscriber.alerts_enabled else ''}"
        "Reply STOP to unsubscribe."
    )
    send_sms(subscriber.phone, welcome_text)

    return JsonResponse({
        'status': 'verified',
        'phone': subscriber.phone,
        'city': subscriber.city,
        'message': f"You're subscribed! Weather alerts for {subscriber.city} are now active for {subscriber.phone}.",
    })


@require_POST
@api_login_required
def sms_unsubscribe(request):
    """
    POST /api/sms/unsubscribe/
    Body: {phone}
    Removes or disables the subscriber from all automated alerts.
    """
    data = _parse_request_data(request)
    phone = data.get('phone', '').strip().replace(' ', '').replace('-', '')

    if not phone:
        return JsonResponse({'error': 'Phone number is required to unsubscribe.'}, status=400)

    sub = Subscriber.objects.filter(phone=phone).first()
    if sub:
        sub.delete()
        # Confirmation SMS
        send_sms(phone, "You have been unsubscribed from all weather SMS updates. Reply START to resubscribe.")
        return JsonResponse({
            'status': 'unsubscribed',
            'message': f"{phone} has been successfully unsubscribed from all weather SMS alerts.",
        })
    else:
        return JsonResponse({
            'status': 'unsubscribed',
            'message': f"No active subscription found for {phone}.",
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE INDIA WEATHER MAP (Folium)
# ═══════════════════════════════════════════════════════════════════════════════

MAP_CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Coimbatore": (11.0168, 76.9558),
    "Kochi": (9.9312, 76.2673),
    "Bhopal": (23.2599, 77.4126),
    "Guwahati": (26.1445, 91.7362),
    "Chandigarh": (30.7333, 76.7794),
}


def _temp_color(temp):
    """Return hex color code according to temperature tier."""
    if temp >= 38:
        return "#dc2626"  # Red
    elif temp >= 30:
        return "#f97316"  # Orange
    elif temp >= 22:
        return "#eab308"  # Yellow
    return "#3b82f6"      # Blue


def _fetch_city_weather(item):
    """
    Call OpenWeatherMap current weather endpoint by lat/lon with 6s timeout.
    Returns parsed weather dict or None on failure.
    """
    city_name, (lat, lon) = item
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(url, timeout=6)
        if resp.ok:
            data = resp.json()
            main = data.get("main", {})
            wind = data.get("wind", {})
            weather_arr = data.get("weather", [{}])
            desc = weather_arr[0].get("description", "") if weather_arr else ""
            return {
                "name": city_name,
                "lat": lat,
                "lon": lon,
                "temp": round(main.get("temp", 0)),
                "feels_like": round(main.get("feels_like", 0)),
                "humidity": main.get("humidity", 0),
                "wind": round(wind.get("speed", 0) * 3.6),
                "description": desc.capitalize() if desc else "Clear",
            }
        return None
    except Exception as e:
        return None


@require_GET
def weather_map(request):
    """
    GET /api/map/?city=<name>
    Builds an interactive Folium map of India with live weather bubbles on ~15 cities.
    Caches the combined city data for 10 minutes.
    If unauthenticated, returns a 401 HTML prompt with a login link.
    """
    if not request.user.is_authenticated:
        next_path = request.get_full_path()
        login_url = f"{settings.LOGIN_URL}?next={url_quote(next_path)}"
        locked_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            margin: 0; padding: 0;
            background: #0f172a; color: #f8fafc;
            font-family: 'Inter', sans-serif;
            display: flex; align-items: center; justify-content: center;
            height: 100vh; text-align: center;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px; padding: 40px 32px;
            max-width: 360px; backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }}
        .icon {{ font-size: 40px; margin-bottom: 16px; }}
        h2 {{ margin: 0 0 10px; font-size: 1.25rem; font-weight: 700; color: #fff; }}
        p {{ margin: 0 0 24px; font-size: 0.85rem; color: #94a3b8; line-height: 1.45; }}
        .btn {{
            display: inline-block; background: #3b82f6; color: #fff;
            padding: 10px 22px; border-radius: 8px; font-size: 0.88rem;
            font-weight: 600; text-decoration: none; transition: background 0.2s;
        }}
        .btn:hover {{ background: #2563eb; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="icon">🔒</div>
        <h2>Sign in to view the weather map</h2>
        <p>Explore live real-time weather bubbles across 15+ Indian cities on our interactive map.</p>
        <a href="{login_url}" target="_top" class="btn">Sign in to unlock</a>
    </div>
</body>
</html>"""
        return HttpResponse(locked_html, status=401, content_type="text/html")

    selected_city = request.GET.get("city", "").strip().lower()

    # 1. Fetch or retrieve from cache
    weather_list = cache.get("map_weather")
    if not weather_list:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_fetch_city_weather, MAP_CITIES.items()))
        weather_list = [r for r in results if r is not None]
        if weather_list:
            cache.set("map_weather", weather_list, 600)  # 10 min cache

    # If no data could be fetched, return friendly 502 HTML
    if not weather_list:
        return HttpResponse(
            """<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <title>Weather Map Unavailable</title>
                <style>
                    body {
                        background: #111827;
                        color: #f3f4f6;
                        font-family: 'Inter', system-ui, sans-serif;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                        text-align: center;
                    }
                    .box {
                        background: rgba(255,255,255,0.05);
                        padding: 32px 40px;
                        border-radius: 16px;
                        border: 1px solid rgba(255,255,255,0.1);
                    }
                    h2 { margin: 0 0 10px; color: #f97316; font-size: 1.3rem; }
                    p { margin: 0; color: #9ca3af; font-size: 0.9rem; }
                </style>
            </head>
            <body>
                <div class="box">
                    <h2>⚠️ Weather Map Unavailable</h2>
                    <p>Could not retrieve weather data for Indian cities. Please try again later.</p>
                </div>
            </body>
            </html>""",
            status=502,
            content_type="text/html"
        )

    # 2. Determine center and zoom (if city matches, center & zoom 8)
    center = [22.0, 78.9]
    zoom = 5
    selected_match = None
    if selected_city:
        for item in weather_list:
            if item["name"].lower() == selected_city:
                center = [item["lat"], item["lon"]]
                zoom = 8
                selected_match = item
                break

    # 3. Base Folium Map using ArcGIS World Topo Map
    india_map = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri &mdash; DeLorme, NAVTEQ, TomTom, Intermap, USGS, FAO, NPS, NRCAN, GeoBase, and the GIS User Community'
    )

    # 4. Add city markers with DivIcon, Tooltip, and Popup
    for item in weather_list:
        name = item["name"]
        lat = item["lat"]
        lon = item["lon"]
        temp = item["temp"]
        feels_like = item["feels_like"]
        humidity = item["humidity"]
        wind = item["wind"]
        desc = item["description"]

        color = _temp_color(temp)
        escaped_name = html.escape(name)
        escaped_desc = html.escape(desc)
        quoted_city = url_quote(name)

        is_selected = selected_match and selected_match["name"].lower() == name.lower()
        glow = "box-shadow: 0 0 0 4px #ffffff, 0 0 18px 6px rgba(249,115,22,0.9); transform: scale(1.18);" if is_selected else "box-shadow: 0 3px 10px rgba(0,0,0,0.45);"

        icon_html = f"""
        <div style="
            background-color: {color};
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border: 2px solid #ffffff;
            color: #ffffff;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 13px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            text-shadow: 0 1px 2px rgba(0,0,0,0.6);
            transition: transform 0.2s ease;
            {glow}
        ">
            {temp}°
        </div>
        """

        tooltip_html = f"""
        <div style="
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            padding: 10px 14px;
            min-width: 175px;
            background: #111827;
            color: #f3f4f6;
            border-radius: 10px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.6);
            border: 1px solid rgba(255,255,255,0.15);
        ">
            <div style="font-size: 15px; font-weight: 700; color: #ffffff; margin-bottom: 4px;">{escaped_name}</div>
            <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px;">
                <span style="font-size: 24px; font-weight: 800; color: {color};">{temp}°C</span>
                <span style="font-size: 12px; color: #9ca3af;">Feels like {feels_like}°C</span>
            </div>
            <div style="font-size: 13px; color: #e5e7eb; font-weight: 500; margin-bottom: 8px;">{escaped_desc}</div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #9ca3af; border-top: 1px solid rgba(255,255,255,0.12); padding-top: 6px;">
                <span>💧 Humidity: <b style="color:#ffffff;">{humidity}%</b></span>
                <span>💨 Wind: <b style="color:#ffffff;">{wind} km/h</b></span>
            </div>
        </div>
        """

        popup_html = f"""
        <div style="
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            padding: 8px 10px;
            text-align: center;
            min-width: 160px;
        ">
            <b style="font-size: 16px; color: #111827;">{escaped_name}</b>
            <div style="margin: 6px 0; font-size: 22px; font-weight: 800; color: {color};">{temp}°C</div>
            <div style="font-size: 12px; color: #6b7280; margin-bottom: 12px;">{escaped_desc}</div>
            <a href="/?city={quoted_city}" target="_top" style="
                display: inline-block;
                background: #f97316;
                color: #ffffff;
                font-weight: 600;
                font-size: 12px;
                text-decoration: none;
                padding: 7px 14px;
                border-radius: 6px;
                box-shadow: 0 2px 6px rgba(249,115,22,0.4);
            ">Open full forecast &rarr;</a>
        </div>
        """

        marker = folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(36, 36),
                icon_anchor=(18, 18),
            ),
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
            popup=folium.Popup(popup_html, max_width=260),
        )
        marker.add_to(india_map)

    rendered_html = india_map.get_root().render()
    return HttpResponse(rendered_html, content_type="text/html")


# ═══════════════════════════════════════════════════════════════════════════════
#  User Profile & Favorites API (authenticated JSON endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

@api_login_required
def profile_api(request):
    """
    GET: Return user profile {default_city, units}
    POST: Update user profile {default_city, units}
    Returns 401 if unauthenticated.
    """
    profile = getattr(request.user, 'profile', None)
    if profile is None:
        from accounts.models import UserProfile
        profile, _ = UserProfile.objects.get_or_create(user=request.user)

    if request.method == 'GET':
        return JsonResponse({
            'default_city': profile.default_city,
            'units': profile.units,
            'email': request.user.email,
            'name': request.user.get_full_name() or request.user.email,
        })

    elif request.method == 'POST':
        try:
            data = json.loads(request.body) if request.body else {}
        except Exception:
            data = request.POST

        if 'default_city' in data:
            profile.default_city = str(data['default_city']).strip()[:100]
        if 'units' in data and data['units'] in ('C', 'F'):
            profile.units = data['units']
        profile.save()

        return JsonResponse({
            'status': 'ok',
            'default_city': profile.default_city,
            'units': profile.units,
        })

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@api_login_required
def favorites_api(request):
    """
    GET: Return list of favorite cities [{city, country, created_at}]
    POST: Add a city to favorites {city, country} or bulk merge [{city, country}, ...]
    DELETE: Remove a city from favorites {city}
    Returns 401 if unauthenticated.
    """
    from accounts.models import FavoriteCity

    if request.method == 'GET':
        favs = FavoriteCity.objects.filter(user=request.user)
        return JsonResponse({
            'favorites': [
                {
                    'city': f.city,
                    'country': f.country,
                    'created_at': f.created_at.isoformat(),
                }
                for f in favs
            ]
        })

    elif request.method == 'POST':
        try:
            data = json.loads(request.body) if request.body else {}
        except Exception:
            data = request.POST

        # Bulk merge support: list of cities
        items = data.get('favorites') if isinstance(data.get('favorites'), list) else None
        if items is not None:
            added = 0
            for item in items:
                c = (item.get('city') or '').strip()
                cntry = (item.get('country') or '').strip()
                if c:
                    _, created = FavoriteCity.objects.get_or_create(
                        user=request.user,
                        city=c,
                        defaults={'country': cntry}
                    )
                    if created:
                        added += 1
            favs = FavoriteCity.objects.filter(user=request.user)
            return JsonResponse({
                'status': 'ok',
                'added': added,
                'favorites': [{'city': f.city, 'country': f.country} for f in favs]
            })

        city = (data.get('city') or '').strip()
        country = (data.get('country') or '').strip()
        if not city:
            return JsonResponse({'error': 'City is required'}, status=400)

        fav, created = FavoriteCity.objects.get_or_create(
            user=request.user,
            city=city,
            defaults={'country': country}
        )
        return JsonResponse({
            'status': 'ok',
            'created': created,
            'favorite': {'city': fav.city, 'country': fav.country}
        })

    elif request.method == 'DELETE':
        try:
            data = json.loads(request.body) if request.body else {}
        except Exception:
            data = request.GET

        city = (data.get('city') or '').strip()
        if not city:
            return JsonResponse({'error': 'City is required'}, status=400)

        deleted_count, _ = FavoriteCity.objects.filter(user=request.user, city__iexact=city).delete()
        return JsonResponse({'status': 'ok', 'deleted': deleted_count > 0})

    return JsonResponse({'error': 'Method not allowed'}, status=405)


