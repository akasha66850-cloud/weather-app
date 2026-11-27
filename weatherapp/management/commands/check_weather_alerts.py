import os
import re
import logging
import requests
import datetime
from collections import defaultdict

from django.core.management.base import BaseCommand
from django.utils import timezone
from django.core.cache import cache

from weatherapp.models import Subscriber
from weatherapp.sms import send_sms
from weatherapp.views import OPENWEATHER_API_KEY

logger = logging.getLogger(__name__)


def fetch_cached_owm_forecast(city):
    """
    Fetch the 5-day / 3-hour forecast for a city, reusing Django cache (10 min TTL)
    so that multiple subscribers in the same city trigger only one API call.
    Returns the parsed JSON dictionary or None on failure.
    """
    cache_key = "forecast_raw_" + re.sub(r"\W+", "_", city.lower())
    cached = cache.get(cache_key)
    if cached:
        return cached

    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.ok:
            data = resp.json()
            cache.set(cache_key, data, 600)  # 10 min cache
            return data
        else:
            logger.warning("OWM Forecast API returned status %s for %s", resp.status_code, city)
            return None
    except Exception as e:
        logger.error("Error fetching forecast for %s: %s", city, e)
        return None


def evaluate_severe_conditions(forecast_data):
    """
    Look at the next 6 hours (first 2 entries of the 3-hour forecast)
    and return an alert dictionary if any severe condition is met, else None.
    Thresholds:
      - Thunderstorm: OWM id 200..232 or 'thunder' in description
      - Rain probability: pop >= 70% or rain_3h >= 7 mm
      - Temperature: >= 40°C (extreme heat) or <= 5°C (extreme cold)
      - Wind speed: >= 50 km/h
      - Snow / Extreme atmospheric codes: 600..622, 781 (tornado), 771 (squall)
    """
    if not forecast_data:
        return None

    items = forecast_data.get("list", [])
    if not items:
        return None

    for idx, entry in enumerate(items[:2]):
        hours_ahead = (idx + 1) * 3
        weather_list = entry.get("weather", [{}])
        weather_id = weather_list[0].get("id", 800)
        desc = weather_list[0].get("description", "severe weather")
        pop = round(entry.get("pop", 0) * 100)
        rain_3h = round(entry.get("rain", {}).get("3h", 0.0), 1)
        temp = round(entry.get("main", {}).get("temp", 25.0), 1)
        wind_speed_ms = entry.get("wind", {}).get("speed", 0.0)
        wind_kph = round(wind_speed_ms * 3.6, 1)

        # 1. Thunderstorm
        if 200 <= weather_id <= 232 or "thunder" in desc.lower():
            return {
                "type": "thunderstorm",
                "headline": f"Thunderstorm ({desc}) likely in ~{hours_ahead} hours",
                "advice": "Stay indoors and avoid open spaces.",
            }

        # 2. Heavy Rain (pop >= 70% or rain_3h >= 7mm)
        if pop >= 70 or rain_3h >= 7.0:
            rain_detail = f"{pop}% chance" + (f", {rain_3h} mm" if rain_3h > 0 else "")
            return {
                "type": "heavy_rain",
                "headline": f"Heavy rain likely in ~{hours_ahead} hours ({rain_detail})",
                "advice": "Carry an umbrella and expect road waterlogging.",
            }

        # 3. Extreme Temperatures (>= 40°C or <= 5°C)
        if temp >= 40.0:
            return {
                "type": "extreme_heat",
                "headline": f"Severe heat wave (~{temp}°C) likely in ~{hours_ahead} hours",
                "advice": "Stay hydrated, use sunscreen, and avoid direct midday sun.",
            }
        if temp <= 5.0:
            return {
                "type": "extreme_cold",
                "headline": f"Severe cold (~{temp}°C) likely in ~{hours_ahead} hours",
                "advice": "Bundle up in warm layers before heading out.",
            }

        # 4. High Wind (>= 50 km/h)
        if wind_kph >= 50.0:
            return {
                "type": "high_wind",
                "headline": f"High winds (~{wind_kph} km/h) likely in ~{hours_ahead} hours",
                "advice": "Secure loose outdoor items and be cautious on roads.",
            }

        # 5. Snow or Extreme Atmosphere codes
        if 600 <= weather_id <= 622:
            return {
                "type": "snow",
                "headline": f"Snowfall ({desc}) expected in ~{hours_ahead} hours",
                "advice": "Watch for icy roads and drive carefully.",
            }
        if weather_id in (771, 781):
            return {
                "type": "squall_tornado",
                "headline": f"Extreme weather warning ({desc}) in ~{hours_ahead} hours",
                "advice": "Seek sturdy shelter immediately.",
            }

    return None


class Command(BaseCommand):
    help = (
        "Check upcoming weather for all verified subscribers with alerts_enabled. "
        "Sends an SMS warning if severe conditions (storm, heavy rain, extreme temps, winds) "
        "are forecast within the next 6 hours."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print simulated alert messages to stdout without sending actual SMS.",
        )

    def handle(self, *args, **options):
        dry_run = options.get("dry_run", False)
        mode_str = "[DRY-RUN] " if dry_run else ""
        self.stdout.write(self.style.NOTICE(f"{mode_str}Starting severe weather alert check..."))

        subscribers = Subscriber.objects.filter(verified=True, alerts_enabled=True)
        if not subscribers.exists():
            self.stdout.write(self.style.SUCCESS("No active verified subscribers with alerts_enabled found."))
            return

        # Group subscribers by city to minimize OWM API queries
        city_subscribers = defaultdict(list)
        for sub in subscribers:
            city_subscribers[sub.city.strip()].append(sub)

        alerts_sent_count = 0
        now = timezone.now()
        today_str = now.strftime("%Y-%m-%d")

        for city, subs in city_subscribers.items():
            forecast_data = fetch_cached_owm_forecast(city)
            if not forecast_data:
                self.stdout.write(self.style.WARNING(f"Could not retrieve forecast for '{city}'. Skipping."))
                continue

            alert = evaluate_severe_conditions(forecast_data)
            if not alert:
                continue

            # Alert key: type:YYYY-MM-DD
            alert_key = f"{alert['type']}:{today_str}"

            for sub in subs:
                # Duplicate protection: don't resend the same alert_key within 6 hours
                if sub.last_alert_key == alert_key and sub.last_alert_at:
                    elapsed_hours = (now - sub.last_alert_at).total_seconds() / 3600.0
                    if elapsed_hours < 6.0:
                        continue

                sms_body = (
                    f"[ALERT] Weather warning for {sub.city.capitalize()}: {alert['headline']}. "
                    f"{alert['advice']} Reply STOP to opt out."
                )

                if dry_run:
                    self.stdout.write(
                        self.style.WARNING(
                            f"[DRY-RUN ALERT] To: {sub.phone} ({sub.city})\n"
                            f"Content: {sms_body}\n"
                            f"Alert Key: {alert_key}\n"
                        )
                    )
                    alerts_sent_count += 1
                else:
                    success, res = send_sms(sub.phone, sms_body)
                    if success:
                        alerts_sent_count += 1
                        sub.last_alert_key = alert_key
                        sub.last_alert_at = now
                        sub.save(update_fields=["last_alert_key", "last_alert_at"])
                        self.stdout.write(self.style.SUCCESS(f"Alert sent to {sub.phone} for {sub.city}: {res}"))
                    else:
                        self.stdout.write(self.style.ERROR(f"Failed to send alert to {sub.phone}: {res}"))

        self.stdout.write(
            self.style.SUCCESS(f"{mode_str}Completed check_weather_alerts. Sent {alerts_sent_count} alert(s).")
        )
