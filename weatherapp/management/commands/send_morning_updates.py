import os
import pytz
import logging
import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone

from weatherapp.models import Subscriber
from weatherapp.sms import send_sms
from weatherapp.views import (
    OPENWEATHER_API_KEY,
    _current_weather_json,
    _run_ml_rain,
)
from weatherapp.management.commands.check_weather_alerts import fetch_cached_owm_forecast

logger = logging.getLogger(__name__)


def build_morning_message(city, weather, forecast_data, ml_result):
    """
    Build a concise, actionable daily morning weather SMS (< 320 chars).
    """
    temp = weather.get("temp", "--")
    feels_like = weather.get("feels_like", "--")
    desc = weather.get("description", "").capitalize()
    temp_min = weather.get("temp_min", temp)
    temp_max = weather.get("temp_max", temp)

    # Compute max rain probability from today's forecast entries
    max_pop = 0
    if forecast_data:
        entries = forecast_data.get("list", [])[:8]  # next 24 hours
        pops = [round(e.get("pop", 0) * 100) for e in entries]
        if pops:
            max_pop = max(pops)
            # Refine min/max with upcoming forecast if available
            temps = [e.get("main", {}).get("temp") for e in entries if "temp" in e.get("main", {})]
            if temps:
                temp_min = min(temp_min, round(min(temps)))
                temp_max = max(temp_max, round(max(temps)))

    # Practical daily recommendation tip (GSM-7 clean ASCII)
    rain_pred, rain_conf, ml_available = ml_result
    if rain_pred == "Yes" or max_pop >= 60:
        tip = "Tip: Carry an umbrella today."
    elif temp >= 35 or temp_max >= 38:
        tip = "Tip: High heat today! Stay hydrated."
    elif temp <= 14 or temp_min <= 12:
        tip = "Tip: Chilly morning! Wear warm clothing."
    elif max_pop >= 30:
        tip = "Tip: Light rain possible. Keep light jacket handy."
    else:
        tip = "Tip: Clear skies today! Have a great day."

    # ML rain prediction summary
    ml_part = ""
    if ml_available and rain_pred:
        ml_part = f" | ML Rain: {rain_pred}"
        if rain_conf:
            ml_part += f" ({rain_conf}%)"

    lines = [
        f"Weather for {city.capitalize()}: {temp}C ({desc}).",
        f"High: {temp_max}C, Low: {temp_min}C | Rain: {max_pop}%{ml_part}.",
        tip,
        "Reply STOP to opt out.",
    ]
    return "\n".join(lines)


class Command(BaseCommand):
    help = (
        "Send daily morning weather update SMS to verified subscribers "
        "whose scheduled morning_time falls in the current 15-minute window."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Simulate the morning dispatch and print SMS bodies to stdout without sending.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Bypass time window and today checks to dispatch to all verified subscribers.",
        )

    def handle(self, *args, **options):
        dry_run = options.get("dry_run", False)
        force = options.get("force", False)
        mode_str = "[DRY-RUN] " if dry_run else ""

        tz = pytz.timezone("Asia/Kolkata")
        now_local = datetime.datetime.now(tz)
        today_date = now_local.date()
        current_minute = now_local.hour * 60 + now_local.minute

        self.stdout.write(
            self.style.NOTICE(
                f"{mode_str}Checking morning updates at {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}..."
            )
        )

        subscribers = Subscriber.objects.filter(verified=True, morning_enabled=True)
        if not subscribers.exists():
            self.stdout.write(self.style.SUCCESS("No verified subscribers with morning_enabled found."))
            return

        sent_count = 0
        skipped_count = 0

        for sub in subscribers:
            if not force:
                # 1. Check if already dispatched today
                if sub.last_morning_sent == today_date:
                    skipped_count += 1
                    continue

                # 2. Check 15-minute delivery window
                sub_minute = sub.morning_time.hour * 60 + sub.morning_time.minute
                diff = current_minute - sub_minute
                if not (0 <= diff < 15):
                    skipped_count += 1
                    continue

            # Fetch current weather & forecast
            try:
                weather = _current_weather_json(sub.city, OPENWEATHER_API_KEY)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Could not fetch weather for {sub.city}: {e}"))
                continue

            ml_result = _run_ml_rain(weather)
            forecast_data = fetch_cached_owm_forecast(sub.city)
            sms_body = build_morning_message(sub.city, weather, forecast_data, ml_result)

            if dry_run:
                self.stdout.write(
                    self.style.WARNING(
                        f"[DRY-RUN MORNING SMS] To: {sub.phone} ({sub.city})\n"
                        f"Char count: {len(sms_body)}\n"
                        f"{sms_body}\n"
                    )
                )
                sent_count += 1
            else:
                success, res = send_sms(sub.phone, sms_body)
                if success:
                    sent_count += 1
                    sub.last_morning_sent = today_date
                    sub.save(update_fields=["last_morning_sent"])
                    self.stdout.write(self.style.SUCCESS(f"Morning update sent to {sub.phone}: {res}"))
                else:
                    self.stdout.write(self.style.ERROR(f"Failed to send to {sub.phone}: {res}"))

        self.stdout.write(
            self.style.SUCCESS(
                f"{mode_str}Completed send_morning_updates. Sent: {sent_count}, Skipped: {skipped_count}."
            )
        )
