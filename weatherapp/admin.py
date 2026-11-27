from django.contrib import admin
from .models import Subscriber


@admin.register(Subscriber)
class SubscriberAdmin(admin.ModelAdmin):
    list_display = (
        "phone",
        "city",
        "morning_time",
        "morning_enabled",
        "alerts_enabled",
        "verified",
        "last_morning_sent",
        "last_alert_at",
        "created_at",
    )
    list_filter = ("verified", "morning_enabled", "alerts_enabled", "city")
    search_fields = ("phone", "city")
    readonly_fields = ("created_at", "last_alert_at", "last_morning_sent", "last_alert_key")
