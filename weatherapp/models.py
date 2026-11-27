import datetime
from django.db import models


class Subscriber(models.Model):
    """
    Subscribers registered for daily morning weather updates and severe weather alerts.
    """
    phone = models.CharField(
        max_length=20,
        unique=True,
        db_index=True,
        help_text="Subscriber phone number in E.164 international format (e.g. +919876543210)",
    )
    city = models.CharField(
        max_length=100,
        help_text="City for weather reports and alerts",
    )
    morning_time = models.TimeField(
        default=datetime.time(7, 0),
        help_text="Preferred time for daily morning weather SMS (local time)",
    )
    morning_enabled = models.BooleanField(
        default=True,
        help_text="Whether daily morning update SMS is enabled",
    )
    alerts_enabled = models.BooleanField(
        default=True,
        help_text="Whether severe weather warning alerts are enabled",
    )
    verified = models.BooleanField(
        default=False,
        help_text="Whether the phone number has been verified via Twilio OTP",
    )
    last_alert_key = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="Key representing the last alert type & date to prevent duplicate alerts within 6 hours",
    )
    last_alert_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when the last weather alert was dispatched",
    )
    last_morning_sent = models.DateField(
        null=True,
        blank=True,
        help_text="Date when the last morning update was sent to avoid duplicate dispatches",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the subscription record was created",
    )

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Weather SMS Subscriber"
        verbose_name_plural = "Weather SMS Subscribers"

    def __str__(self):
        status = "Verified" if self.verified else "Pending"
        return f"{self.phone} ({self.city}) [{status}]"
