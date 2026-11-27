"""
Twilio SMS & Verify service integration for WeatherApp.
Credentials are read from environment variables, .env file, or fallback defaults.
Supports Twilio Messaging Services for high-deliverability international routing.
"""
import os
import secrets
import logging
from pathlib import Path
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Attempt to load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

def get_twilio_credentials():
    """
    Retrieve Twilio credentials from environment variables or .env file.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.environ.get("TWILIO_FROM_NUMBER", "").strip()
    messaging_service_sid = os.environ.get("TWILIO_MESSAGING_SERVICE_SID", "").strip()
    verify_sid = os.environ.get("TWILIO_VERIFY_SID", "").strip()

    missing = []
    if not account_sid:
        missing.append("TWILIO_ACCOUNT_SID")
    if not auth_token:
        missing.append("TWILIO_AUTH_TOKEN")
    if not from_number and not messaging_service_sid:
        missing.append("TWILIO_FROM_NUMBER or TWILIO_MESSAGING_SERVICE_SID")

    if missing:
        raise ValueError(
            f"Missing required Twilio configuration: {', '.join(missing)}. "
            "Please configure them in your environment or .env file."
        )

    return account_sid, auth_token, from_number, messaging_service_sid, verify_sid


def get_twilio_client():
    """Return an authenticated Twilio REST Client instance."""
    account_sid, auth_token, _, _, _ = get_twilio_credentials()
    return Client(account_sid, auth_token)


def send_sms(to, body):
    """
    Send an outbound SMS message to an E.164 phone number.
    Uses MessagingServiceSid when available for optimal carrier delivery.
    Catches TwilioRestException and general exceptions, logs them, and does not crash.
    Returns (success: bool, message_sid_or_error: str).
    """
    try:
        account_sid, auth_token, from_number, messaging_service_sid, _ = get_twilio_credentials()
        client = Client(account_sid, auth_token)

        params = {"to": to, "body": body}
        if messaging_service_sid:
            params["messaging_service_sid"] = messaging_service_sid
        else:
            params["from_"] = from_number

        message = client.messages.create(**params)
        logger.info("Twilio SMS sent to %s: SID=%s", to, message.sid)
        return True, message.sid
    except TwilioRestException as e:
        error_msg = f"Twilio API Error [{e.code}]: {e.msg}"
        logger.error("Failed to send SMS to %s: %s", to, error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error sending SMS: {str(e)}"
        logger.error("Failed to send SMS to %s: %s", to, error_msg)
        return False, error_msg


def start_verification(phone):
    """
    Start phone number verification:
    - If TWILIO_VERIFY_SID is set, uses Twilio Verify Service.
    - If TWILIO_VERIFY_SID is not set, generates a secure 6-digit OTP, caches it in Django
      cache for 10 minutes, and dispatches it via Twilio SMS.
    Returns (success: bool, error_message: str).
    """
    try:
        account_sid, auth_token, _, _, verify_sid = get_twilio_credentials()

        # Mode A: Twilio Verify Service (if configured)
        if verify_sid:
            client = Client(account_sid, auth_token)
            verification = client.verify.v2.services(verify_sid).verifications.create(
                to=phone,
                channel="sms",
            )
            logger.info("Twilio Verify OTP initiated for %s: status=%s", phone, verification.status)
            return True, ""

        # Mode B: Direct SMS OTP via Twilio SMS + Django Cache
        otp_code = str(secrets.randbelow(900000) + 100000)
        cache_key = f"sms_otp_{phone}"
        cache.set(cache_key, otp_code, 600)  # 10 minutes expiry

        sms_body = (
            f"Your WeatherApp verification code is {otp_code} (valid for 10 minutes). "
            "Reply STOP to unsubscribe."
        )
        success, res = send_sms(phone, sms_body)
        if not success:
            return False, res

        logger.info("Direct SMS OTP dispatched to %s", phone)
        return True, ""
    except TwilioRestException as e:
        error_msg = f"Twilio Verify Error [{e.code}]: {e.msg}"
        logger.error("Failed to initiate verification for %s: %s", phone, error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = str(e)
        logger.error("Failed to initiate verification for %s: %s", phone, error_msg)
        return False, error_msg


def check_verification(phone, code):
    """
    Check the submitted OTP code:
    - If TWILIO_VERIFY_SID is set, verifies with Twilio Verify Service.
    - If TWILIO_VERIFY_SID is not set, verifies against Django cache.
    Returns (approved: bool, error_message: str).
    """
    try:
        account_sid, auth_token, _, _, verify_sid = get_twilio_credentials()

        # Mode A: Twilio Verify Service
        if verify_sid:
            client = Client(account_sid, auth_token)
            verification_check = client.verify.v2.services(verify_sid).verification_checks.create(
                to=phone,
                code=code.strip(),
            )
            logger.info(
                "Twilio Verify OTP check for %s: status=%s", phone, verification_check.status
            )
            if verification_check.status == "approved":
                return True, ""
            return False, "Invalid verification code. Please check and try again."

        # Mode B: Direct SMS OTP Cache Check
        cache_key = f"sms_otp_{phone}"
        stored_code = cache.get(cache_key)
        if stored_code and str(stored_code).strip() == code.strip():
            cache.delete(cache_key)  # One-time use
            return True, ""

        return False, "Invalid or expired verification code. Please request a new one."
    except TwilioRestException as e:
        error_msg = f"Twilio Verify Check Error [{e.code}]: {e.msg}"
        logger.error("Verification check failed for %s: %s", phone, error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = str(e)
        logger.error("Verification check failed for %s: %s", phone, error_msg)
        return False, error_msg
