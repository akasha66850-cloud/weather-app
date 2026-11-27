from django.core.cache import cache

LOCKOUT_THRESHOLD = 5
LOCKOUT_TIMEOUT_SECONDS = 15 * 60  # 15 minutes


def _get_cache_key(ip: str, email: str) -> str:
    norm_email = (email or '').strip().lower()
    norm_ip = (ip or '').strip()
    return f"login_attempts_{norm_ip}_{norm_email}"


def get_client_ip(request) -> str:
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR', '127.0.0.1')
    return ip


def is_locked_out(ip: str, email: str) -> bool:
    key = _get_cache_key(ip, email)
    attempts = cache.get(key, 0)
    return attempts >= LOCKOUT_THRESHOLD


def record_failed_login(ip: str, email: str) -> int:
    key = _get_cache_key(ip, email)
    attempts = cache.get(key, 0) + 1
    cache.set(key, attempts, LOCKOUT_TIMEOUT_SECONDS)
    return attempts


def clear_failed_logins(ip: str, email: str):
    key = _get_cache_key(ip, email)
    cache.delete(key)


# ── Rate Limiting for Public Endpoints ─────────────────────────────────────────

def check_rate_limit(request, action: str, limit: int, timeout_seconds: int = 600) -> bool:
    """
    Returns True if request is allowed, False if limit exceeded.
    Default timeout: 600 seconds (10 minutes).
    """
    ip = get_client_ip(request)
    user_id = request.user.pk if request.user.is_authenticated else 'anon'
    cache_key = f"rl_{action}_{user_id}_{ip}"
    current_count = cache.get(cache_key, 0)
    if current_count >= limit:
        return False
    cache.set(cache_key, current_count + 1, timeout_seconds)
    return True


# ── JSON 401 API Login Required Decorator ─────────────────────────────────────

from functools import wraps
from urllib.parse import quote as url_quote
from django.http import JsonResponse
from django.conf import settings

def api_login_required(view_func):
    """
    Decorator for JSON API endpoints ensuring the user is logged in.
    Returns HTTP 401 JSON: {"error": "Login required", "login_url": "/accounts/login/?next=..."}
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            next_path = request.get_full_path()
            login_url = f"{settings.LOGIN_URL}?next={url_quote(next_path)}"
            return JsonResponse({
                'error': 'Login required',
                'login_url': login_url
            }, status=401)
        return view_func(request, *args, **kwargs)
    return _wrapped_view

