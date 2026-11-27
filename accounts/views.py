"""
accounts/views.py – Login, Signup, Logout views with brute-force protection.
Password-reset is handled by Django's built-in views (configured in urls.py).
"""
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password, ValidationError
from django.http import JsonResponse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST
from django.conf import settings

from .security import is_locked_out, record_failed_login, clear_failed_logins, get_client_ip


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_next(request, fallback='/'):
    """Return `next` only when it is a safe same-host URL."""
    next_url = request.POST.get('next') or request.GET.get('next', '')
    if next_url and url_has_allowed_host_and_scheme(
        url=next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return next_url
    return fallback


# ── Login ──────────────────────────────────────────────────────────────────────

def login_view(request):
    if request.user.is_authenticated:
        return redirect(_safe_next(request))

    errors = {}
    email_value = ''

    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '')
        remember = request.POST.get('remember_me') == 'on'
        email_value = email

        ip = get_client_ip(request)

        # Brute-force check
        if is_locked_out(ip, email):
            errors['non_field'] = (
                'Too many failed login attempts. Please wait 15 minutes and try again.'
            )
        elif not email:
            errors['email'] = 'Email address is required.'
        elif not password:
            errors['password'] = 'Password is required.'
        else:
            user = authenticate(request, email=email, password=password)
            if user is None:
                record_failed_login(ip, email)
                errors['non_field'] = 'Incorrect email or password.'
            else:
                clear_failed_logins(ip, email)
                login(request, user, backend='accounts.backends.EmailBackend')
                if remember:
                    request.session.set_expiry(60 * 60 * 24 * 30)
                else:
                    request.session.set_expiry(0)
                return redirect(_safe_next(request))

    google_enabled = bool(getattr(settings, 'GOOGLE_CLIENT_ID', ''))

    return render(request, 'registration/login.html', {
        'errors': errors,
        'email_value': email_value,
        'next': request.GET.get('next', ''),
        'google_enabled': google_enabled,
    })


# ── Signup ─────────────────────────────────────────────────────────────────────

def signup_view(request):
    if request.user.is_authenticated:
        return redirect(_safe_next(request))

    errors = {}
    form_data = {}

    if request.method == 'POST':
        full_name = request.POST.get('full_name', '').strip()
        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '')
        confirm = request.POST.get('confirm_password', '')
        form_data = {'full_name': full_name, 'email': email}

        # Validate fields
        if not full_name:
            errors['full_name'] = 'Full name is required.'
        if not email:
            errors['email'] = 'Email address is required.'
        elif User.objects.filter(email__iexact=email).exists():
            errors['email'] = 'An account with this email already exists.'
        if not password:
            errors['password'] = 'Password is required.'
        elif password != confirm:
            errors['confirm_password'] = 'Passwords do not match.'
        else:
            try:
                validate_password(password)
            except ValidationError as e:
                errors['password'] = ' '.join(e.messages)

        if not errors:
            # Create user: username = email (truncated to 150 chars to be safe)
            username = email[:150]
            # Make username unique if collision
            if User.objects.filter(username=username).exists():
                username = email[:140] + str(User.objects.count())[:10]
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
            )
            # Set full name
            parts = full_name.split(' ', 1)
            user.first_name = parts[0]
            user.last_name = parts[1] if len(parts) > 1 else ''
            user.save()

            login(request, user, backend='accounts.backends.EmailBackend')
            return redirect(_safe_next(request))

    google_enabled = bool(getattr(settings, 'GOOGLE_CLIENT_ID', ''))

    return render(request, 'registration/signup.html', {
        'errors': errors,
        'form_data': form_data,
        'next': request.GET.get('next', '') or request.POST.get('next', ''),
        'google_enabled': google_enabled,
    })


# ── Logout ─────────────────────────────────────────────────────────────────────

@require_POST
def logout_view(request):
    logout(request)
    return redirect(settings.LOGOUT_REDIRECT_URL)
