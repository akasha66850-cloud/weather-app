"""
accounts/urls.py – Authentication routes.
"""
from django.urls import path, include
from django.contrib.auth import views as auth_views

from . import views

# Custom templates for password-reset (same card style)
_pw_reset_kwargs = dict(
    template_name='registration/password_reset_form.html',
    email_template_name='registration/password_reset_email.txt',
    subject_template_name='registration/password_reset_subject.txt',
)
_pw_done_kwargs = dict(template_name='registration/password_reset_done.html')
_pw_confirm_kwargs = dict(template_name='registration/password_reset_confirm.html')
_pw_complete_kwargs = dict(template_name='registration/password_reset_complete.html')

urlpatterns = [
    path('login/',  views.login_view,  name='account_login'),
    path('signup/', views.signup_view, name='account_signup'),
    path('logout/', views.logout_view, name='account_logout'),

    # Password reset (Django built-in views, custom templates)
    path('password-reset/',
         auth_views.PasswordResetView.as_view(**_pw_reset_kwargs),
         name='password_reset'),
    path('password-reset/done/',
         auth_views.PasswordResetDoneView.as_view(**_pw_done_kwargs),
         name='password_reset_done'),
    path('password-reset/confirm/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(**_pw_confirm_kwargs),
         name='password_reset_confirm'),
    path('password-reset/complete/',
         auth_views.PasswordResetCompleteView.as_view(**_pw_complete_kwargs),
         name='password_reset_complete'),

    # django-allauth (Google OAuth etc.)
    path('', include('allauth.urls')),
]
