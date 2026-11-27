"""
URL configuration for weatherproject project.
"""
from django.contrib import admin
from django.http import JsonResponse
from django.urls import path, include


def healthz(request):
    return JsonResponse({'status': 'ok'})


urlpatterns = [
    path('healthz/', healthz, name='healthz'),
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('', include('weatherapp.urls')),
]
