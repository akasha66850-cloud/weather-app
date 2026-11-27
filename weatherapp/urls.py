from django.urls import path
from . import views

urlpatterns = [
    path('',                     views.home,             name='home'),
    path('autocomplete/',        views.city_autocomplete, name='city_autocomplete'),
    path('api/weather/',         views.weather_api,       name='weather_api'),
    path('api/forecast/',        views.forecast_api,      name='forecast_api'),
    path('api/reverse/',         views.reverse_geocode,   name='reverse_geocode'),
    path('api/news/',            views.weather_news,      name='weather_news'),
    path('api/sms/subscribe/',   views.sms_subscribe,     name='sms_subscribe'),
    path('api/sms/verify/',      views.sms_verify,        name='sms_verify'),
    path('api/sms/unsubscribe/', views.sms_unsubscribe,   name='sms_unsubscribe'),
    path('api/map/',             views.weather_map,       name='weather_map'),
    path('api/profile/',         views.profile_api,       name='profile_api'),
    path('api/favorites/',       views.favorites_api,     name='favorites_api'),
]
