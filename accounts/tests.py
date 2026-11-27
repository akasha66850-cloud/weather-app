from unittest.mock import patch
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.core.cache import cache
from accounts.models import UserProfile, FavoriteCity
from accounts.security import record_failed_login, is_locked_out, clear_failed_logins


class AccountsAuthTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = Client()

    def test_signup_creates_user_and_profile(self):
        resp = self.client.post('/accounts/signup/', {
            'full_name': 'Test User',
            'email': 'test@example.com',
            'password': 'SecurePassword123!',
            'confirm_password': 'SecurePassword123!',
        })
        self.assertEqual(resp.status_code, 302)
        user = User.objects.filter(email='test@example.com').first()
        self.assertIsNotNone(user)
        self.assertEqual(user.first_name, 'Test')
        self.assertEqual(user.last_name, 'User')
        self.assertTrue(hasattr(user, 'profile'))
        self.assertEqual(user.profile.units, 'C')

    def test_signup_duplicate_email_rejected(self):
        User.objects.create_user(username='dup@example.com', email='dup@example.com', password='password123')
        resp = self.client.post('/accounts/signup/', {
            'full_name': 'Another User',
            'email': 'dup@example.com',
            'password': 'SecurePassword123!',
            'confirm_password': 'SecurePassword123!',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'An account with this email already exists')

    def test_login_by_email_success(self):
        User.objects.create_user(username='user@example.com', email='user@example.com', password='password123')
        resp = self.client.post('/accounts/login/', {
            'email': 'user@example.com',
            'password': 'password123',
        })
        self.assertEqual(resp.status_code, 302)
        self.assertIn('_auth_user_id', self.client.session)

    def test_login_wrong_password_fails(self):
        User.objects.create_user(username='user2@example.com', email='user2@example.com', password='password123')
        resp = self.client.post('/accounts/login/', {
            'email': 'user2@example.com',
            'password': 'wrongpassword',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Incorrect email or password')

    def test_remember_me_sets_30_day_session(self):
        User.objects.create_user(username='rem@example.com', email='rem@example.com', password='password123')
        resp = self.client.post('/accounts/login/', {
            'email': 'rem@example.com',
            'password': 'password123',
            'remember_me': 'on',
        })
        self.assertEqual(resp.status_code, 302)
        self.assertGreater(self.client.session.get_expiry_age(), 60 * 60 * 24 * 25)

    def test_brute_force_lockout_after_5_attempts(self):
        ip = '127.0.0.1'
        email = 'brute@example.com'
        for _ in range(5):
            record_failed_login(ip, email)
        self.assertTrue(is_locked_out(ip, email))

        resp = self.client.post('/accounts/login/', {
            'email': email,
            'password': 'anypassword',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Too many failed login attempts')

    def test_open_redirect_blocked(self):
        User.objects.create_user(username='redir@example.com', email='redir@example.com', password='password123')
        resp = self.client.post('/accounts/login/?next=https://malicious-site.com', {
            'email': 'redir@example.com',
            'password': 'password123',
        })
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, '/')

    def test_favorites_api_requires_login(self):
        resp = self.client.get('/api/favorites/')
        self.assertEqual(resp.status_code, 401)
        resp2 = self.client.post('/api/favorites/', {'city': 'London'}, content_type='application/json')
        self.assertEqual(resp2.status_code, 401)

    def test_favorites_api_flow_for_logged_in_user(self):
        user = User.objects.create_user(username='fav@example.com', email='fav@example.com', password='password123')
        self.client.force_login(user)

        # 1. Add favorite
        resp = self.client.post('/api/favorites/', {'city': 'London', 'country': 'GB'}, content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(FavoriteCity.objects.filter(user=user).count(), 1)

        # 2. Get favorites
        resp = self.client.get('/api/favorites/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data['favorites']), 1)
        self.assertEqual(data['favorites'][0]['city'], 'London')

        # 3. Delete favorite
        resp = self.client.delete('/api/favorites/', {'city': 'London'}, content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(FavoriteCity.objects.filter(user=user).count(), 0)

    def test_favorites_are_per_user_isolated(self):
        user_a = User.objects.create_user(username='a@example.com', email='a@example.com', password='password123')
        user_b = User.objects.create_user(username='b@example.com', email='b@example.com', password='password123')

        FavoriteCity.objects.create(user=user_a, city='Tokyo')
        FavoriteCity.objects.create(user=user_b, city='Paris')

        self.client.force_login(user_a)
        resp = self.client.get('/api/favorites/')
        cities = [f['city'] for f in resp.json()['favorites']]
        self.assertIn('Tokyo', cities)
        self.assertNotIn('Paris', cities)

    def test_profile_api_get_and_update(self):
        user = User.objects.create_user(username='prof@example.com', email='prof@example.com', password='password123')
        self.client.force_login(user)

        # GET
        resp = self.client.get('/api/profile/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['units'], 'C')

        # POST update
        resp = self.client.post('/api/profile/', {'units': 'F', 'default_city': 'London'}, content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        user.profile.refresh_from_db()
        self.assertEqual(user.profile.units, 'F')
        self.assertEqual(user.profile.default_city, 'London')

    def test_anonymous_weather_api_returns_basic_no_ml_fields(self):
        with patch('weatherapp.views._current_weather_json') as mock_weather:
            mock_weather.return_value = {
                'city': 'London', 'country': 'GB', 'temp': 15, 'feels_like': 14,
                'temp_min': 12, 'temp_max': 18, 'humidity': 70, 'pressure': 1012,
                'wind_kph': 15, 'wind_deg': 200, 'wind_dir': 'SSW', 'visibility': 10,
                'clouds': 40, 'description': 'Scattered clouds',
                'current_datetime': '2026-11-25 20:00',
            }
            resp = self.client.get('/api/weather/?city=London')
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertFalse(data['authenticated'])
            self.assertFalse(data['ml_available'])
            self.assertNotIn('rain_result', data)
            self.assertNotIn('rain_confidence', data)
            self.assertEqual(data['temp'], 15)

    def test_authenticated_weather_api_includes_ml_fields(self):
        user = User.objects.create_user(username='auth@example.com', email='auth@example.com', password='password123')
        self.client.force_login(user)
        with patch('weatherapp.views._current_weather_json') as mock_weather, \
             patch('weatherapp.views._run_ml_rain', return_value=('Yes', 75.0, True)):
            mock_weather.return_value = {
                'city': 'London', 'country': 'GB', 'temp': 15, 'feels_like': 14,
                'temp_min': 12, 'temp_max': 18, 'humidity': 70, 'pressure': 1012,
                'wind_kph': 15, 'wind_deg': 200, 'wind_dir': 'SSW', 'visibility': 10,
                'clouds': 40, 'description': 'Scattered clouds',
                'current_datetime': '2026-11-25 20:00',
            }
            resp = self.client.get('/api/weather/?city=London')
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data['authenticated'])
            self.assertTrue(data['ml_available'])
            self.assertEqual(data['rain_result'], 'Yes')
            self.assertEqual(data['rain_confidence'], 75.0)

    def test_anonymous_endpoints_return_401_json(self):
        endpoints = [
            ('/api/forecast/?city=London', 'GET', {}),
            ('/api/news/?city=London', 'GET', {}),
            ('/api/reverse/?lat=51.5&lon=-0.1', 'GET', {}),
            ('/api/sms/subscribe/', 'POST', {'phone': '+919876543210', 'city': 'London'}),
            ('/api/sms/verify/', 'POST', {'phone': '+919876543210', 'code': '123456'}),
            ('/api/sms/unsubscribe/', 'POST', {'phone': '+919876543210'}),
            ('/api/profile/', 'GET', {}),
            ('/api/favorites/', 'GET', {}),
        ]
        for url, method, data in endpoints:
            if method == 'GET':
                resp = self.client.get(url)
            else:
                resp = self.client.post(url, data, content_type='application/json')
            self.assertEqual(resp.status_code, 401, f"{url} did not return 401")
            self.assertEqual(resp.json().get('error'), 'Login required')
            self.assertIn('/accounts/login/?next=', resp.json().get('login_url', ''))

    def test_anonymous_weather_map_returns_401_html(self):
        resp = self.client.get('/api/map/?city=London')
        self.assertEqual(resp.status_code, 401)
        self.assertIn('Sign in to view the weather map', resp.content.decode('utf-8'))
        self.assertIn('href="/accounts/login/?next=', resp.content.decode('utf-8'))

    def test_anonymous_rate_limiting_returns_429(self):
        # Limit is 60 in 10 minutes
        for i in range(60):
            self.client.get('/autocomplete/?q=Lon')
        resp = self.client.get('/autocomplete/?q=Lon')
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.json().get('error'), 'Rate limit exceeded. Please wait a few minutes.')

    def test_home_page_contains_app_context_and_locked_cards(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        html = resp.content.decode('utf-8')
        self.assertIn('"authenticated": false', html)
        self.assertIn('5-Day Forecast &amp; ML Prediction', html)
        self.assertIn('SMS Weather Alerts', html)
        self.assertIn('Weather News', html)

