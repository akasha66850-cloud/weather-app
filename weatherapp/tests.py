"""
weatherapp/tests.py  –  Unit tests for the India weather ML pipeline
======================================================================
Run with: python manage.py test weatherapp
"""

import datetime
import math
import os
import json
import types
import unittest
from unittest.mock import patch, MagicMock

from django.test import TestCase, Client


class HealthCheckTests(TestCase):
    def test_healthz_returns_json_without_external_dependencies(self):
        response = self.client.get('/healthz/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})


# ── 1. Compass conversion ──────────────────────────────────────────────────────
class CompassDirectionTests(TestCase):
    """Test get_compass_direction for key angles and boundaries."""

    def setUp(self):
        from weatherapp.ml_model import get_compass_direction
        self.fn = get_compass_direction

    def test_north(self):
        self.assertEqual(self.fn(0),   "N")
        self.assertEqual(self.fn(360), "N")   # wraps to 0

    def test_east(self):
        self.assertEqual(self.fn(90),  "E")

    def test_south(self):
        self.assertEqual(self.fn(180), "S")

    def test_west(self):
        self.assertEqual(self.fn(270), "W")

    def test_northeast(self):
        self.assertEqual(self.fn(45),  "NE")

    def test_boundary_NNE(self):
        # 11.25 is the start of NNE
        self.assertEqual(self.fn(11.25), "NNE")

    def test_boundary_N_wraparound(self):
        # 326.25 is the start of NNW; 348.75 starts the final N sector
        self.assertEqual(self.fn(326.25), "NNW")   # start of NNW
        self.assertEqual(self.fn(348.75), "N")     # start of final N (348.75–360)
        self.assertEqual(self.fn(355),    "N")


# ── 2. Unknown wind direction fallback ────────────────────────────────────────
class WindFallbackTests(TestCase):
    """Unknown wind direction must map to 'N', not crash or silently use index 0."""

    def test_unknown_direction_uses_fallback(self):
        from sklearn.preprocessing import LabelEncoder
        from weatherapp.ml_model import KNOWN_COMPASS, WIND_FALLBACK, _encode_wind
        le = LabelEncoder()
        le.fit(KNOWN_COMPASS)
        # 'XYZ' is not a compass point
        result = _encode_wind("XYZ", le)
        expected = int(le.transform([WIND_FALLBACK])[0])
        self.assertEqual(result, expected, "Unknown direction should encode as WIND_FALLBACK='N'")

    def test_known_direction_encoded_correctly(self):
        from sklearn.preprocessing import LabelEncoder
        from weatherapp.ml_model import KNOWN_COMPASS, _encode_wind
        le = LabelEncoder()
        le.fit(KNOWN_COMPASS)
        result = _encode_wind("SW", le)
        expected = int(le.transform(["SW"])[0])
        self.assertEqual(result, expected)


# ── 3. Month / hour cyclical features ────────────────────────────────────────
class CyclicalFeaturesTests(TestCase):
    def test_month_sin_cos(self):
        from weatherapp.ml_model import _month_features
        s, c = _month_features(1)
        self.assertAlmostEqual(s, math.sin(2 * math.pi / 12), places=6)
        self.assertAlmostEqual(c, math.cos(2 * math.pi / 12), places=6)

    def test_hour_sin_cos(self):
        from weatherapp.ml_model import _hour_features
        s, c = _hour_features(0)
        self.assertAlmostEqual(s, 0.0, places=6)
        self.assertAlmostEqual(c, 1.0, places=6)


# ── 4. predict_rain returns valid output ──────────────────────────────────────
class PredictRainTests(TestCase):
    """Mock get_trained_models; verify _run_ml_rain returns correct types."""

    def _make_weather(self):
        return {
            'temp': 30.0, 'temp_min': 25.0, 'temp_max': 35.0,
            'humidity': 70, 'pressure': 1010,
            'wind_gust_speed': 5.0, 'wind_direction': 'SW',
            'lat': 13.08, 'lon': 80.27,
        }

    def _make_mock_models(self, rain_class='Yes', proba=0.75):
        import numpy as np
        from sklearn.preprocessing import LabelEncoder

        rain_model = MagicMock()
        rain_model.predict_proba.return_value = np.array([[1 - proba, proba]])

        wind_le = LabelEncoder()
        wind_le.fit(['E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'])
        rain_le = LabelEncoder()
        rain_le.fit(['No', 'Yes'])
        threshold = 0.5
        return (rain_model, None, None, wind_le, rain_le, threshold)

    @patch('weatherapp.views.get_trained_models')
    def test_returns_yes_no_string(self, mock_gtm):
        mock_gtm.return_value = self._make_mock_models(proba=0.80)
        from weatherapp.views import _run_ml_rain
        result, conf, available = _run_ml_rain(self._make_weather())
        self.assertIn(result, ('Yes', 'No'))
        self.assertTrue(available)

    @patch('weatherapp.views.get_trained_models')
    def test_confidence_in_range(self, mock_gtm):
        mock_gtm.return_value = self._make_mock_models(proba=0.80)
        from weatherapp.views import _run_ml_rain
        result, conf, available = _run_ml_rain(self._make_weather())
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 100.0)

    @patch('weatherapp.views.get_trained_models', side_effect=Exception("boom"))
    def test_exception_returns_none(self, mock_gtm):
        from weatherapp.views import _run_ml_rain
        result, conf, available = _run_ml_rain(self._make_weather())
        self.assertIsNone(result)
        self.assertIsNone(conf)
        self.assertFalse(available)


# ── 5. weather_api response keys ─────────────────────────────────────────────
class WeatherApiKeysTests(TestCase):
    """Mock OWM + ML; verify JSON response contains required keys."""

    OWM_RESPONSE = {
        'name': 'Chennai',
        'sys': {'country': 'IN'},
        'timezone': 19800,
        'main': {
            'temp': 32.0, 'feels_like': 35.0,
            'temp_min': 28.0, 'temp_max': 36.0,
            'humidity': 75, 'pressure': 1008,
        },
        'weather': [{'description': 'partly cloudy', 'icon': '02d'}],
        'wind': {'speed': 4.0, 'deg': 180},
        'visibility': 10000,
        'clouds': {'all': 40},
        'coord': {'lat': 13.08, 'lon': 80.27},
    }

    def _mock_owm(self, url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.ok = True
        resp.json.return_value = self.OWM_RESPONSE
        resp.raise_for_status = lambda: None
        return resp

    def _make_mock_models(self):
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        rain_model = MagicMock()
        rain_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        wind_le = LabelEncoder()
        wind_le.fit(['E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'])
        rain_le = LabelEncoder()
        rain_le.fit(['No', 'Yes'])
        return (rain_model, None, None, wind_le, rain_le, 0.5)

    @patch('weatherapp.views.get_trained_models')
    @patch('weatherapp.views.requests.get')
    def test_response_has_required_keys(self, mock_get, mock_gtm):
        mock_get.side_effect = self._mock_owm
        mock_gtm.return_value = self._make_mock_models()

        # Mock city image so it doesn't make a real HTTP call
        with patch('weatherapp.views.get_city_image', return_value={
            'image_url': None, 'author': None, 'license': None, 'source_url': None
        }):
            from django.contrib.auth.models import User
            user = User.objects.create_user(username='tester@example.com', email='tester@example.com', password='password123')
            client = Client()
            client.force_login(user)
            resp = client.get('/api/weather/?city=Chennai')

        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        for key in ('rain_result', 'rain_confidence', 'ml_available', 'authenticated'):
            self.assertIn(key, data, f"Missing key: {key}")
        self.assertTrue(data['authenticated'])

    @patch('weatherapp.views.requests.get')
    def test_missing_city_returns_400(self, mock_get):
        client = Client()
        resp = client.get('/api/weather/')
        self.assertEqual(resp.status_code, 400)


# ── 6. predict_future shape ────────────────────────────────────────────────────
class PredictFutureTests(TestCase):
    """predict_future must return two lists of exactly `steps` finite items."""

    @patch('weatherapp.ml_model.get_trained_models')
    def test_returns_correct_length_with_lag_model(self, mock_gtm):
        import math
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        temp_model = MagicMock()
        temp_model.predict.return_value = np.array([30.0])
        hum_model  = MagicMock()
        hum_model.predict.return_value = np.array([65.0])
        wind_le = LabelEncoder().fit(['N'])
        rain_le = LabelEncoder().fit(['No', 'Yes'])
        mock_gtm.return_value = (None, temp_model, hum_model, wind_le, rain_le, 0.5)

        from weatherapp.ml_model import predict_future
        start = datetime.datetime(2026, 6, 1, 0, 0, 0)
        temps, hums = predict_future(32.0, 70.0, 13.08, 80.27, start, steps=10)
        self.assertEqual(len(temps), 10)
        self.assertEqual(len(hums),  10)
        for t in temps:
            self.assertTrue(math.isfinite(t))
        for h in hums:
            self.assertTrue(math.isfinite(h))

    @patch('weatherapp.ml_model.get_trained_models')
    def test_fallback_returns_correct_length(self, mock_gtm):
        from sklearn.preprocessing import LabelEncoder
        wind_le = LabelEncoder().fit(['N'])
        rain_le = LabelEncoder().fit(['No', 'Yes'])
        mock_gtm.return_value = (None, None, None, wind_le, rain_le, 0.5)

        from weatherapp.ml_model import predict_future
        start = datetime.datetime(2026, 6, 1, 0, 0, 0)
        temps, hums = predict_future(32.0, 70.0, 13.08, 80.27, start, steps=5)
        self.assertEqual(len(temps), 5)
        self.assertEqual(len(hums),  5)
