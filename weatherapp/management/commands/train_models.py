"""
Management command: python manage.py train_models

Runs the full ML training pipeline:
  1. Trains rain classifier on weather_india.csv (time-based split)
  2. Trains temp & humidity regressors on data/hourly_india.csv
  3. Saves models to models/*.joblib
  4. Writes metrics to models/metrics.json
  5. Clears the in-memory model cache so the app immediately uses the new files
"""

import json
import os
import time

from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = "Train India weather ML models and save to models/*.joblib"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Retrain even if model files already exist",
        )

    def handle(self, *args, **options):
        from weatherapp.ml_model import run_training_pipeline, clear_model_cache

        md = os.path.join(settings.BASE_DIR, "models")
        rain_path = os.path.join(md, "rain_model.joblib")

        if os.path.exists(rain_path) and not options["force"]:
            self.stdout.write(
                self.style.WARNING(
                    "Model files already exist. Use --force to retrain."
                )
            )
            return

        self.stdout.write("Starting ML training pipeline…")
        t0 = time.time()

        try:
            run_training_pipeline(save=True)
        except FileNotFoundError as e:
            self.stderr.write(self.style.ERROR(str(e)))
            return
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Training failed: {e}"))
            raise

        # Clear cache so running server picks up new models immediately
        clear_model_cache()

        elapsed = time.time() - t0
        self.stdout.write(self.style.SUCCESS(f"Training complete in {elapsed:.1f}s"))

        # Print key metrics
        metrics_path = os.path.join(md, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            clf = m.get("rain_classifier", {})
            self.stdout.write("\n── Rain Classifier ──────────────────────────────")
            self.stdout.write(f"  Model     : {clf.get('model_type')}")
            self.stdout.write(f"  Accuracy  : {clf.get('accuracy'):.4f}")
            self.stdout.write(f"  Precision : {clf.get('precision_rain'):.4f}")
            self.stdout.write(f"  Recall    : {clf.get('recall_rain'):.4f}")
            self.stdout.write(f"  F1 (rain) : {clf.get('f1_rain'):.4f}")
            self.stdout.write(f"  ROC-AUC   : {clf.get('roc_auc'):.4f}")
            self.stdout.write(f"  Threshold : {clf.get('threshold'):.2f}")
            cm = clf.get("confusion_matrix", [])
            if cm:
                self.stdout.write(f"  Conf Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

            for key, label in [("temp_regressor", "Temp"), ("humidity_regressor", "Humidity")]:
                reg = m.get(key, {})
                if reg:
                    self.stdout.write(f"\n── {label} Regressor ──────────────────────────────")
                    self.stdout.write(f"  MAE           : {reg.get('mae'):.4f}")
                    self.stdout.write(f"  RMSE          : {reg.get('rmse'):.4f}")
                    self.stdout.write(f"  Baseline MAE  : {reg.get('baseline_mae'):.4f}")
                    self.stdout.write(f"  Beats baseline: {reg.get('beats_baseline')}")

            self.stdout.write(
                f"\n  Dataset rows : {m.get('dataset_rows')}  |  "
                f"Train ≤ {m.get('train_cutoff')}  |  "
                f"Test ≥ {m.get('test_start')}"
            )
            self.stdout.write("\nModel files saved to: " + md)
