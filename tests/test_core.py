"""
test_core.py — Temel Dogrulama Testleri
=========================================
Projenin yeniden yapilandirma sonrasi saglamligini dogrular.

Calistirma:
    python -m pytest tests/test_core.py -v
    python tests/test_core.py   (pytest olmadan)
"""
import os
import sys
import json
import unittest

# Proje kokunu sys.path'e ekle
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.ml_config import TICKERS, FEATURE_COLUMNS, WINDOW_SIZE, DATA_ML_DIR


class TestImports(unittest.TestCase):
    """Tum ana modullerin import edilebilirligini test eder."""

    def test_import_feature_store(self):
        from src.data.feature_store import FeatureStore
        self.assertIsNotNone(FeatureStore)

    def test_import_weighted_hybrid(self):
        from src.models.weighted_hybrid import (
            compute_weighted_hybrid, generate_signals,
            select_weights, is_degenerate
        )
        self.assertIsNotNone(compute_weighted_hybrid)

    def test_import_meta_inference(self):
        from src.models.meta_inference import compute_final_probs
        self.assertIsNotNone(compute_final_probs)

    def test_import_evaluation(self):
        from src.evaluation.metrics import compute_classification_metrics
        from src.evaluation.weight_selector import evaluate_weight_configs
        self.assertIsNotNone(compute_classification_metrics)
        self.assertIsNotNone(evaluate_weight_configs)

    def test_import_pipeline(self):
        from src.pipeline import run_all, run_eval, run_backtest
        self.assertIsNotNone(run_all)

    def test_import_backtester(self):
        from src.utils.backtester import SimpleBacktester
        self.assertIsNotNone(SimpleBacktester)


class TestManifest(unittest.TestCase):
    """Manifest dosyalarinin okunabilirligini test eder."""

    def test_manifest_exists_for_each_ticker(self):
        for ticker in TICKERS:
            manifest_path = os.path.join(DATA_ML_DIR, ticker, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                self.assertIn("ticker", manifest)
                self.assertIn("feature_columns", manifest)
                self.assertIn("window_size", manifest)

    def test_manifest_has_split_indices(self):
        """v3 manifest'te split_indices olmali."""
        for ticker in TICKERS:
            manifest_path = os.path.join(DATA_ML_DIR, ticker, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                if manifest.get("version", 0) >= 3:
                    self.assertIn("split_indices", manifest)
                    si = manifest["split_indices"]
                    self.assertIn("train", si)
                    self.assertIn("val", si)
                    self.assertIn("test", si)


class TestFeatureStore(unittest.TestCase):
    """FeatureStore'un dogru calistigini test eder."""

    @classmethod
    def setUpClass(cls):
        from src.data.feature_store import FeatureStore
        ticker = "BTC-USD"
        manifest_path = os.path.join(DATA_ML_DIR, ticker, "manifest.json")
        if not os.path.exists(manifest_path):
            raise unittest.SkipTest(f"{ticker} dataset bulunamadi")
        cls.store = FeatureStore(ticker)

    def test_split_no_overlap(self):
        """Train/val/test indeksleri ortusmuyor mu?"""
        split = self.store.split
        train_end = split["train"][1]
        val_start = split["val"][0]
        val_end = split["val"][1]
        test_start = split["test"][0]

        self.assertLessEqual(train_end, val_start,
                             "Train ve val ortusuyur!")
        self.assertLessEqual(val_end, test_start,
                             "Val ve test ortusuyur!")

    def test_purge_gap(self):
        """Train sonu ile val basi arasinda yeterli bosluk var mi?"""
        split = self.store.split
        gap = split["val"][0] - split["train"][1]
        self.assertGreaterEqual(gap, 1,
                                f"Purge gap yetersiz: {gap}")

    def test_lstm_xgb_same_y(self):
        """LSTM ve XGB ayni y donduruyur mu?"""
        import numpy as np
        _, y_lstm = self.store.get_lstm_split("train")
        _, y_xgb = self.store.get_xgb_split("train")
        np.testing.assert_array_equal(
            y_lstm.astype(int), y_xgb.astype(int),
            "LSTM ve XGB farkli y donduruyor!"
        )

    def test_no_negative_labels(self):
        """y dizisinde -1 (dead-zone) deger yok mu?"""
        import numpy as np
        y = self.store.y
        self.assertFalse(np.any(y == -1),
                         "y dizisinde dead-zone (-1) deger var!")

    def test_feature_count(self):
        """Feature sayisi ml_config ile uyumlu mu?"""
        X_xgb, _ = self.store.get_xgb_split("train")
        self.assertEqual(X_xgb.shape[1], len(FEATURE_COLUMNS),
                         f"Feature sayisi uyumsuz: {X_xgb.shape[1]} vs {len(FEATURE_COLUMNS)}")

    def test_window_size(self):
        """LSTM window boyutu dogru mu?"""
        X_lstm, _ = self.store.get_lstm_split("train")
        self.assertEqual(X_lstm.shape[1], WINDOW_SIZE,
                         f"Window size uyumsuz: {X_lstm.shape[1]} vs {WINDOW_SIZE}")


class TestWeightedHybrid(unittest.TestCase):
    """Weighted hybrid fonksiyonlarini test eder."""

    def test_compute_weighted_hybrid(self):
        import numpy as np
        from src.models.weighted_hybrid import compute_weighted_hybrid
        rf = np.array([0.6, 0.4, 0.7])
        xgb = np.array([0.5, 0.5, 0.8])
        lstm = np.array([0.7, 0.3, 0.6])
        result = compute_weighted_hybrid(rf, xgb, lstm)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(0 <= p <= 1 for p in result))

    def test_final_prob_range(self):
        """final_prob 0-1 arasinda mi?"""
        import numpy as np
        from src.models.weighted_hybrid import compute_weighted_hybrid
        rf = np.random.rand(100)
        xgb = np.random.rand(100)
        lstm = np.random.rand(100)
        result = compute_weighted_hybrid(rf, xgb, lstm)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_generate_signals(self):
        import numpy as np
        from src.models.weighted_hybrid import generate_signals
        probs = np.array([0.6, 0.5, 0.3, 0.8, 0.45])
        signals = generate_signals(probs)
        self.assertEqual(signals[0], 2)   # > 0.55 → AL
        self.assertEqual(signals[1], 1)   # 0.45-0.55 → TUT
        self.assertEqual(signals[2], 0)   # < 0.45 → KAPAT
        self.assertEqual(signals[3], 2)   # > 0.55 → AL

    def test_select_weights(self):
        from src.models.weighted_hybrid import select_weights
        w = select_weights(0.55)
        self.assertEqual(w, (0.40, 0.40, 0.20))
        w = select_weights(0.30)
        self.assertEqual(w, (0.45, 0.45, 0.10))
        w = select_weights(0.0)
        self.assertEqual(w, (0.50, 0.50, 0.00))

    def test_is_degenerate(self):
        import numpy as np
        from src.models.weighted_hybrid import is_degenerate
        # Hep ayni sinif
        degen = np.ones(100) * 0.9
        self.assertTrue(is_degenerate(degen))
        # Dengeli
        balanced = np.random.rand(100)
        self.assertFalse(is_degenerate(balanced))


class TestSavedModels(unittest.TestCase):
    """Kayitli modellerin yuklenebilirligini test eder."""

    def test_rf_model_loadable(self):
        import joblib
        models_dir = os.path.join(_project_root, "src", "models", "saved_models")
        for ticker in TICKERS:
            rf_path = os.path.join(models_dir, f"{ticker}_rf_classifier.pkl")
            if os.path.exists(rf_path):
                model = joblib.load(rf_path)
                self.assertTrue(hasattr(model, "predict_proba"))

    def test_xgb_model_loadable(self):
        import joblib
        models_dir = os.path.join(_project_root, "src", "models", "saved_models")
        for ticker in TICKERS:
            xgb_path = os.path.join(models_dir, f"{ticker}_xgb_classifier.pkl")
            if os.path.exists(xgb_path):
                model = joblib.load(xgb_path)
                self.assertTrue(hasattr(model, "predict_proba"))

    def test_lstm_model_loadable(self):
        import torch
        from src.models.time_series_models import TimeSeriesNet
        models_dir = os.path.join(_project_root, "src", "models", "saved_models")
        for ticker in TICKERS:
            lstm_path = os.path.join(models_dir, f"{ticker}_lstm_best.pth")
            if os.path.exists(lstm_path):
                sd = torch.load(lstm_path, map_location="cpu", weights_only=True)
                num_layers = max(len([k for k in sd.keys()
                                      if k.startswith('rnn.weight_ih')]), 1)
                # hidden_size'i state_dict'ten otomatik tespit et
                hidden_size = sd['rnn.weight_ih_l0'].shape[0] // 4
                model = TimeSeriesNet(
                    len(FEATURE_COLUMNS), hidden_size=hidden_size,
                    num_layers=num_layers, output_size=1,
                    model_type="LSTM", use_attention=True
                )
                model.load_state_dict(sd)
                self.assertIsNotNone(model)


class TestMetrics(unittest.TestCase):
    """Evaluation metrik fonksiyonlarini test eder."""

    def test_classification_metrics(self):
        import numpy as np
        from src.evaluation.metrics import compute_classification_metrics
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_compare_to_dummy(self):
        import numpy as np
        from src.evaluation.metrics import compare_to_dummy
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        result = compare_to_dummy(y_true, y_pred)
        self.assertIn("model_accuracy", result)
        self.assertIn("dummy_majority_accuracy", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
