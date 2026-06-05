"""
Feature Store — Tum modeller icin merkezi veri erisim noktasi.

Bu modul, LSTM/GRU, XGBoost/RF, Ensemble ve RL modelleri icin
tek bir veri kaynagi saglar. Boylece tum modeller AYNI
train/val/test bolumlendirmesini kullanir.

Kullanim:
    store = FeatureStore("BTC-USD")
    X_train, y_train = store.get_lstm_split("train")
    X_test, y_test = store.get_xgb_split("test")
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from src.data.ml_config import DATA_ML_DIR, FEATURE_COLUMNS, WINDOW_SIZE


@dataclass
class FeatureStore:
    """Tum modeller icin tek veri erisim noktasi."""

    ticker: str
    _manifest: dict = field(default_factory=dict, repr=False)
    _X_lstm: np.ndarray | None = field(default=None, repr=False)
    _y: np.ndarray | None = field(default=None, repr=False)
    _X_xgb: np.ndarray | None = field(default=None, repr=False)
    _close: np.ndarray | None = field(default=None, repr=False)
    _scaler: object | None = field(default=None, repr=False)
    _features_df: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self):
        self._load()

    # ── Veri Yukleme ──────────────────────────────────────────

    def _ticker_dir(self) -> str:
        return os.path.join(DATA_ML_DIR, self.ticker)

    def _load(self):
        d = self._ticker_dir()
        manifest_path = os.path.join(d, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"FeatureStore: {self.ticker} icin manifest bulunamadi.\n"
                f"Once calistirin: python -m src.data.build_dataset --ticker {self.ticker}"
            )
        with open(manifest_path, encoding="utf-8") as f:
            self._manifest = json.load(f)

        self._X_lstm = np.load(os.path.join(d, "X_lstm.npy"))
        self._X_xgb = np.load(os.path.join(d, "X_xgb.npy"))
        self._close = np.load(os.path.join(d, "close_aligned.npy"))
        self._scaler = joblib.load(os.path.join(d, "feature_scaler.pkl"))

        # Unified target (v3) veya legacy (v2)
        y_path = os.path.join(d, "y.npy")
        if os.path.exists(y_path):
            self._y = np.load(y_path)
        else:
            # Legacy uyumluluk
            self._y = np.load(os.path.join(d, "y_lstm.npy"))

        # features CSV (opsiyonel — backtest / grafik icin)
        csv_path = os.path.join(d, "features.csv")
        if os.path.exists(csv_path):
            self._features_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # ── Split Bilgisi ─────────────────────────────────────────

    @property
    def split(self) -> dict:
        """{'train': [s,e], 'val': [s,e], 'test': [s,e]}"""
        si = self._manifest.get("split_indices")
        if si is None:
            raise ValueError(
                f"{self.ticker}: split_indices manifest'te yok. "
                f"Dataset'i yeniden olusturun (v3)."
            )
        return si

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def feature_columns(self) -> list[str]:
        return self._manifest.get("feature_columns", FEATURE_COLUMNS)

    @property
    def window_size(self) -> int:
        return self._manifest.get("window_size", WINDOW_SIZE)

    @property
    def n_samples(self) -> int:
        return len(self._y)

    @property
    def scaler(self):
        return self._scaler

    @property
    def features_df(self) -> pd.DataFrame | None:
        return self._features_df

    # ── LSTM/GRU Verisi ───────────────────────────────────────

    def get_lstm_split(self, split_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        LSTM/GRU icin (X_windows, y) dondurur.
        X shape: (n, window_size, n_features) — olceklenmis
        y shape: (n,)
        """
        s, e = self.split[split_name]
        return self._X_lstm[s:e].astype(np.float32), self._y[s:e].astype(np.float32)

    # ── XGBoost/RF Verisi ─────────────────────────────────────

    def get_xgb_split(self, split_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        XGBoost/RF icin (X_tabular, y) dondurur.
        X shape: (n, n_features) — olceklenmemis (ham)
        y shape: (n,)
        """
        s, e = self.split[split_name]
        return self._X_xgb[s:e].astype(np.float32), self._y[s:e].astype(np.int32)

    # ── Close Fiyatlari ───────────────────────────────────────

    def get_close_split(self, split_name: str) -> np.ndarray:
        """Backtest icin close fiyatlari."""
        s, e = self.split[split_name]
        return self._close[s:e]

    # ── Tum Veri ──────────────────────────────────────────────

    @property
    def X_lstm(self) -> np.ndarray:
        return self._X_lstm

    @property
    def X_xgb(self) -> np.ndarray:
        return self._X_xgb

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def close(self) -> np.ndarray:
        return self._close

    # ── Sinif Agirliklari ─────────────────────────────────────

    def get_class_weights(self, split_name: str = "train") -> dict:
        """Egitim setindeki sinif agirliklari (dengesizlik duzeltmesi icin)."""
        _, y_split = self.get_lstm_split(split_name)
        n_up = int(np.sum(y_split == 1))
        n_down = int(np.sum(y_split == 0))
        total = n_up + n_down
        return {
            0: total / (2 * n_down + 1e-8),
            1: total / (2 * n_up + 1e-8),
        }

    # ── Ozet ──────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"FeatureStore({self.ticker})",
            f"  Version   : {self._manifest.get('version', '?')}",
            f"  Samples   : {self.n_samples}",
            f"  Features  : {len(self.feature_columns)}",
            f"  Window    : {self.window_size}",
            f"  Threshold : {self._manifest.get('direction_threshold', 'N/A')}",
        ]
        for name, (s, e) in self.split.items():
            lines.append(f"  {name:5s}     : [{s}:{e}] ({e-s} samples)")
        return "\n".join(lines)
