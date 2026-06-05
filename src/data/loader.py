"""
Egitim kodlari icin tek veri yukleme arayuzu.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src.data.ml_config import (
    DATA_ML_DIR,
    DATA_PROCESSED_DIR,
    FEATURE_COLUMNS,
    WINDOW_SIZE,
)


@dataclass
class MLDataset:
    ticker: str
    feature_columns: list[str]
    window_size: int

    # LSTM (olceklenmis pencereler)
    X_lstm: np.ndarray
    y_lstm: np.ndarray

    # XGBoost / RF (olceklenmemis tek satir)
    X_xgb: np.ndarray
    y_xgb: np.ndarray

    # Grafik / backtest
    features_df: pd.DataFrame
    close_aligned: np.ndarray
    scaler: object | None = None


def _read_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_dataset(ticker: str = "BTC-USD") -> MLDataset:
    """
    Oncelik: data/ml/{ticker}/  yoksa data/processed/ legacy.
    """
    ml_dir = os.path.join(DATA_ML_DIR, ticker)
    if os.path.exists(os.path.join(ml_dir, "X_lstm.npy")):
        manifest = _read_json(os.path.join(ml_dir, "manifest.json")) or {}
        fc = manifest.get("feature_columns", FEATURE_COLUMNS)
        ws = manifest.get("window_size", WINDOW_SIZE)
        return MLDataset(
            ticker=ticker,
            feature_columns=fc,
            window_size=ws,
            X_lstm=np.load(os.path.join(ml_dir, "X_lstm.npy")),
            y_lstm=np.load(os.path.join(ml_dir, "y_lstm.npy")),
            X_xgb=np.load(os.path.join(ml_dir, "X_xgb.npy")),
            y_xgb=np.load(os.path.join(ml_dir, "y_xgb.npy")),
            features_df=pd.read_csv(os.path.join(ml_dir, "features.csv"), index_col=0, parse_dates=True),
            close_aligned=np.load(os.path.join(ml_dir, "close_aligned.npy")),
            scaler=joblib.load(os.path.join(ml_dir, "feature_scaler.pkl")),
        )

    proc = DATA_PROCESSED_DIR
    x_path = os.path.join(proc, f"{ticker}_X_windows.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"Veri seti bulunamadi. Once calistirin:\n"
            f"  python -m src.data.build_dataset --ticker {ticker}"
        )

    manifest = _read_json(os.path.join(proc, f"{ticker}_manifest.json")) or {}
    fc = manifest.get("feature_columns", FEATURE_COLUMNS)
    ws = manifest.get("window_size", WINDOW_SIZE)

    df = pd.read_csv(os.path.join(proc, f"{ticker}_processed_scaled.csv"), index_col=0, parse_dates=True)
    y_col = "Direction" if "Direction" in df.columns else "direction"
    X_ml = df[fc].values if all(c in df.columns for c in fc) else df[[c for c in fc if c in df.columns]].values

    n = len(df) - ws
    X_xgb_list, y_list = [], []
    for i in range(n):
        X_xgb_list.append(X_ml[i + ws - 1])
        y_list.append(int(df[y_col].iloc[i + ws - 1]))

    scaler_path = os.path.join(proc, f"{ticker}_feature_scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    close_col = "Close" if "Close" in df.columns else "close"
    closes = df[close_col].values[ws - 1 : ws - 1 + len(y_list)]

    return MLDataset(
        ticker=ticker,
        feature_columns=fc,
        window_size=ws,
        X_lstm=np.load(x_path),
        y_lstm=np.load(os.path.join(proc, f"{ticker}_y_targets.npy")),
        X_xgb=np.array(X_xgb_list, dtype=np.float32),
        y_xgb=np.array(y_list, dtype=np.int32),
        features_df=df,
        close_aligned=closes,
        scaler=scaler,
    )
