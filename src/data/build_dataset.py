"""
ML veri seti olusturucu — LSTM ve XGBoost icin tek, tutarli cikti.

Adimlar:
  1. OHLCV cek
  2. Duragan ozellikler uret
  3. Dead-zone filtreleme (kucuk fiyat hareketlerini cikar)
  4. Kisa veri analizi raporu yazdir
  5. LSTM: StandardScaler (sadece egitim gunleri) + 60 gunluk pencere
  6. XGBoost: ayni gunlerdeki tek satirlik ozellikler (olceklenmemis)
  7. Unified split indices hesapla (tum modeller ayni bolumu kullanir)
  8. manifest.json + legacy dosyalar (data/processed/)
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.feature_engineering import build_features
from src.data.fetch_ohlcv import fetch_ohlcv
from src.data.ml_config import (
    DATA_ML_DIR,
    DATA_PROCESSED_DIR,
    DIRECTION_THRESHOLD,
    FEATURE_COLUMNS,
    PRICE_COLUMN,
    PURGE_GAP,
    SCALER_FIT_RATIO,
    TARGET_COLUMN,
    TICKERS,
    TRAIN_RATIO,
    VAL_RATIO,
    WINDOW_SIZE,
)


def _ticker_dir(ticker: str) -> str:
    path = os.path.join(DATA_ML_DIR, ticker)
    os.makedirs(path, exist_ok=True)
    return path


def compute_split_indices(n_samples: int) -> dict:
    """
    Tum modeller icin AYNI train/val/test bolme indekslerini hesaplar.
    Her set arasinda PURGE_GAP kadar boslik birakilir.

    Duzen:
    [───── TRAIN ─────][gap][──── VAL ────][gap][──── TEST ────]
    """
    usable = n_samples - 2 * PURGE_GAP
    train_end = int(usable * TRAIN_RATIO)
    val_size = int(usable * VAL_RATIO)
    val_end = train_end + val_size

    split = {
        "train": [0, train_end],
        "val": [train_end + PURGE_GAP, val_end + PURGE_GAP],
        "test": [val_end + 2 * PURGE_GAP, n_samples],
    }
    return split


def print_eda_report(df: pd.DataFrame, ticker: str) -> None:
    """Basit veri analizi — baslangic seviyesi icin anlasilir ozet."""
    print(f"\n{'=' * 60}")
    print(f"  VERI ANALIZI OZETI — {ticker}")
    print(f"{'=' * 60}")
    print(f"  Tarih araligi : {df.index.min().date()}  ->  {df.index.max().date()}")
    print(f"  Toplam gun    : {len(df):,}")
    print(f"  Ozellik sayisi: {len(FEATURE_COLUMNS)}")

    up = int((df[TARGET_COLUMN] == 1).sum())
    down = int((df[TARGET_COLUMN] == 0).sum())
    dead = int((df[TARGET_COLUMN] == -1).sum())
    print(f"\n  Hedef (direction, threshold={DIRECTION_THRESHOLD:.4f}):")
    print(f"    YUKARI  (1) : {up:5d}  ({100 * up / len(df):.1f}%)")
    print(f"    ASAGI   (0) : {down:5d}  ({100 * down / len(df):.1f}%)")
    print(f"    DEAD-ZONE   : {dead:5d}  ({100 * dead / len(df):.1f}%) [cikarilacak]")

    missing = df[FEATURE_COLUMNS].isnull().sum().sum()
    print(f"\n  Eksik hucre   : {missing} (0 olmali)")

    print(f"\n  Ornek istatistik (ilk 5 ozellik):")
    stats = df[FEATURE_COLUMNS[:5]].describe().T[["mean", "std", "min", "max"]]
    for name, row in stats.iterrows():
        print(f"    {name:20s}  ort={row['mean']:+.4f}  std={row['std']:.4f}")

    print(f"{'=' * 60}\n")


def _create_lstm_windows(
    feature_matrix: np.ndarray,
    directions: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(N, window, F) ve (N,) hedef dizileri."""
    x_list, y_list = [], []
    for i in range(len(feature_matrix) - window_size):
        x_list.append(feature_matrix[i : i + window_size])
        y_list.append(directions[i + window_size - 1])
    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def _create_xgb_tabular(
    feature_matrix: np.ndarray,
    directions: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """LSTM ile ayni zaman noktasinda tek satir (pencerenin son gunu)."""
    x_list, y_list = [], []
    for i in range(len(feature_matrix) - window_size):
        x_list.append(feature_matrix[i + window_size - 1])
        y_list.append(directions[i + window_size - 1])
    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def build(ticker: str = "BTC-USD") -> bool:
    print(f"\n{'#' * 60}")
    print(f"  ML VERI SETI OLUSTURULUYOR: {ticker}")
    print(f"{'#' * 60}")

    ohlcv = fetch_ohlcv(ticker)
    features_df = build_features(ohlcv)

    if len(features_df) < WINDOW_SIZE + 50:
        print(f"HATA: Yetersiz veri ({len(features_df)} gun). En az {WINDOW_SIZE + 50} gerekli.")
        return False

    print_eda_report(features_df, ticker)

    # --- Dead-zone filtreleme ---
    n_before = len(features_df)
    features_df = features_df[features_df[TARGET_COLUMN] != -1].copy()
    n_removed = n_before - len(features_df)
    features_df[TARGET_COLUMN] = features_df[TARGET_COLUMN].astype(int)
    print(f"  Dead-zone filtreleme: {n_removed} ornek cikarildi "
          f"({100 * n_removed / n_before:.1f}%), kalan: {len(features_df)}")

    out_dir = _ticker_dir(ticker)
    features_df.to_csv(os.path.join(out_dir, "features.csv"))

    # --- Olcekleme (yalnizca ilk %70 gun — LSTM icin) ---
    n_rows = len(features_df)
    fit_end = int(n_rows * SCALER_FIT_RATIO)
    scaler = StandardScaler()
    X_raw = features_df[FEATURE_COLUMNS].values
    scaler.fit(X_raw[:fit_end])
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    directions = features_df[TARGET_COLUMN].values.astype(np.int32)
    closes = features_df[PRICE_COLUMN].values

    X_lstm, y_lstm = _create_lstm_windows(X_scaled, directions, WINDOW_SIZE)
    X_xgb, y_xgb = _create_xgb_tabular(X_raw, directions, WINDOW_SIZE)

    assert len(X_lstm) == len(X_xgb), "LSTM ve XGBoost ornek sayisi uyusmuyor"

    # --- Unified split indices ---
    n_samples = len(X_lstm)
    split = compute_split_indices(n_samples)

    # Sinif dagilimini split bazinda hesapla
    class_balance = {}
    for split_name, (s, e) in split.items():
        y_part = y_lstm[s:e]
        class_balance[split_name] = {
            "up": int(np.sum(y_part == 1)),
            "down": int(np.sum(y_part == 0)),
            "total": int(len(y_part)),
        }

    print(f"\n  Unified Split (tum modeller icin):")
    for split_name, (s, e) in split.items():
        cb = class_balance[split_name]
        print(f"    {split_name:5s}: [{s:5d}:{e:5d}] "
              f"({cb['total']:4d} ornek, UP={cb['up']}, DOWN={cb['down']})")

    # --- Kaydet ---
    np.save(os.path.join(out_dir, "X_lstm.npy"), X_lstm)
    np.save(os.path.join(out_dir, "y.npy"), y_lstm)  # Unified target (tek dosya)
    np.save(os.path.join(out_dir, "X_xgb.npy"), X_xgb)
    np.save(os.path.join(out_dir, "close_aligned.npy"), closes[WINDOW_SIZE - 1:])

    # Legacy uyumluluk: eski isimli dosyalar da yaz
    np.save(os.path.join(out_dir, "y_lstm.npy"), y_lstm)
    np.save(os.path.join(out_dir, "y_xgb.npy"), y_xgb)

    joblib.dump(scaler, os.path.join(out_dir, "feature_scaler.pkl"))

    manifest = {
        "ticker": ticker,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "window_size": WINDOW_SIZE,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "price_column": PRICE_COLUMN,
        "direction_threshold": DIRECTION_THRESHOLD,
        "n_days_original": int(n_before),
        "n_days_after_filter": int(n_rows),
        "dead_zone_removed": int(n_removed),
        "n_samples": int(n_samples),
        "scaler_fit_rows": int(fit_end),
        "date_start": str(features_df.index.min().date()),
        "date_end": str(features_df.index.max().date()),
        "split_indices": split,
        "class_balance": class_balance,
        "version": 3,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    _write_legacy_compat(ticker, features_df, X_lstm, y_lstm, scaler, split)

    print(f"\n  Kaydedildi: {out_dir}")
    print(f"  LSTM X shape: {X_lstm.shape}  |  XGBoost X shape: {X_xgb.shape}")
    print(f"  Ornek sayisi: {n_samples:,} (her ikisi icin ayni)")
    return True


def _write_legacy_compat(
    ticker: str,
    features_df: pd.DataFrame,
    X_lstm: np.ndarray,
    y_lstm: np.ndarray,
    scaler: StandardScaler,
    split: dict | None = None,
) -> None:
    """
    Eski kod (time_series_models, ensemble, dashboard) icin
    data/processed/ dosyalarini uretir.
    """
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    scaled = features_df.copy()
    scaled[FEATURE_COLUMNS] = scaler.transform(features_df[FEATURE_COLUMNS].values)
    scaled["Direction"] = features_df[TARGET_COLUMN]
    scaled["Close"] = features_df[PRICE_COLUMN]
    scaled.to_csv(os.path.join(DATA_PROCESSED_DIR, f"{ticker}_processed_scaled.csv"))

    np.save(os.path.join(DATA_PROCESSED_DIR, f"{ticker}_X_windows.npy"), X_lstm)
    np.save(os.path.join(DATA_PROCESSED_DIR, f"{ticker}_y_targets.npy"), y_lstm)
    joblib.dump(scaler, os.path.join(DATA_PROCESSED_DIR, f"{ticker}_feature_scaler.pkl"))

    legacy_manifest = {
        "feature_columns": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "direction_threshold": DIRECTION_THRESHOLD,
        "version": 3,
    }
    if split is not None:
        legacy_manifest["split_indices"] = split
    with open(os.path.join(DATA_PROCESSED_DIR, f"{ticker}_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(legacy_manifest, f, indent=2)


def build_all(tickers: list[str] | None = None) -> None:
    tickers = tickers or TICKERS
    ok = 0
    for t in tickers:
        if build(t):
            ok += 1
    print(f"\nTamamlandi: {ok}/{len(tickers)} coin basarili.")


def main(ticker: str = "BTC-USD") -> bool:
    return build(ticker)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML veri seti olustur (LSTM + XGBoost)")
    parser.add_argument("--ticker", default="BTC-USD", help="Orn: BTC-USD")
    parser.add_argument("--all", action="store_true", help="Tum coinler")
    args = parser.parse_args()
    if args.all:
        build_all()
    else:
        main(args.ticker)
