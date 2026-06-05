"""
weight_selector.py — Agirlik Secici (Dogrulama Seti Uzerinde)
===============================================================
Kripto Para Trading Botu | Ahmet Yilmaz

Farkli agirlik konfigurasyonlarini dogrulama (validation) seti
uzerinde deneyerek en iyi RF/XGBoost/LSTM agirlik kombinasyonunu
secer. Bu islem TEST setinden ONCE yapilir — veri sizintisi yoktur.

Kullanim:
    from src.evaluation.weight_selector import evaluate_weight_configs

    best = evaluate_weight_configs(store, rf_model, xgb_model, lstm_model, "BTC-USD")
    print(best)  # {"weights": (0.4, 0.4, 0.2), "f1": 0.58, ...}
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

from src.data.feature_store import FeatureStore
from src.models.weighted_hybrid import (
    compute_weighted_hybrid,
    generate_signals,
    is_degenerate,
)
from src.evaluation.metrics import compute_classification_metrics


# ══════════════════════════════════════════════════════════════
#  VARSAYILAN AGIRLIK KONFIGURASYONLARI
# ══════════════════════════════════════════════════════════════

DEFAULT_WEIGHT_CONFIGS = [
    (0.45, 0.45, 0.10),   # Tercih edilen — LSTM dusuk destekleyici sinyal
    (0.40, 0.40, 0.20),   # Karsilastirma — LSTM orta agirlik
    (0.50, 0.50, 0.00),   # Fallback — LSTM devre disi, sadece RF+XGB
]


# ══════════════════════════════════════════════════════════════
#  MODEL TAHMINLERI (YARDIMCI)
# ══════════════════════════════════════════════════════════════

def get_model_predictions(
    store: FeatureStore,
    rf_model,
    xgb_model,
    lstm_model,
    split_name: str = "val",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Uc modelden olasilik tahminlerini alir.

    Parametreler:
        store      : FeatureStore nesnesi
        rf_model   : Egitilmis RandomForest modeli
        xgb_model  : Egitilmis XGBoost modeli
        lstm_model : Egitilmis LSTM modeli (TimeSeriesNet)
        split_name : Veri bolumu ("train", "val", "test")

    Donus:
        (rf_probs, xgb_probs, lstm_probs, y_true)
        Her biri (n,) boyutunda numpy dizisi.

    Not:
        RF ve XGB icin tabular veri (X_xgb),
        LSTM icin pencereli veri (X_lstm) kullanilir.
        Ayni FeatureStore'dan geldigi icin indeksler uyumludur.
    """
    # ── RF ve XGBoost tahminleri ──
    X_xgb, y_true = store.get_xgb_split(split_name)

    # RF: UP sinifi olasiligi (sinif 1)
    rf_probs = rf_model.predict_proba(X_xgb)[:, 1]

    # XGBoost: UP sinifi olasiligi (sinif 1)
    xgb_probs = xgb_model.predict_proba(X_xgb)[:, 1]

    # ── LSTM tahminleri ──
    X_lstm, _ = store.get_lstm_split(split_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.to(device)
    lstm_model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_lstm).to(device)
        logits = lstm_model(X_tensor)
        lstm_probs = torch.sigmoid(logits).cpu().numpy().flatten()

    return rf_probs, xgb_probs, lstm_probs, y_true


# ══════════════════════════════════════════════════════════════
#  LSTM THRESHOLD YUKLEME (YARDIMCI)
# ══════════════════════════════════════════════════════════════

def _load_lstm_threshold(ticker: str) -> float:
    """
    Kaydedilmis LSTM threshold degerini yukler.
    Bulunamazsa varsayilan 0.50 dondurur.
    """
    from src.data.ml_config import PROJECT_ROOT

    thr_path = os.path.join(
        PROJECT_ROOT, "src", "models", "saved_models",
        f"{ticker}_dl_thresholds.json"
    )
    if os.path.exists(thr_path):
        with open(thr_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        return thresholds.get("LSTM", 0.50)
    return 0.50


# ══════════════════════════════════════════════════════════════
#  AGIRLIK KONFIGURASYONLARINI DEGERLENDIR
# ══════════════════════════════════════════════════════════════

def evaluate_weight_configs(
    store: FeatureStore,
    rf_model,
    xgb_model,
    lstm_model,
    ticker: str,
    weight_configs: list[tuple[float, float, float]] | None = None,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> dict[str, Any]:
    """
    Farkli agirlik konfigurasyonlarini dogrulama seti uzerinde dener.

    Islem adimlari:
        1. Dogrulama setinden uc modelin olasilik ciktilarini alir
        2. Her agirlik konfigurasyonu icin:
           a. Agirlikli birlestirme uygular
           b. Sinyal uretir
           c. F1 skoru hesaplar
        3. En yuksek F1 skoruna sahip konfigurasyonu secer

    Parametreler:
        store          : FeatureStore nesnesi
        rf_model       : Egitilmis RF modeli
        xgb_model      : Egitilmis XGBoost modeli
        lstm_model     : Egitilmis LSTM modeli
        ticker         : Coin adi (ornegin "BTC-USD")
        weight_configs : Denenecek agirlik listesi [(w_rf, w_xgb, w_lstm), ...]
        buy_threshold  : Alis esigi (sinyal uretimi icin)
        sell_threshold : Satis esigi (sinyal uretimi icin)

    Donus:
        dict: {
            "best_weights": (w_rf, w_xgb, w_lstm),
            "best_f1": float,
            "best_accuracy": float,
            "all_results": [
                {"weights": (w_rf, w_xgb, w_lstm), "f1": float, "accuracy": float},
                ...
            ],
            "lstm_degenerate": bool,
            "ticker": str,
        }
    """
    if weight_configs is None:
        weight_configs = DEFAULT_WEIGHT_CONFIGS

    print(f"\n{'─' * 60}")
    print(f"  {ticker} — Agirlik Secimi (Dogrulama Seti)")
    print(f"{'─' * 60}")

    # ── Dogrulama setinden tahminler ──
    rf_probs, xgb_probs, lstm_probs, y_true = get_model_predictions(
        store, rf_model, xgb_model, lstm_model, "val"
    )

    print(f"  Dogrulama seti boyutu: {len(y_true)}")
    print(f"  Sinif dagilimi: UP={int(np.sum(y_true == 1))}, "
          f"DOWN={int(np.sum(y_true == 0))}")

    # ── LSTM dejenere kontrolu ──
    lstm_deg = is_degenerate(lstm_probs)
    if lstm_deg:
        print(f"  ⚠️  LSTM dejenere tespit edildi — LSTM agirligi 0 olabilir.")

    # ── Her konfigurasyonu dene ──
    all_results = []

    for w_rf, w_xgb, w_lstm in weight_configs:
        # LSTM dejenere ise ve agirlik > 0 → atla
        if lstm_deg and w_lstm > 0:
            print(f"  ({w_rf:.2f}, {w_xgb:.2f}, {w_lstm:.2f}) — LSTM dejenere, atlandi")
            all_results.append({
                "weights": (w_rf, w_xgb, w_lstm),
                "f1": 0.0,
                "accuracy": 0.0,
                "skipped": True,
            })
            continue

        # Agirlikli birlestirme
        final_probs = compute_weighted_hybrid(
            rf_probs, xgb_probs, lstm_probs,
            w_rf=w_rf, w_xgb=w_xgb, w_lstm=w_lstm,
        )

        # Ikili tahmin (UP/DOWN) — olasilik 0.5 esigi
        y_pred = (final_probs > 0.5).astype(int)

        # Metrikler
        metrics = compute_classification_metrics(y_true, y_pred, final_probs)

        result = {
            "weights": (w_rf, w_xgb, w_lstm),
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "skipped": False,
        }
        all_results.append(result)

        print(f"  ({w_rf:.2f}, {w_xgb:.2f}, {w_lstm:.2f}) → "
              f"F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    # ── En iyi konfigurasyonu sec ──
    valid_results = [r for r in all_results if not r.get("skipped", False)]

    if not valid_results:
        # Tum konfigurasyonlar atlandi — varsayilan ML-only
        best = {
            "weights": (0.50, 0.50, 0.00),
            "f1": 0.0,
            "accuracy": 0.0,
        }
        print(f"\n  ⚠️  Tum konfigurasyonlar atlandi. Varsayilan: (0.50, 0.50, 0.00)")
    else:
        best = max(valid_results, key=lambda r: r["f1"])

    print(f"\n  ✅ En iyi agirliklar: {best['weights']}")
    print(f"     F1={best['f1']:.4f}, Accuracy={best['accuracy']:.4f}")

    return {
        "best_weights": best["weights"],
        "best_f1": best["f1"],
        "best_accuracy": best["accuracy"],
        "all_results": all_results,
        "lstm_degenerate": lstm_deg,
        "ticker": ticker,
    }
