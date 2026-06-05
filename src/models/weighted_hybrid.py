"""
weighted_hybrid.py — Agirlikli Hibrit Karar Modulu
====================================================
Kripto Para Trading Botu | Ahmet Yilmaz

Uc modelin (RF, XGBoost, LSTM) olasilik ciktilarini agirlikli
ortalama ile birlestirerek AL / TUT / SAT sinyali uretir.

Kullanim:
    from src.models.weighted_hybrid import (
        compute_weighted_hybrid, generate_signals, select_weights
    )

    final_probs = compute_weighted_hybrid(rf_p, xgb_p, lstm_p)
    signals = generate_signals(final_probs)
"""
from __future__ import annotations

import numpy as np


# ══════════════════════════════════════════════════════════════
#  AGIRLIKLI BIRLESTIRME
# ══════════════════════════════════════════════════════════════

def compute_weighted_hybrid(
    rf_probs: np.ndarray,
    xgb_probs: np.ndarray,
    lstm_probs: np.ndarray,
    w_rf: float = 0.40,
    w_xgb: float = 0.40,
    w_lstm: float = 0.20,
) -> np.ndarray:
    """
    Uc modelin olasilik ciktilarini agirlikli ortalama ile birlestirir.

    Parametreler:
        rf_probs   : RF'nin UP sinifi icin olaslik ciktisi (n,)
        xgb_probs  : XGBoost'un UP sinifi icin olaslik ciktisi (n,)
        lstm_probs : LSTM'in sigmoid ciktisi (n,)
        w_rf       : RF agirligi  (varsayilan 0.40)
        w_xgb      : XGBoost agirligi (varsayilan 0.40)
        w_lstm     : LSTM agirligi (varsayilan 0.20)

    Donus:
        final_probs : Agirlikli ortalama olasiliklar (n,)

    Not:
        Agirliklar toplami 1.0 olmalidir. Fonksiyon bunu kontrol eder
        ve gerekirse normalizasyon uygular.
    """
    rf_probs = np.asarray(rf_probs, dtype=np.float64).ravel()
    xgb_probs = np.asarray(xgb_probs, dtype=np.float64).ravel()
    lstm_probs = np.asarray(lstm_probs, dtype=np.float64).ravel()

    # Uzunluk kontrolu
    n = len(rf_probs)
    if len(xgb_probs) != n or len(lstm_probs) != n:
        raise ValueError(
            f"Olasilik dizileri ayni uzunlukta olmali: "
            f"RF={n}, XGB={len(xgb_probs)}, LSTM={len(lstm_probs)}"
        )

    # Agirliklari normalize et (toplam 1.0)
    w_total = w_rf + w_xgb + w_lstm
    if abs(w_total - 1.0) > 1e-6:
        w_rf /= w_total
        w_xgb /= w_total
        w_lstm /= w_total

    final_probs = w_rf * rf_probs + w_xgb * xgb_probs + w_lstm * lstm_probs
    return final_probs


# ══════════════════════════════════════════════════════════════
#  SINYAL URETIMI
# ══════════════════════════════════════════════════════════════

def generate_signals(
    final_probs: np.ndarray,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> np.ndarray:
    """
    Olasilik dizisinden AL / TUT / SAT sinyalleri uretir.

    Esik mantigi:
        prob > buy_threshold  → 2 (BUY / AL)
        prob < sell_threshold → 0 (SELL / SAT)
        diger                 → 1 (HOLD / TUT)

    Parametreler:
        final_probs    : Agirlikli birlestirme sonrasi olasiliklar (n,)
        buy_threshold  : Alis esigi (varsayilan 0.55)
        sell_threshold : Satis esigi (varsayilan 0.45)

    Donus:
        signals : Sinyal dizisi (n,)  — 0=SELL, 1=HOLD, 2=BUY
    """
    final_probs = np.asarray(final_probs, dtype=np.float64).ravel()

    if buy_threshold <= sell_threshold:
        raise ValueError(
            f"buy_threshold ({buy_threshold}) sell_threshold'dan "
            f"({sell_threshold}) buyuk olmali."
        )

    signals = np.ones(len(final_probs), dtype=np.int32)  # varsayilan HOLD
    signals[final_probs > buy_threshold] = 2   # BUY
    signals[final_probs < sell_threshold] = 0  # SELL

    return signals


# ══════════════════════════════════════════════════════════════
#  DEJENERE MODEL TESPITI
# ══════════════════════════════════════════════════════════════

def is_degenerate(predictions: np.ndarray, threshold: float = 0.95) -> bool:
    """
    Modelin dejenere olup olmadigini kontrol eder.

    Dejenere model: Tahminlerin buyuk cogunlugu tek sinifa aittir.
    Ornegin %95'ten fazla "UP" veya %95'ten fazla "DOWN" tahmini
    yapan bir model gercek ogrenme yapmamistir.

    Parametreler:
        predictions : Model tahminleri (0/1 ikili veya olasilik)
        threshold   : Tek sinif orani esigi (varsayilan 0.95)

    Donus:
        True ise model dejenere, False ise normal.
    """
    predictions = np.asarray(predictions).ravel()

    if len(predictions) == 0:
        return True

    # Eger olasilik ise ikili sinifa cevir
    if not np.all(np.isin(predictions, [0, 1])):
        predictions = (predictions > 0.5).astype(int)

    unique_classes = np.unique(predictions)
    if len(unique_classes) <= 1:
        return True

    # Tek sinif orani kontrolu
    up_ratio = predictions.mean()
    return up_ratio > threshold or up_ratio < (1 - threshold)


# ══════════════════════════════════════════════════════════════
#  ADAPTIF AGIRLIK SECIMI
# ══════════════════════════════════════════════════════════════

def select_weights(lstm_val_f1: float) -> tuple[float, float, float]:
    """
    LSTM'in dogrulama setindeki F1 skoruna gore agirlik secimi yapar.

    LSTM kalitesine gore uc senaryo:
        1. F1 > 0.40  : LSTM iyi calisiyor → (0.40, 0.40, 0.20)
        2. 0 < F1 <= 0.40 : LSTM zayif → (0.45, 0.45, 0.10)
        3. F1 == 0 veya dejenere : LSTM kullanilmaz → (0.50, 0.50, 0.00)

    Parametreler:
        lstm_val_f1 : LSTM'in dogrulama seti F1 skoru

    Donus:
        (w_rf, w_xgb, w_lstm) agirliklari
    """
    if lstm_val_f1 > 0.40:
        # LSTM iyi calisiyor — standart agirliklar
        return (0.40, 0.40, 0.20)
    elif lstm_val_f1 > 0.0:
        # LSTM zayif ama bir seyler ogreniyor — dusuk agirlik
        return (0.45, 0.45, 0.10)
    else:
        # LSTM dejenere veya F1=0 — tamamen devre disi
        return (0.50, 0.50, 0.00)


# ══════════════════════════════════════════════════════════════
#  SINYAL ETIKETLERI (YARDIMCI)
# ══════════════════════════════════════════════════════════════

SIGNAL_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


def signal_to_label(signal: int) -> str:
    """Sinyal kodunu okunabilir etikete cevirir."""
    return SIGNAL_LABELS.get(signal, "UNKNOWN")


def signal_distribution(signals: np.ndarray) -> dict:
    """
    Sinyal dagilimini rapor eder.

    Donus:
        {"BUY": (adet, oran), "HOLD": (adet, oran), "SELL": (adet, oran)}
    """
    signals = np.asarray(signals).ravel()
    n = len(signals)
    if n == 0:
        return {}

    dist = {}
    for code, label in SIGNAL_LABELS.items():
        count = int(np.sum(signals == code))
        ratio = count / n
        dist[label] = (count, ratio)
    return dist
