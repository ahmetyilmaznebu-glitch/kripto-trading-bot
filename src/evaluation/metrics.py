"""
metrics.py — Model Degerlendirme Metrikleri
=============================================
Kripto Para Trading Botu | Ahmet Yilmaz

Siniflandirma ve backtest metriklerini hesaplayan fonksiyonlar.
Tum modeller (RF, XGBoost, LSTM, Hibrit) icin ortak metrik altyapisi saglar.

Kullanim:
    from src.evaluation.metrics import (
        compute_classification_metrics,
        compute_backtest_metrics,
        compare_to_dummy,
        save_metrics,
    )

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    save_metrics(metrics, "sonuclar.json")
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.dummy import DummyClassifier


# ══════════════════════════════════════════════════════════════
#  SINIFLANDIRMA METRIKLERI
# ══════════════════════════════════════════════════════════════

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Siniflandirma performans metriklerini hesaplar.

    Parametreler:
        y_true : Gercek etiketler (n,)
        y_pred : Model tahminleri (n,)
        y_prob : Olasilik ciktisi (n,) — AUC-ROC icin (opsiyonel)

    Donus:
        dict: {
            "accuracy": float,
            "f1": float,
            "precision": float,
            "recall": float,
            "confusion_matrix": [[TN, FP], [FN, TP]],
            "auc_roc": float veya None,
            "support": {"total": int, "class_0": int, "class_1": int},
        }
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # AUC-ROC hesapla (olasilik verildiyse ve her iki sinif mevcutsa)
    auc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()
        unique_classes = np.unique(y_true)
        if len(unique_classes) >= 2:
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                auc = None

    # Sinif dagilimi
    n_class_0 = int(np.sum(y_true == 0))
    n_class_1 = int(np.sum(y_true == 1))

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm.tolist(),
        "auc_roc": auc,
        "support": {
            "total": len(y_true),
            "class_0": n_class_0,
            "class_1": n_class_1,
        },
    }


# ══════════════════════════════════════════════════════════════
#  BACKTEST METRIKLERI
# ══════════════════════════════════════════════════════════════

def compute_backtest_metrics(
    portfolio_values: np.ndarray,
    initial_capital: float,
    trades: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Backtest performans metriklerini hesaplar.

    Parametreler:
        portfolio_values : Portfoy deger serisi (n,) — gunluk
        initial_capital  : Baslangic sermayesi
        trades           : Islem kayitlari listesi
                           [{"type": "BUY"/"SELL", "pnl": float}, ...]

    Donus:
        dict: {
            "total_return": float (yuzde),
            "sharpe_ratio": float,
            "max_drawdown": float (yuzde),
            "win_rate": float veya None,
            "trade_count": int,
            "final_value": float,
        }
    """
    portfolio_values = np.asarray(portfolio_values, dtype=np.float64).ravel()

    if len(portfolio_values) < 2:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": None,
            "trade_count": 0,
            "final_value": float(initial_capital),
        }

    # ── Toplam getiri ──
    final_value = float(portfolio_values[-1])
    total_return = ((final_value - initial_capital) / initial_capital) * 100.0

    # ── Gunluk getiriler ve Sharpe orani ──
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10:
        # Yillik Sharpe (252 islem gunu)
        sharpe = float(
            np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        )
    else:
        sharpe = 0.0

    # ── Maksimum dususme (Max Drawdown) ──
    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cummax) / cummax
    max_drawdown = float(np.min(drawdowns)) * 100.0  # yuzde

    # ── Kazanma orani ──
    win_rate = None
    trade_count = 0
    if trades is not None and len(trades) > 0:
        trade_count = len(trades)
        pnls = [t.get("pnl", 0) for t in trades]
        winning = sum(1 for p in pnls if p > 0)
        win_rate = float(winning / trade_count) if trade_count > 0 else 0.0

    return {
        "total_return": round(total_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "trade_count": trade_count,
        "final_value": round(final_value, 2),
    }


# ══════════════════════════════════════════════════════════════
#  DUMMY BASELINE KARSILASTIRMASI
# ══════════════════════════════════════════════════════════════

def compare_to_dummy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Modeli dummy (cogunluk sinifi) baseline ile karsilastirir.

    Gercek bir ML modelinin katma degerini olcmek icin,
    hic ogrenmeyen bir model (DummyClassifier) ile kiyaslar.

    Parametreler:
        y_true : Gercek etiketler (n,)
        y_pred : Model tahminleri (n,)

    Donus:
        dict: {
            "model_accuracy": float,
            "model_f1": float,
            "dummy_majority_accuracy": float,
            "dummy_majority_f1": float,
            "dummy_stratified_accuracy": float,
            "dummy_stratified_f1": float,
            "beats_majority": bool,
            "beats_stratified": bool,
        }
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    # Model metrikleri
    model_acc = float(accuracy_score(y_true, y_pred))
    model_f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Dummy — Cogunluk sinifi (most_frequent)
    dummy_maj = DummyClassifier(strategy="most_frequent", random_state=42)
    # fit icin sahte X olustur (DummyClassifier X'e bakmaz)
    X_fake = np.zeros((len(y_true), 1))
    dummy_maj.fit(X_fake, y_true)
    dummy_maj_pred = dummy_maj.predict(X_fake)
    dummy_maj_acc = float(accuracy_score(y_true, dummy_maj_pred))
    dummy_maj_f1 = float(f1_score(y_true, dummy_maj_pred, zero_division=0))

    # Dummy — Orantili rastgele (stratified)
    dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
    dummy_strat.fit(X_fake, y_true)
    dummy_strat_pred = dummy_strat.predict(X_fake)
    dummy_strat_acc = float(accuracy_score(y_true, dummy_strat_pred))
    dummy_strat_f1 = float(f1_score(y_true, dummy_strat_pred, zero_division=0))

    return {
        "model_accuracy": model_acc,
        "model_f1": model_f1,
        "dummy_majority_accuracy": dummy_maj_acc,
        "dummy_majority_f1": dummy_maj_f1,
        "dummy_stratified_accuracy": dummy_strat_acc,
        "dummy_stratified_f1": dummy_strat_f1,
        "beats_majority": model_acc > dummy_maj_acc and model_f1 > 0,
        "beats_stratified": model_acc > dummy_strat_acc and model_f1 > 0,
    }


# ══════════════════════════════════════════════════════════════
#  KAYDETME
# ══════════════════════════════════════════════════════════════

def save_metrics(metrics_dict: dict, filepath: str) -> None:
    """
    Metrik sozlugunu JSON dosyasina kaydeder.

    Parametreler:
        metrics_dict : Kaydedilecek metrikler
        filepath     : Hedef dosya yolu (.json)

    Not:
        Dosya yolu icindeki dizinler otomatik olusturulur.
        numpy dizileri otomatik olarak listeye cevrilir.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    # numpy tiplerini JSON-uyumlu hale getir
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    serializable = _convert(metrics_dict)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"  💾 Metrikler kaydedildi: {filepath}")
