"""
Meta Inference — Weighted Hybrid model ile final_prob hesaplama.

Eski ensemble_model + GradientBoosting meta-model yerine,
seffaf agirlikli hibrit formul kullanir:

    final_prob = w_rf * RF_prob + w_xgb * XGB_prob + w_lstm * LSTM_prob

Kullanim:
    from src.models.meta_inference import compute_final_probs
    probs, closes = compute_final_probs("BTC-USD")
"""
import os
import sys
import json
import numpy as np
import joblib
import torch

# Proje kokunu sys.path'e ekle
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.feature_store import FeatureStore
from src.data.ml_config import TICKERS
from src.models.time_series_models import TimeSeriesNet


def _detect_num_layers(state_dict):
    """State dict'ten num_layers'i otomatik tespit eder."""
    return max(len([k for k in state_dict.keys() if k.startswith('rnn.weight_ih')]), 1)


def load_lstm(models_dir: str, input_size: int, ticker: str = "BTC-USD"):
    """LSTM modelini yukler ve eval modunda dondurur."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_path = os.path.join(models_dir, f'{ticker}_lstm_best.pth')
    if not os.path.exists(lstm_path):
        return None

    lstm_sd = torch.load(lstm_path, map_location=device, weights_only=True)
    num_layers = _detect_num_layers(lstm_sd)
    # hidden_size'i state_dict'ten otomatik tespit et
    hidden_size = lstm_sd['rnn.weight_ih_l0'].shape[0] // 4  # LSTM: 4 * hidden_size
    lstm = TimeSeriesNet(input_size, hidden_size=hidden_size, num_layers=num_layers,
                         output_size=1, model_type="LSTM", use_attention=True)
    lstm.load_state_dict(lstm_sd)
    lstm.to(device)
    lstm.eval()
    return lstm


def get_lstm_probs(lstm_model, X_lstm: np.ndarray) -> np.ndarray:
    """LSTM modelinden olasilik tahminleri alir (sigmoid uygular)."""
    if lstm_model is None:
        return np.full(len(X_lstm), 0.5)  # Notr olasilik

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        tensor = torch.FloatTensor(X_lstm).to(device)
        logits = lstm_model(tensor).cpu().numpy().flatten()
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    return probs


def compute_final_probs(ticker: str = "BTC-USD",
                        split_name: str = "test",
                        w_rf: float = 0.4,
                        w_xgb: float = 0.4,
                        w_lstm: float = 0.2):
    """
    Weighted hybrid formulu ile final olasiliklari hesaplar.

    Donus:
        final_probs: np.ndarray — UP olasiligi (0-1)
        closes: np.ndarray — Hizalanmis kapanis fiyatlari
        component_probs: dict — Her modelin ayri olasiliklari
    """
    base_dir = _project_root
    models_dir = os.path.join(base_dir, "src", "models", "saved_models")

    # FeatureStore'dan veri al (unified split)
    store = FeatureStore(ticker)
    X_lstm, y = store.get_lstm_split(split_name)
    X_xgb, _ = store.get_xgb_split(split_name)
    closes = store.get_close_split(split_name)

    # RF modeli
    rf_path = os.path.join(models_dir, f'{ticker}_rf_classifier.pkl')
    rf_model = joblib.load(rf_path)
    rf_probs = rf_model.predict_proba(X_xgb)[:, 1]

    # XGBoost modeli
    xgb_path = os.path.join(models_dir, f'{ticker}_xgb_classifier.pkl')
    xgb_model = joblib.load(xgb_path)
    xgb_probs = xgb_model.predict_proba(X_xgb)[:, 1]

    # LSTM modeli
    input_size = X_lstm.shape[2]
    lstm_model = load_lstm(models_dir, input_size, ticker)
    lstm_probs = get_lstm_probs(lstm_model, X_lstm)

    # LSTM dejenere kontrolu
    is_degen = _is_degenerate(lstm_probs)
    if is_degen:
        print(f"  [UYARI] {ticker} LSTM dejenere — agirligi 0'a dusuruldu")
        actual_w_rf = w_rf + w_lstm / 2
        actual_w_xgb = w_xgb + w_lstm / 2
        actual_w_lstm = 0.0
    else:
        actual_w_rf = w_rf
        actual_w_xgb = w_xgb
        actual_w_lstm = w_lstm

    # Weighted hybrid formulu
    final_probs = (actual_w_rf * rf_probs +
                   actual_w_xgb * xgb_probs +
                   actual_w_lstm * lstm_probs)

    component_probs = {
        "rf": rf_probs,
        "xgb": xgb_probs,
        "lstm": lstm_probs,
        "weights": {"rf": actual_w_rf, "xgb": actual_w_xgb, "lstm": actual_w_lstm},
        "lstm_degenerate": is_degen,
    }

    return final_probs, closes, component_probs


def _is_degenerate(probs: np.ndarray, threshold: float = 0.95) -> bool:
    """Tahminlerin buyuk cogunlugu tek sinifta mi kontrol eder."""
    preds = (probs > 0.5).astype(int)
    if len(preds) == 0:
        return True
    majority_ratio = max(np.mean(preds), 1 - np.mean(preds))
    return majority_ratio >= threshold


# ── Geriye uyumluluk (eski backtester icin) ─────────────────────

def compute_meta_probs(base_dir: str, ticker: str = "BTC-USD"):
    """
    [DEPRECATED] Eski ensemble meta-model olasiliklari.
    Yeni kod compute_final_probs() kullanmali.

    Geriye uyumluluk icin: final_probs'u meta_probs olarak dondurur.
    """
    import warnings
    warnings.warn(
        "compute_meta_probs() kullanımdan kaldırılacak. "
        "compute_final_probs() kullanın.",
        DeprecationWarning, stacklevel=2
    )
    final_probs, closes, _ = compute_final_probs(ticker, split_name="test")

    # Eski arayuz: df ve df_original dondurmeli
    store = FeatureStore(ticker)
    df = store.features_df

    import pandas as pd
    raw_path = os.path.join(base_dir, "data", "raw", f"{ticker}_raw.csv")
    ohlcv_path = os.path.join(base_dir, "data", "raw", f"{ticker}_ohlcv.csv")
    if os.path.exists(raw_path):
        df_original = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    elif os.path.exists(ohlcv_path):
        df_original = pd.read_csv(ohlcv_path, index_col=0, parse_dates=True)
    else:
        df_original = df.copy() if df is not None else pd.DataFrame()

    # Tum ornekler icin (train+val+test) final_probs hesapla
    all_probs, _, _ = compute_final_probs(ticker, split_name="test")

    return all_probs, df, df_original


WINDOW_SIZE = 60  # Geriye uyumluluk
