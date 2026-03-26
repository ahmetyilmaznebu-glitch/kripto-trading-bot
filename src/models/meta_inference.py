import os
import sys
import numpy as np
import pandas as pd
import joblib

# Proje kokunu sys.path'e ekle (subprocess olarak calistiginda src.* importlari icin)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.ensemble_model import load_models, get_predictions, _add_cross_features


WINDOW_SIZE = 60


def compute_meta_probs(base_dir: str, ticker: str = "BTC-USD"):
    """
    Hazirlanmis veriler ve kayitli modeller uzerinden
    her pencere icin ensemble meta-model olasiligini hesaplar.

    Donus:
        meta_probs: np.ndarray, sekil (num_windows,)
        df: pandas.DataFrame, islenmis ve olceklenmis fiyat/veri DataFrame'i
        df_original: pandas.DataFrame, olceklenmemis orijinal fiyat verisi
                     (RL ve Backtester icin gercek fiyatlar)
    """
    data_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "src", "models", "saved_models")

    # Zaman serisi pencereleri ve islenmis dataframe'i yukle
    X_dl_path = os.path.join(data_dir, f"{ticker}_X_windows.npy")
    df_path = os.path.join(data_dir, f"{ticker}_processed_scaled.csv")

    if not os.path.exists(X_dl_path) or not os.path.exists(df_path):
        raise FileNotFoundError(
            "Gerekli veri dosyalari bulunamadi. Once 'data_pipeline.py', "
            "'time_series_models.py', 'ml_classification_models.py' ve 'ensemble_model.py' calistirilmali."
        )

    X_dl = np.load(X_dl_path, allow_pickle=True).astype(np.float32)
    df = pd.read_csv(df_path, index_col=0)

    # DUZELTME (Hata #4, #5): Orijinal (olceklenmemis) fiyat verisini de yukle
    raw_path = os.path.join(base_dir, "data", "raw", f"{ticker}_raw.csv")
    if os.path.exists(raw_path):
        df_original = pd.read_csv(raw_path, index_col=0)
    else:
        # Fallback: scaler'i ters cevirerek orijinal fiyatlari elde et
        scaler_path = os.path.join(data_dir, f"{ticker}_feature_scaler.pkl")
        if os.path.exists(scaler_path):
            feature_scaler = joblib.load(scaler_path)
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD',
                           'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid', 'SMA_20', 'EMA_50']
            existing_cols = [c for c in feature_cols if c in df.columns]
            df_original = df.copy()
            df_original[existing_cols] = feature_scaler.inverse_transform(df[existing_cols])
        else:
            print("UYARI: Orijinal fiyat verisi bulunamadi, olceklenmis veriler kullanilacak.")
            df_original = df.copy()

    # Tabular ML girdilerini, ensemble_model.main icindeki mantikla uyumlu sekilde hazirla
    ml_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_High",
        "BB_Low",
        "BB_Mid",
        "SMA_20",
        "EMA_50",
    ]

    X_ml_base = df[ml_features].values

    X_ml = []

    for i in range(len(df) - WINDOW_SIZE):
        X_ml.append(X_ml_base[i + WINDOW_SIZE - 1])

    X_ml = np.array(X_ml)

    # Modelleri yukle ve tum pencereler icin base tahminleri al
    lstm, gru, rf, xgb = load_models(models_dir, input_size=X_dl.shape[2], ticker=ticker)
    lstm_all, gru_all, rf_all, xgb_all = get_predictions(X_dl, X_ml, lstm, gru, rf, xgb)

    meta_X = np.column_stack((lstm_all, gru_all, rf_all, xgb_all))
    meta_X = _add_cross_features(meta_X)

    meta_model_path = os.path.join(models_dir, f"{ticker}_ensemble_meta_model.pkl")
    if not os.path.exists(meta_model_path):
        raise FileNotFoundError(
            f"Ensemble meta-model dosyasi bulunamadi: {meta_model_path}. "
            "Once 'ensemble_model.py' calistirilip meta-model egitilmelidir."
        )

    meta_model = joblib.load(meta_model_path)
    meta_probs = meta_model.predict_proba(meta_X)[:, 1]

    return meta_probs, df, df_original


