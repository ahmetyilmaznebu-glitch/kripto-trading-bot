import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch

# Proje kokunu sys.path'e ekle (subprocess olarak calistiginda src.* importlari icin)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.time_series_models import TimeSeriesNet
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def _detect_num_layers(state_dict):
    """State dict'ten num_layers'i otomatik tespit eder.
    Boylece model mimarisi degistiginde (ornegin 3->2) hardcode hata olmaz."""
    return max(len([k for k in state_dict.keys() if k.startswith('rnn.weight_ih')]), 1)


def load_models(models_dir, input_size, ticker="BTC-USD"):
    """Kayitli DL ve ML modellerini yukler."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LSTM modeli — num_layers state dict'ten otomatik tespit edilir
    lstm_path = os.path.join(models_dir, f'{ticker}_lstm_best.pth')
    lstm_sd = torch.load(lstm_path, map_location=device, weights_only=True)
    lstm = TimeSeriesNet(input_size, hidden_size=64, num_layers=_detect_num_layers(lstm_sd),
                         output_size=1, model_type="LSTM", use_attention=True)
    lstm.load_state_dict(lstm_sd)
    lstm.to(device)
    lstm.eval()

    # GRU modeli — num_layers state dict'ten otomatik tespit edilir
    gru_path = os.path.join(models_dir, f'{ticker}_gru_best.pth')
    gru_sd = torch.load(gru_path, map_location=device, weights_only=True)
    gru = TimeSeriesNet(input_size, hidden_size=64, num_layers=_detect_num_layers(gru_sd),
                        output_size=1, model_type="GRU", use_attention=True)
    gru.load_state_dict(gru_sd)
    gru.to(device)
    gru.eval()
    
    # Random Forest
    rf = joblib.load(os.path.join(models_dir, f'{ticker}_rf_classifier.pkl'))
    
    # XGBoost
    xgb = joblib.load(os.path.join(models_dir, f'{ticker}_xgb_classifier.pkl'))
    
    return lstm, gru, rf, xgb


def get_predictions(X_dl, X_ml, lstm, gru, rf, xgb):
    """Tum base modellerden tahmin alir. DL cikislarina sigmoid uygulanir."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        dl_tensor = torch.FloatTensor(X_dl).to(device)
        # KRITIK: BCEWithLogitsLoss ile egitilen modellerin cikisina sigmoid uygula
        # boylece 0-1 araliginda olasilik degerlerine donusur
        lstm_preds = torch.sigmoid(lstm(dl_tensor)).cpu().numpy().flatten()
        gru_preds = torch.sigmoid(gru(dl_tensor)).cpu().numpy().flatten()
    
    # ML modelleri: sinif 1 olasiligini al (zaten 0-1 arasinda)
    rf_preds = rf.predict_proba(X_ml)[:, 1]
    xgb_preds = xgb.predict_proba(X_ml)[:, 1]
    
    return lstm_preds, gru_preds, rf_preds, xgb_preds

def _add_cross_features(meta_X):
    """
    Base model tahminleri arasinda capraz ozellikler ekler.
    meta_X shape: (N, 4) -> (N, 10)

    Yeni ozellikler (6 adet):
      - DL modellerin ortalamasi: (LSTM + GRU) / 2
      - ML modellerin ortalamasi: (RF + XGB) / 2
      - DL modeller arasi uyumsuzluk: |LSTM - GRU|
      - ML modeller arasi uyumsuzluk: |RF - XGB|
      - Tum modellerin ortalamasi
      - Modeller arasi belirsizlik (std)
    """
    lstm, gru, rf, xgb = meta_X[:, 0], meta_X[:, 1], meta_X[:, 2], meta_X[:, 3]
    cross = np.column_stack([
        (lstm + gru) / 2,           # DL modellerin ortalamasi
        (rf + xgb) / 2,             # ML modellerin ortalamasi
        np.abs(lstm - gru),          # DL modeller arasindaki uyumsuzluk
        np.abs(rf - xgb),           # ML modeller arasindaki uyumsuzluk
        np.mean(meta_X, axis=1),    # Tum modellerin ortalamasi
        np.std(meta_X, axis=1),     # Modeller arasi belirsizlik
    ])
    return np.hstack([meta_X, cross])

def main(ticker="BTC-USD"):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, 'src', 'models', 'saved_models')

    from src.data.feature_store import FeatureStore
    store = FeatureStore(ticker)

    print(f"\n{'=' * 60}")
    print(f"  {ticker} Ensemble Meta-Model (Unified FeatureStore)")
    print(f"{'=' * 60}")

    # Unified split — tum modeller ayni bolumu kullanir
    X_dl_train, y_train = store.get_lstm_split("train")
    X_ml_train, _ = store.get_xgb_split("train")
    X_dl_val, y_val = store.get_lstm_split("val")
    X_ml_val, _ = store.get_xgb_split("val")
    X_dl_test, y_test = store.get_lstm_split("test")
    X_ml_test, _ = store.get_xgb_split("test")

    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Base modelleri yukle
    lstm, gru, rf, xgb = load_models(models_dir, input_size=X_dl_train.shape[2], ticker=ticker)

    # VERI SIZINTISI AZALTMA:
    # Meta-model egitimi icin train setinin son %50'sini kullan
    # (base modeller tum train seti ile egitildi)
    mid = len(X_dl_train) // 2
    lstm_meta, gru_meta, rf_meta, xgb_meta = get_predictions(
        X_dl_train[mid:], X_ml_train[mid:], lstm, gru, rf, xgb
    )
    meta_X_train = np.column_stack((lstm_meta, gru_meta, rf_meta, xgb_meta))
    meta_X_train = _add_cross_features(meta_X_train)
    meta_y_train = y_train[mid:]

    # Val verisi uzerinde tahminler (early stopping icin)
    lstm_val, gru_val, rf_val, xgb_val = get_predictions(X_dl_val, X_ml_val, lstm, gru, rf, xgb)
    meta_X_val = np.column_stack((lstm_val, gru_val, rf_val, xgb_val))
    meta_X_val = _add_cross_features(meta_X_val)

    # Test verisi uzerinde tahminler
    lstm_test, gru_test, rf_test, xgb_test = get_predictions(X_dl_test, X_ml_test, lstm, gru, rf, xgb)
    meta_X_test = np.column_stack((lstm_test, gru_test, rf_test, xgb_test))
    meta_X_test = _add_cross_features(meta_X_test)

    # ────────── Meta-model: GradientBoosting ──────────
    print("\n--- Ensemble Meta-Model Egitiliyor (GradientBoosting, Early Stopping ile) ---")
    meta_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        min_samples_leaf=10,
        validation_fraction=0.15,
        n_iter_no_change=15,
        tol=1e-4,
        random_state=42
    )
    meta_model.fit(meta_X_train, meta_y_train)

    # Degerlendirme
    meta_preds = meta_model.predict(meta_X_test)
    meta_acc = accuracy_score(y_test, meta_preds)
    meta_f1 = f1_score(y_test, meta_preds)

    print(f"  Ensemble Accuracy: {meta_acc:.4f}")
    print(f"  Ensemble F1 Score: {meta_f1:.4f}")
    print(f"\n{classification_report(y_test, meta_preds, target_names=['DOWN', 'UP'])}")

    # Meta-modeli kaydet
    joblib.dump(meta_model, os.path.join(models_dir, f'{ticker}_ensemble_meta_model.pkl'))
    print(f"{ticker} meta-model '{models_dir}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()

