# ============================================================
#  dl_forecaster.py  —  Derin Öğrenme Fiyat Tahmin Modülü
#  LSTM, Bi-LSTM ve GRU modellerinin kurulumu ve eğitimi
#
#  [DEPRECATED] Bu dosya 1-5. hafta prototip mimarisine aittir.
#  Framework: TensorFlow/Keras | Görev: Regresyon (fiyat tahmini)
#
#  Aktif mimari için: src/models/time_series_models.py
#  (Framework: PyTorch | Görev: Binary sınıflandırma | Attention + Focal Loss)
# ============================================================

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from config import MODEL_DIR
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Bidirectional, GRU, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    tf = None

logger = logging.getLogger(__name__)

class DeepLearningForecaster:
    def __init__(self, sequence_length: int = 60, feature_size: int = 1, model_type: str = "lstm"):
        """
        Derin Öğrenme tabanlı fiyat tahminci sınıfı.

        Args:
            sequence_length: Zaman serisi pencere boyutu (örn: 60 periyot)
            feature_size: Girdi değişkeni sayısı
            model_type: 'lstm', 'bilstm' veya 'gru'
        """
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.model_type = model_type.lower()
        self.model = None

        if tf is None:
            logger.error("⚠️ TensorFlow kurulu değil! Lütfen 'pip install tensorflow' çalıştırın.")
        else:
            self._build_model()

    def _build_model(self):
        """Seçilen model tipine göre (LSTM/Bi-LSTM/GRU) ağı oluşturur."""
        if tf is None:
            return

        self.model = Sequential()
        self.model.add(Input(shape=(self.sequence_length, self.feature_size)))

        if self.model_type == "lstm":
            self.model.add(LSTM(units=64, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=64))
            self.model.add(Dropout(0.2))

        elif self.model_type == "bilstm":
            self.model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
            self.model.add(Dropout(0.2))
            self.model.add(Bidirectional(LSTM(units=32)))
            self.model.add(Dropout(0.2))

        elif self.model_type == "gru":
            self.model.add(GRU(units=64, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(GRU(units=64))
            self.model.add(Dropout(0.2))
        else:
            raise ValueError(f"Geçersiz model_type: {self.model_type}. Sadece 'lstm', 'bilstm', 'gru' desteklenir.")

        # Çıktı katmanı (Bir sonraki periyodun fiyatını tahmin etmek için 1 nöron)
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        logger.info(f"✅ {self.model_type.upper()} modeli derlendi.")

    @staticmethod
    def prepare_sequences(data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        zaman serisi verisini model eğitimi için (X, y) formatına çevirir.
        """
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(data[i, 0])  # İlk sütunun (clos fiyatı/getiri varsayılır) hedeflenmesi
        return np.array(X), np.array(y)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int = 50, batch_size: int = 32, symbol: str = "UNKNOWN"):
        """Modeli eğitir ve en iyi ağırlıkları kaydeder."""
        if self.model is None:
            raise RuntimeError("Model oluşturulamadı.")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{symbol}_{self.model_type}_best.keras")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
        ]

        logger.info(f"🚀 {self.model_type.upper()} egitimi başlıyor... ({epochs} epoch)")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        logger.info(f"✅ {self.model_type.upper()} egitimi tamamlandı.")
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Eğitilmiş model ile tahmin yapar."""
        if self.model is None:
            raise RuntimeError("Model oluşturulamadı veya yüklenmedi.")
        return self.model.predict(X, verbose=0)

    def load(self, model_path: str):
        """Eğitilmiş bir modeli diskten yükler."""
        if tf is None:
            return
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            logger.info(f"📂 Model yüklendi: {model_path}")
        else:
            logger.error(f"⚠️ Model dosyası bulunamadı: {model_path}")

