# ============================================================
#  ml_signal_generator.py  —  Makine Öğrenmesi Sinyal Modülü
#  XGBoost ve Random Forest ile sinyal üretimi
#
#  [DEPRECATED] Bu dosya 1-5. hafta prototip mimarisine aittir.
#  Görev: 3-sınıflı sinyal (-1/0/1) | metric: mlogloss
#
#  Aktif mimari için: src/models/ml_classification_models.py
#  (Görev: Binary sınıflandırma | TimeSeriesSplit CV | Hyperparameter tuning)
# ============================================================

import logging
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from config import MODEL_DIR

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import xgboost as xgb
    import joblib
except ImportError:
    xgb = None
    RandomForestClassifier = None

logger = logging.getLogger(__name__)

class MLSignalGenerator:
    def __init__(self, model_type: str = "xgboost"):
        """
        Geleneksel Makine Öğrenmesi ile Sinyal Üretimi
        
        Args:
            model_type: 'xgboost' veya 'random_forest'
        """
        self.model_type = model_type.lower()
        self.model = None

        if xgb is None or RandomForestClassifier is None:
            logger.error("⚠️ xgboost veya scikit-learn kurulu değil! Lütfen bağımlılıkları yükleyin.")
        else:
            self._build_model()

    def _build_model(self):
        """Seçilen modele göre (XGBoost/RF) objeyi oluşturur."""
        if xgb is None:
            return

        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Geçersiz model_type: {self.model_type}. Sadece 'xgboost', 'random_forest' desteklenir.")

        logger.info(f"✅ {self.model_type.upper()} modeli oluşturuldu.")

    @staticmethod
    def prepare_data(df: pd.DataFrame, target_col: str = "close", lookforward: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Geçmiş verilere bakarak gelecekteki fiyat yönünü (-1: Sat, 0: Tut, 1: Al) hedefler.
        """
        df = df.copy()
        
        # Gelecekteki fiyat değişimi
        future_price = df[target_col].shift(-lookforward)
        price_change_pct = (future_price - df[target_col]) / df[target_col] * 100
        
        # Sınıfları belirleme (Al: > %0.1, Sat: < -0.1%, Tut: Arası)
        conditions = [
            price_change_pct > 0.1,
            price_change_pct < -0.1
        ]
        choices = [1, -1] # 1: BUY, -1: SELL
        
        df['target'] = np.select(conditions, choices, default=0) # 0: HOLD
        
        # Shift nedeniyle oluşan NaN'ları temizle
        df = df.dropna()
        
        y = df.pop('target')
        # Zamana bağlı olmayan özellikleri seç (Eğitim için target ve open_time kullanılmaz)
        X = df.select_dtypes(include=[np.number])
        
        return X, y

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, symbol: str = "UNKNOWN"):
        """Modeli eğitir ve kaydeder."""
        if self.model is None:
            raise RuntimeError("Model oluşturulamadı.")

        # Sınıfların XGBoost'un beklediği (0, 1, 2) formatına getirilmesi
        # Sadece etiketler {-1, 0, 1} ise dönüştür; {0, 1} binary ise dönüştürme
        if y_train.min() == -1:
            y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        else:
            y_train_mapped = y_train

        logger.info(f"🚀 {self.model_type.upper()} eğitimi başlıyor...")
        self.model.fit(X_train, y_train_mapped)
        logger.info(f"✅ {self.model_type.upper()} eğitimi tamamlandı.")
        
        # Modeli kaydet (Joblib ile)
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{symbol}_{self.model_type}_model.pkl")
        joblib.dump(self.model, model_path)
        logger.info(f"💾 Model kaydedildi: {model_path}")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Model performansını değerlendirir."""
        if self.model is None:
            raise RuntimeError("Model oluşturulamadı veya yüklenmedi.")
        
        if y_test.min() == -1:
            y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2})
            target_names = ["SELL (-1)", "HOLD (0)", "BUY (1)"]
        else:
            y_test_mapped = y_test
            target_names = ["DOWN (0)", "UP (1)"]
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test_mapped, y_pred)
        report = classification_report(y_test_mapped, y_pred, target_names=target_names, output_dict=True)
        
        logger.info(f"📊 Accuracy: {acc:.4f}")
        return {"accuracy": acc, "report": report}

    def predict_signals(self, X: pd.DataFrame) -> np.ndarray:
        """Yeni veriler üzerinden sinyal (-1, 0, 1) tahmini yapar."""
        if self.model is None:
            raise RuntimeError("Model yüklenmedi.")
        
        y_pred_mapped = self.model.predict(X)
        
        # Geri eşleştirme (0 -> -1, 1 -> 0, 2 -> 1)
        reverse_map = {0: -1, 1: 0, 2: 1}
        return np.array([reverse_map.get(k, 0) for k in y_pred_mapped])

    def load(self, model_path: str):
        """Joblib ile kaydedilmiş modeli yükler."""
        if os.path.exists(model_path) and joblib is not None:
            self.model = joblib.load(model_path)
            logger.info(f"📂 Model yüklendi: {model_path}")
        else:
            logger.error(f"⚠️ Model dosyası bulunamadı veya joblib eksik: {model_path}")

