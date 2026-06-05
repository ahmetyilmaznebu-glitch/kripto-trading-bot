"""
ML veri seti ayarlari — LSTM ve XGBoost icin ortak sabitler.
"""
import os

# Proje kok dizini (src/data -> src -> proje)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Veri dizinleri
DATA_ML_DIR = os.path.join(PROJECT_ROOT, "data", "ml")
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Coin listesi (AI pipeline formatı)
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]

TICKER_TO_BINANCE = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
}

# Zaman serisi parametreleri
# DEĞİŞTİRİLDİ (2026-06-05): LSTM veri yetersizliğini çözmek için
# Eski: WINDOW_SIZE=60, PURGE_GAP=60 → Train ~92 örnek (yetersiz)
# Yeni: WINDOW_SIZE=30, PURGE_GAP=14 → Train ~1200+ örnek (yeterli)
WINDOW_SIZE = 30
PURGE_GAP = 14

# Egitim bolme oranlari (pencere sayisi uzerinden; time_series_models ile uyumlu)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Scaler yalnizca bu orandaki *gunluk satirlara* fit edilir (veri sizintisi onlemi)
SCALER_FIT_RATIO = 0.70
TEST_RATIO = 0.15

# Dead-zone esigi: abs(next_return) <= DIRECTION_THRESHOLD olan ornekler cikarilir
# Bu, %0.15'ten kucuk fiyat hareketlerinin gurultu olarak kabul edilmesini saglar.
DIRECTION_THRESHOLD = 0.0015

# Binance gunluk veri
BINANCE_INTERVAL = "1d"
# DEĞİŞTİRİLDİ (2026-06-05): Maksimum geçmiş veri için artırıldı
# BTC/ETH ~3215 gün (2017'den beri), SOL ~2126 gün (2020'den beri)
FETCH_DAYS = 3500

# Hedef: ertesi gun kapanis yukari mi?
TARGET_COLUMN = "direction"

# Model girdisi olarak kullanilacak ozellikler (duragan / oransal)
FEATURE_COLUMNS = [
    "log_return_1d",
    "high_low_range",
    "close_open_return",
    "volume_change",
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "bb_position",
    "bb_width",
    "sma_ratio_20",
    "ema_ratio_50",
    "atr_pct",
    "adx_14",
    "stoch_k",
    "obv_change",
    "return_lag_5",
    "return_lag_10",
    "return_lag_20",
    "volatility_20",
    "momentum_10",
]

# Dashboard ve grafikler icin saklanan ham fiyat sutunu
PRICE_COLUMN = "close"
