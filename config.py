# ============================================================
#  config.py  —  Proje Ayarları
# ============================================================

# ── Takip Edilecek Coinler ──────────────────────────────────
SYMBOLS = {
    "BTC": "BTCUSDT",   # Bitcoin
    "ETH": "ETHUSDT",   # Ethereum
    "SOL": "SOLUSDT",   # Solana
}

COINGECKO_IDS = {
    "BTC": "bitcoin",  
    "ETH": "ethereum",
    "SOL": "solana",
}

# ── Zaman Dilimleri ─────────────────────────────────────────
INTERVALS = ["1h", "4h", "1d"]   # Binance kline interval'ları

# ── Veritabanı ───────────────────────────────────────────────
DB_PATH = "crypto_data.db"

# ── Binance API ──────────────────────────────────────────────
BINANCE_BASE_URL  = "https://api.binance.com"
BINANCE_API_KEY   = ""   # İsteğe bağlı: public endpoint'ler için gerekmez
BINANCE_SECRET    = ""   # İsteğe bağlı: trade işlemleri için gerekir

# ── CoinGecko API ────────────────────────────────────────────
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY  = ""   # Pro plan için; boş bırakılırsa ücretsiz plan kullanılır

# ── Tarihsel Veri Ayarları ───────────────────────────────────
HISTORICAL_DAYS = 1500  # Kaç günlük geçmiş veri çekilsin (~4 yıl)
KLINE_LIMIT     = 1000  # Tek istekte max kline sayısı (Binance limiti)

# ── Zamanlayıcı ──────────────────────────────────────────────
UPDATE_INTERVAL_SECONDS = 60   # Gerçek zamanlı güncelleme sıklığı

# ── 3. Hafta: Sinyal Parametreleri ───────────────────────────
RSI_OVERSOLD   = 30     # RSI aşırı satım eşiği → BUY sinyali
RSI_OVERBOUGHT = 70     # RSI aşırı alım eşiği → SELL sinyali

SIGNAL_WEIGHTS = {       # Kombine sinyal ağırlıkları
    "rsi":  0.30,        #   RSI sinyali ağırlığı
    "macd": 0.30,        #   MACD crossover ağırlığı
    "bb":   0.20,        #   Bollinger Bands ağırlığı
    "ma":   0.20,        #   MA crossover ağırlığı
}

# ── 3. Hafta: Görselleştirme Ayarları ────────────────────────
REPORTS_DIR = "reports"  # Grafik çıktı dizini

# ── 5. Hafta: Gelişmiş Özellik Parametreleri ─────────────────
VOLATILITY_WINDOWS = [5, 10, 21]   # Rolling volatilite pencere boyutları
VOLUME_MA_PERIOD   = 20            # Hacim hareketli ortalama periyodu
PATTERN_LOOKBACK   = 5             # Pattern tespiti bakış periyodu

# ── Dashboard Işlenmiş Veri Özeti Ayarları ────────────────────
DISPLAY_DAYS       = 1500          # Dashboard'da gösterilecek gün miktarı (veya len(df))

# ── 6-11. Hafta: Model Ayarları ──────────────────────────────
MODEL_DIR = "models"
RL_MODEL_DIR = "rl_models"

