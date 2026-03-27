import pandas as pd
import numpy as np
import ta
import os
import sys
import joblib
import requests
import time
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# SSL uyarisini kapat (universite proxy bypass icin)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')


# ─────────────────────────────────────────────────────────────
#  Veri Cekme: yfinance + CoinGecko Fallback
# ─────────────────────────────────────────────────────────────

def fetch_data_yfinance(ticker="BTC-USD", start="2020-01-01", end="2025-01-01"):
    """yfinance ile veri cekmeyi dener. Yahoo Finance API sorunlarına karşı korumalı."""
    try:
        import yfinance as yf
        print(f"[yfinance] {ticker} verisi {start} - {end} icin cekiliyor...")
        
        # User-Agent ekle (Yahoo Finance bot tespiti önlemek için)
        # ve timeout ayarla (bağlantı sorunlarına karşı)
        df = yf.download(
            ticker, 
            start=start, 
            end=end, 
            interval="1d", 
            progress=False,
            ignore_tz=True,
            threads=False
        )
        
        if df is None or df.empty:
            print("[yfinance] Veri bos dondu.")
            return None
        
        # Multi-level column kontrolu
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Sutun adlarini standardize et
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Gerekli sutunlari kontrol et
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"[yfinance] Eksik sutun. Mevcut: {list(df.columns)}")
            return None
        
        df.dropna(inplace=True)
        if len(df) < 100:
            print(f"[yfinance] Yetersiz veri: sadece {len(df)} satir.")
            return None
        
        print(f"[yfinance] Basarili! {len(df)} satir cekildi.")
        return df
    except Exception as e:
        print(f"[yfinance] Hata: {type(e).__name__}: {e}")
        # Detaylı hata bilgisi hiçbir şey tutmuyoruz, fallback'e geçiyoruz
        return None


def fetch_data_pandas_datareader(ticker="BTC-USD", start="2021-01-01", end="2025-06-01"):
    """pandas_datareader ile Yahoo Finance verisini çeker (yfinance'ın alternatifi)."""
    try:
        from pandas_datareader import data as web
        print(f"[pandas_datareader] {ticker} verisi {start} - {end} icin cekiliyor...")
        
        df = web.DataReader(ticker, 'yahoo', start=start, end=end)
        
        if df is None or df.empty:
            print("[pandas_datareader] Veri bos dondu.")
            return None
        
        # Sutun adlarini standardize et
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Gerekli sutunlari kontrol et
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"[pandas_datareader] Eksik sutun. Mevcut: {list(df.columns)}")
            return None
        
        df.dropna(inplace=True)
        if len(df) < 100:
            print(f"[pandas_datareader] Yetersiz veri: sadece {len(df)} satir.")
            return None
        
        print(f"[pandas_datareader] Basarili! {len(df)} satir cekildi.")
        return df
    except Exception as e:
        print(f"[pandas_datareader] Hata: {type(e).__name__}: {e}")
        return None


def fetch_data_binance(symbol="BTCUSDT", interval="1d", days=2000):
    """Binance API ile kripto para verisini çeker. Önce connector, başarısız olursa REST API (SSL bypass)."""
    
    # --- Yontem A: binance-connector kutuphanesi ---
    try:
        print(f"[Binance] {symbol} verisi son {days} gun icin cekiliyor...")
        from binance.client import Client
        client = Client(api_key="", api_secret="", testnet=False)
        klines = client.get_historical_klines(symbol, interval, f"{days} days ago UTC")
        if klines and len(klines) >= 100:
            df = _klines_to_dataframe(klines)
            print(f"[Binance] Basarili! {len(df)} satir cekildi.")
            return df
    except Exception as e:
        print(f"[Binance-connector] Hata: {type(e).__name__}: {e}")
    
    # --- Yontem B: Dogrudan REST API (SSL bypass) ---
    try:
        print(f"[Binance REST] SSL bypass ile {symbol} verisi cekiliyor...")
        all_klines = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        limit = 1000
        
        current = start_time
        page = 0
        while current < end_time:
            page += 1
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_time,
                "limit": limit
            }
            resp = requests.get(url, params=params, timeout=30, verify=False)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_klines.extend(data)
            print(f"  Sayfa {page}: {len(data)} kayit alindi (toplam: {len(all_klines)})")
            current = data[-1][6] + 1  # Close Time + 1ms
            if len(data) < limit:
                break
            time.sleep(0.3)  # Rate limit
        
        if not all_klines:
            print("[Binance REST] Veri bos dondu.")
            return None
        
        df = _klines_to_dataframe(all_klines)
        print(f"[Binance REST] Basarili! {len(df)} satir cekildi (SSL bypass).")
        return df
    except Exception as e:
        print(f"[Binance REST] Hata: {type(e).__name__}: {e}")
        return None


def _klines_to_dataframe(klines):
    """Binance klines listesini pandas DataFrame'e donusturur."""
    df = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Volume', 'Trades', 'Taker Buy Base',
        'Taker Buy Quote', 'Ignore'
    ])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df.set_index('Open Time', inplace=True)
    df.drop(columns=['Close Time', 'Ignore'], inplace=True)
    df.dropna(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df


def fetch_data_coingecko(ticker="BTC-USD", days=2000):
    """CoinGecko ucretsiz API uzerinden kripto verisi ceker."""
    coin_map = {
        "BTC-USD": "bitcoin",
        "ETH-USD": "ethereum",
        "SOL-USD": "solana",
        "BNB-USD": "binancecoin",
    }
    coin_id = coin_map.get(ticker, "bitcoin")
    print(f"[CoinGecko] {ticker} ({coin_id}) verisi son {days} gun icin cekiliyor...")
    
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    
    # User-Agent ekle (bazı API'ler bunu gerekli kılıyor)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
        
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        
        if not prices:
            print("[CoinGecko] Fiyat verisi bos.")
            return None
        
        df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.set_index("Date", inplace=True)
        
        # Hacim ekle
        if volumes and len(volumes) == len(prices):
            df["Volume"] = [v[1] for v in volumes]
        else:
            df["Volume"] = 0
        
        # CoinGecko OHLC verisi vermiyor; kapanistan deterministik turetme
        # Rastgele deger yerine rolling window ile gercekci High/Low uretimi
        df["Open"] = df["Close"].shift(1)
        # 3 gunluk rolling max/min ile gercekci High/Low (random yerine)
        df["High"] = df["Close"].rolling(window=3, min_periods=1).max()
        df["Low"]  = df["Close"].rolling(window=3, min_periods=1).min()
        # High ve Low'un Close'dan mantikli olmasini garanti et
        df["High"] = df[["High", "Close"]].max(axis=1)
        df["Low"]  = df[["Low", "Close"]].min(axis=1)
        
        df.drop(columns=["Timestamp"], inplace=True)
        df.dropna(inplace=True)
        
        # Sutunlari yfinance formatina uyumlu sirala
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        
        print(f"[CoinGecko] Basarili! {len(df)} satir cekildi.")
        return df
    except Exception as e:
        print(f"[CoinGecko] Hata: {e}")
        return None

def generate_synthetic_btc(days=1400):
    """Internet yoksa gercekci sentetik BTC verisi uretir (son carem)."""
    print(f"[Sentetik] {days} gunluk yapay BTC-USD verisi uretiliyor...")
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    
    # Geometric Brownian Motion ile gercekci fiyat serisi
    price = 10000  # Baslangic fiyati
    prices = [price]
    for _ in range(days - 1):
        daily_return = np.random.normal(0.0005, 0.035)  # ortalama %0.05 getiri, %3.5 volatilite
        price *= (1 + daily_return)
        price = max(price, 1000)  # 1000 altina dusmesin
        prices.append(price)
    
    closes = np.array(prices)
    highs = closes * (1 + np.abs(np.random.normal(0.01, 0.015, days)))
    lows  = closes * (1 - np.abs(np.random.normal(0.01, 0.015, days)))
    opens = closes * (1 + np.random.normal(0, 0.008, days))
    volumes = np.random.uniform(1e9, 5e10, days)
    
    df = pd.DataFrame({
        'Open': opens, 'High': highs, 'Low': lows,
        'Close': closes, 'Volume': volumes
    }, index=dates)
    df.index.name = 'Date'
    
    print(f"[Sentetik] Basarili! {len(df)} satir uretildi.")
    return df


def _fetch_from_database(ticker="BTC-USD"):
    """Yerel SQLite veritabanindan kline verisi okur (en hizli kaynak)."""
    try:
        import sqlite3
        # Proje kokunden database modulunu import et
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        from config import DB_PATH
        
        if not os.path.exists(DB_PATH):
            print("[Yerel DB] Veritabani dosyasi bulunamadi.")
            return None
        
        # Ticker'i Binance formatina cevir
        symbol_map = {
            "BTC-USD": "BTCUSDT",
            "ETH-USD": "ETHUSDT", 
            "SOL-USD": "SOLUSDT",
            "BNB-USD": "BNBUSDT",
        }
        binance_symbol = symbol_map.get(ticker, ticker.replace("-USD", "USDT"))
        
        conn = sqlite3.connect(DB_PATH)
        
        # Oncelikle 1d (gunluk) verisini dene
        count = conn.execute(
            "SELECT COUNT(*) FROM klines WHERE symbol=? AND interval=?",
            (binance_symbol, "1d")
        ).fetchone()[0]
        
        if count >= 100:
            interval = "1d"
            print(f"[Yerel DB] {binance_symbol} gunluk veri bulundu: {count} kayit")
        else:
            # Gunluk yoksa 4 saatlik veriyi kullan (OHLCV resample ile)
            count_4h = conn.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol=? AND interval=?",
                (binance_symbol, "4h")
            ).fetchone()[0]
            if count_4h >= 100:
                interval = "4h"
                print(f"[Yerel DB] {binance_symbol} 4 saatlik veri bulundu: {count_4h} kayit")
            else:
                # 1 saatlik veriyi kullan
                count_1h = conn.execute(
                    "SELECT COUNT(*) FROM klines WHERE symbol=? AND interval=?",
                    (binance_symbol, "1h")
                ).fetchone()[0]
                if count_1h >= 100:
                    interval = "1h"
                    print(f"[Yerel DB] {binance_symbol} saatlik veri bulundu: {count_1h} kayit")
                else:
                    print(f"[Yerel DB] {binance_symbol} icin yeterli veri yok.")
                    conn.close()
                    return None
        
        # Veriyi cek
        query = """
            SELECT open_time_dt, open, high, low, close, volume
            FROM klines 
            WHERE symbol=? AND interval=?
            ORDER BY open_time ASC
        """
        rows = conn.execute(query, (binance_symbol, interval)).fetchall()
        conn.close()
        
        if not rows:
            return None
        
        # DataFrame'e donustur
        df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Sayisal turler
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        
        # Eger saatlik veya 4 saatlik ise, gunluk veriye resample et
        if interval in ("1h", "4h"):
            print(f"[Yerel DB] {interval} veri gunluk (1D) periyoda resample ediliyor...")
            df = df.resample('1D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        print(f"[Yerel DB] Basarili! {len(df)} gunluk kayit okundu.")
        return df
        
    except Exception as e:
        print(f"[Yerel DB] Hata: {type(e).__name__}: {e}")
        return None


def fetch_data(ticker="BTC-USD"):
    """Veri kaynagi fallback sistemi: Yerel DB -> Binance REST -> CoinGecko -> CSV -> Sentetik."""
    
    # config'den hedef gun sayisini al
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        from config import HISTORICAL_DAYS
        min_days = HISTORICAL_DAYS
    except Exception:
        min_days = 350
    
    # Yontem 0: Yerel SQLite veritabani (en hizli ve en guvenilir)
    print("\n🔄 Veri kaynağı denemeleri başlıyor...\n")
    df = _fetch_from_database(ticker)
    if df is not None and len(df) >= min_days:
        return df
    elif df is not None:
        print(f"  ℹ️  Yerel DB'de {len(df)} kayit var ama hedef {min_days}. API'lerden daha fazla veri aranıyor...")
    
    # Yontem 1: Binance API (kripto için en güvenilir)
    print("\n⚠️  Yerel DB yetersiz, Binance API deneniyor...\n")
    if "USDT" in ticker or ticker in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        binance_symbol = ticker.replace("-USD", "USDT")
        df = fetch_data_binance(binance_symbol, days=2000)
        if df is not None and len(df) >= 100:
            return df
    
    # Yontem 2: pandas_datareader
    print("\n⚠️  Binance basarisiz, pandas_datareader deneniyor...\n")
    df = fetch_data_pandas_datareader(ticker)
    if df is not None and len(df) >= 100:
        return df
    
    # Yontem 3: yfinance (eski ama hala denebilir)
    print("\n⚠️  Binance basarisiz, yfinance'a geciliyor...\n")
    df = fetch_data_yfinance(ticker)
    if df is not None and len(df) >= 100:
        return df
    
    # Yontem 4: CoinGecko (kripto için ücretsiz API — coin bazlı)
    print("\n⚠️  yfinance basarisiz oldu, CoinGecko API'ye geciliyor...\n")
    df = fetch_data_coingecko(ticker=ticker, days=2000)
    if df is not None and len(df) >= 100:
        return df
    
    # Yontem 5: Onceden indirilmis CSV varsa onu kullan
    raw_csv = os.path.join(DATA_DIR, 'raw', f'{ticker}_raw.csv')
    if os.path.exists(raw_csv):
        print(f"\n📂 Mevcut veri bulundu: {raw_csv}")
        df = pd.read_csv(raw_csv, index_col=0)
        if len(df) >= 100:
            print(f"   {len(df)} satir yuklendi (onceden indirilmis veri).")
            return df
    
    # Yontem 4: Sentetik veri uret (son carem)
    print("\n⚠️  API'ler erisilemiyor. Sentetik veri uretiliyor...\n")
    return generate_synthetic_btc(days=1400)


# ─────────────────────────────────────────────────────────────
#  Teknik Indikatorler
# ─────────────────────────────────────────────────────────────

def add_technical_indicators(df):
    print("Teknik indikatorler hesaplaniyor (temel + gelişmiş özellikler)...")
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # ── Temel İndikatörler ──
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    # MACD
    macd = ta.trend.MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    # Moving Averages
    df['SMA_20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    
    # ── Getiri Tabanlı Özellikler (DL modeller için kritik) ──
    # Log-return: fiyat seviyesi yerine değişim oranı — LSTM'in en iyi anlayacağı format
    df['Log_Return'] = np.log(close / close.shift(1))
    # Çok periyotlu log-return (kısa/orta/uzun vadeli trend)
    df['Return_Lag_5'] = np.log(close / close.shift(5))
    df['Return_Lag_10'] = np.log(close / close.shift(10))
    df['Return_Lag_20'] = np.log(close / close.shift(20))
    
    # ── Günlük getiri ve volatilite/momentum ──
    df['Daily_Return'] = close.pct_change()
    df['Volatility_20'] = close.rolling(window=20).std() / close.rolling(window=20).mean()
    df['Momentum_10'] = close / close.shift(10) - 1
    df['Volume_Change'] = volume.pct_change()
    
    # ── Ek Teknik Göstergeler ──
    # ATR: Average True Range (volatilite ölçüsü)
    df['ATR'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    # ADX: Trend gücü göstergesi
    df['ADX'] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
    # Stochastic Oscillator %K: Aşırı alım/satım
    df['Stoch_K'] = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14).stoch()
    # OBV değişim oranı: Hacim-fiyat uyumu
    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df['OBV_Change'] = obv.pct_change()
    
    # ── Fiyat-Ortalama Oranları (mean-reversion sinyali) ──
    df['Price_To_SMA20'] = close / df['SMA_20']
    df['Price_To_EMA50'] = close / df['EMA_50']
    
    # ── Hedef Değişken ──
    df['Target_Close'] = df['Close'].shift(-1)
    
    # KRITIK: Binary yon etiketini OLCEKLEME ONCESI olustur (gercek fiyat karsilastirmasi)
    df['Direction'] = (df['Target_Close'] > df['Close']).astype(int)
    
    # Inf ve NaN temizle
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print(f"  Toplam özellik sayısı: {len(df.columns) - 2} (Target_Close ve Direction hariç)")
    return df


# ─────────────────────────────────────────────────────────────
#  Olceklendirme & Sliding Window
# ─────────────────────────────────────────────────────────────

def scale_data(df, feature_cols, target_col, train_ratio=0.8, ticker="BTC-USD"):
    print("Ozellikler ve hedef degisken Min-Max yontemi ile (0, 1) arasina olcekleniyor...")
    print(f"   ⚠️  Scaler sadece egitim verisi uzerinde fit ediliyor (oran: {train_ratio})")
    
    if df.empty or len(df) == 0:
        print("❌ Olceklenecek veri bos! Islem iptal edildi.")
        return None
    
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    df_scaled = df.copy()
    
    # KRITIK: Scaler'i sadece egitim verisi uzerinde fit et (veri sizintisi onlemi)
    train_size = int(len(df) * train_ratio)
    feature_scaler.fit(df[feature_cols].iloc[:train_size])
    target_scaler.fit(df[[target_col]].iloc[:train_size])
    
    # Tum veriye transform uygula
    df_scaled[feature_cols] = feature_scaler.transform(df[feature_cols])
    df_scaled[[target_col]] = target_scaler.transform(df[[target_col]])
    
    proc_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    
    joblib.dump(feature_scaler, os.path.join(proc_dir, f'{ticker}_feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(proc_dir, f'{ticker}_target_scaler.pkl'))
    
    return df_scaled


def create_sliding_windows(data, window_size, feature_cols=None, normalize_windows=True):
    """
    Sliding window dizileri olusturur.

    normalize_windows=True (varsayilan):
        Fiyat seviyesi iceren sutunlar (Open, High, Low, Close, Volume,
        BB_High, BB_Low, BB_Mid, SMA_20, EMA_50, ATR) her pencere icinde
        kendi ilk degerine gore yuzde-degisim olarak normalize edilir:
            X[t, j] = X[t, j] / X[0, j] - 1.0

        Boylece LSTM mutlak fiyat seviyesi degil HAREKET gorur.
        RSI, MACD, Log_Return gibi zaten duragan ozellikler dokunulmadan
        MinMaxScaler ciktisi olarak kalir.

        Neden gerekli:
        - MinMaxScaler egitim verisine fit edilir (orn. BTC $5K-$50K)
        - Test setinde BTC $80K+ → olcekli deger > 1.0 (dagılım dışı)
        - LSTM bu durumda 'hep UP tahmin et' minimine sıkışır
        - Pencere-ici normalizasyon bu sorunu tamamen ortadan kaldirir
    """
    # Fiyat seviyesi ozellikleri — pencere bazlı normalize edilecek
    # NOT: Volume ve ATR dahil edilmez — bu sutunlar MinMaxScaler sonrasi
    # zaman zaman 0'a cok yakin referans deger alir. Bolme islemi yapilinca
    # 500-25000 gibi asiri buyuk degerler olusur ve LSTM gradyanlarini bozar.
    # Volume zaten Volume_Change (pct_change) olarak, ATR de Log_Return /
    # Volatility_20 ile temsil edilmektedir.
    PRICE_LEVEL_COLS = {
        'Open', 'High', 'Low', 'Close',
        'BB_High', 'BB_Low', 'BB_Mid', 'SMA_20', 'EMA_50'
    }

    print(f"Sliding windows ({window_size} adimlik) olusturuluyor"
          f"{' [pencere-ici normalizasyon ACIK]' if normalize_windows else ''}...")
    X, y = [], []

    if feature_cols is not None:
        features = data[feature_cols].values
        col_names = list(feature_cols)
    else:
        drop_cols = ['Target_Close', 'Direction']
        existing_drop = [c for c in drop_cols if c in data.columns]
        feat_df = data.drop(columns=existing_drop)
        features = feat_df.values
        col_names = list(feat_df.columns)

    # Hangi indekslerin fiyat seviyesi oldugunu tespit et
    price_indices = [i for i, c in enumerate(col_names) if c in PRICE_LEVEL_COLS]

    if 'Direction' in data.columns:
        targets = data['Direction'].values
    else:
        targets = data['Target_Close'].values

    for i in range(len(data) - window_size):
        window = features[i:i + window_size].copy()

        if normalize_windows and price_indices:
            for j in price_indices:
                ref = window[0, j]
                if ref != 0.0:
                    window[:, j] = window[:, j] / ref - 1.0
                    # Asiri buyuk/kucuk degerleri kisit: [-3, 3] araligi
                    # kripto icin 3 sigma siniri yeterli (300% kazanc/kayip)
                    window[:, j] = np.clip(window[:, j], -3.0, 3.0)
                # ref == 0 ise sutunu oldugu gibi birak (NaN/Inf riski yok)

        X.append(window)
        y.append(targets[i + window_size - 1])

    if normalize_windows:
        print(f"  Normalize edilen sutunlar ({len(price_indices)}): "
              f"{[col_names[i] for i in price_indices]}")

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────
#  Ana Akis
# ─────────────────────────────────────────────────────────────

def main(ticker="BTC-USD"):
    print(f"\n{'='*60}")
    print(f"  📊 Veri Pipeline: {ticker}")
    print(f"{'='*60}")
    
    # 1. Veri Cekme
    raw_df = fetch_data(ticker=ticker)
    
    if raw_df is None or raw_df.empty:
        print(f"\n❌ {ticker} verisi cekilemedi. Pipeline durduruluyor.")
        return False
    
    print(f"\n✅ {ticker}: Toplam {len(raw_df)} satir ham veri elde edildi.\n")
    
    raw_dir = os.path.join(DATA_DIR, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    raw_df.to_csv(os.path.join(raw_dir, f'{ticker}_raw.csv'))
    
    # 2. Teknik Indikatorlerin Eklenmesi
    processed_df = add_technical_indicators(raw_df)
    
    if processed_df.empty:
        print(f"\n❌ {ticker}: Teknik indikator hesaplamasi sonrasi veri bos kaldi.")
        return False
    
    print(f"✅ {ticker}: Indikator hesaplamasi sonrasi {len(processed_df)} satir kaldi.\n")
    
    # 3. Olceklendirme (Scaling)
    # Genişletilmiş özellik seti (27 özellik)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD',
                       'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid', 'SMA_20', 'EMA_50',
                       'Log_Return', 'Return_Lag_5', 'Return_Lag_10', 'Return_Lag_20',
                       'Daily_Return', 'Volatility_20', 'Momentum_10', 'Volume_Change',
                       'ATR', 'ADX', 'Stoch_K', 'OBV_Change',
                       'Price_To_SMA20', 'Price_To_EMA50']
    
    # Sadece mevcut sutunlari olcekle (eger bazi sutunlar yoksa hata vermesin)
    feature_columns = [c for c in feature_columns if c in processed_df.columns]
    
    scaled_df = scale_data(processed_df, feature_columns, 'Target_Close', ticker=ticker)
    
    if scaled_df is None or scaled_df.empty:
        print(f"\n❌ {ticker}: Olcekleme basarisiz.")
        return False
    
    proc_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    scaled_df.to_csv(os.path.join(proc_dir, f'{ticker}_processed_scaled.csv'))
    
    # 4. Sliding Windows Olusturma
    # DUZELTME (Hata #2): Sadece feature_columns kullaniliyor (DL/ML uyumu)
    X, y = create_sliding_windows(scaled_df, window_size=60, feature_cols=feature_columns)
    
    if len(X) == 0:
        print(f"\n❌ {ticker}: Yeterli veri yok (60 gunluk pencere icin en az 61 satir gerekli).")
        return False
    
    # Veriyi float32'ye dönüştür
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    np.save(os.path.join(proc_dir, f'{ticker}_X_windows.npy'), X)
    np.save(os.path.join(proc_dir, f'{ticker}_y_targets.npy'), y)
    
    print(f"\n🎉 {ticker}: Veri isleme tamamlandi! X shape: {X.shape}, y shape: {y.shape}")
    return True


if __name__ == "__main__":
    main()
