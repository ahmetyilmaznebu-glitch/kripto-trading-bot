# ============================================================
#  indicators.py  —  Teknik Göstergeler Modülü
#  Kripto Para Trading Botu | Ahmet Yılmaz | 2. Hafta
#
#  Hesaplanan Göstergeler:
#    SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP
#  Tüm hesaplamalar saf pandas + numpy ile yapılır.
# ============================================================

import logging
import numpy as np
import pandas as pd

# Modül seviyesi loglama için logger oluşturma
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Hareketli Ortalamalar (Moving Averages)
# ─────────────────────────────────────────────────────────────
# Hareketli ortalamalar, fiyat trendini belirlemede ve gürültü
# filtrelemesinde kullanılan temel göstergelerdir.
# ─────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Simple Moving Average (Basit Hareketli Ortalama).
    Son 'period' kapanış fiyatının aritmetik ortalaması.
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Exponential Moving Average (Üstel Hareketli Ortalama).
    Yakın zamana daha fazla ağırlık verir.
    """
    return series.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────
#  RSI — Relative Strength Index
# ─────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Göreceli Güç Endeksi).
    0-100 aralığında bir osilatör:
      > 70 → Aşırı alım (overbought)
      < 30 → Aşırı satım (oversold)

    Hesaplama:
      delta  = fiyat değişimi
      gain   = pozitif değişimler ortalaması (EMA)
      loss   = negatif değişimlerin mutlak ortalaması (EMA)
      RS     = gain / loss
      RSI    = 100 - (100 / (1 + RS))
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder'ın yumuşatma yöntemi (EMA, alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    # Sıfıra bölme durumunda NaN olabilir, 50 ile doldur
    rsi_values = rsi_values.replace([np.inf, -np.inf], np.nan)

    return rsi_values


# ─────────────────────────────────────────────────────────────
#  MACD — Moving Average Convergence Divergence
# ─────────────────────────────────────────────────────────────

def macd(series: pd.Series,
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9) -> pd.DataFrame:
    """
    MACD (Hareketli Ortalama Yakınsama/Iraksama).

    Bileşenler:
      macd_line   = EMA(fast) - EMA(slow)
      signal_line = EMA(macd_line, signal_period)
      histogram   = macd_line - signal_line

    Alım sinyali  : macd_line > signal_line (histogram > 0)
    Satım sinyali : macd_line < signal_line (histogram < 0)
    """
    ema_fast = ema(series, fast_period)
    ema_slow = ema(series, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd_line":   macd_line,
        "macd_signal": signal_line,
        "macd_hist":   histogram,
    })


# ─────────────────────────────────────────────────────────────
#  Bollinger Bands (Bollinger Bantları)
# ─────────────────────────────────────────────────────────────

def bollinger_bands(series: pd.Series,
                    period: int = 20,
                    num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bantları.
      middle = SMA(period)
      upper  = middle + num_std × std
      lower  = middle - num_std × std

    Fiyat üst banda yakınsa → Aşırı alım bölgesi
    Fiyat alt banda yakınsa → Aşırı satım bölgesi
    Bandın daralması → Volatilite düşük, kırılım beklentisi
    """
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()

    upper = middle + (rolling_std * num_std)
    lower = middle - (rolling_std * num_std)

    return pd.DataFrame({
        "bb_upper":  upper,
        "bb_middle": middle,
        "bb_lower":  lower,
    })


# ─────────────────────────────────────────────────────────────
#  ATR — Average True Range (Ortalama Gerçek Aralık)
# ─────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — Volatilite göstergesi.

    True Range = max(
        high - low,
        |high - prev_close|,
        |low  - prev_close|
    )
    ATR = SMA(True Range, period)

    Yüksek ATR → yüksek volatilite
    Düşük ATR  → düşük volatilite (sıkışma)

    Not: df'te 'high', 'low', 'close' sütunları olmalı.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.rolling(window=period, min_periods=period).mean()


# ─────────────────────────────────────────────────────────────
#  VWAP — Volume Weighted Average Price
# ─────────────────────────────────────────────────────────────

def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (Hacim Ağırlıklı Ortalama Fiyat).

    VWAP = Σ(typical_price × volume) / Σ(volume)
    typical_price = (high + low + close) / 3

    Fiyat VWAP üstünde → Yükseliş eğilimi
    Fiyat VWAP altında → Düşüş eğilimi

    Not: df'te 'high', 'low', 'close', 'volume' sütunları olmalı.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()

    vwap_values = cumulative_tp_vol / cumulative_vol
    vwap_values = vwap_values.replace([np.inf, -np.inf], np.nan)

    return vwap_values


# ─────────────────────────────────────────────────────────────
#  Tümünü Ekle
# ─────────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kline DataFrame'ine tüm teknik göstergeleri ekler.

    Gerekli sütunlar: open, high, low, close, volume
    Eklenen sütunlar:
      sma_20, ema_20, rsi_14,
      macd_line, macd_signal, macd_hist,
      bb_upper, bb_middle, bb_lower,
      atr_14, vwap

    Returns:
        Göstergeler eklenmiş DataFrame (orijinal sütunlar korunur).
    """
    df = df.copy()
    close = df["close"]

    # ── Hareketli Ortalamalar ──
    df["sma_20"] = sma(close, 20)
    df["ema_20"] = ema(close, 20)

    # ── RSI ──
    df["rsi_14"] = rsi(close, 14)

    # ── MACD ──
    macd_df = macd(close)
    df["macd_line"]   = macd_df["macd_line"]
    df["macd_signal"] = macd_df["macd_signal"]
    df["macd_hist"]   = macd_df["macd_hist"]

    # ── Bollinger Bands ──
    bb_df = bollinger_bands(close, 20)
    df["bb_upper"]  = bb_df["bb_upper"]
    df["bb_middle"] = bb_df["bb_middle"]
    df["bb_lower"]  = bb_df["bb_lower"]

    # ── ATR ──
    df["atr_14"] = atr(df, 14)

    # ── VWAP ──
    df["vwap"] = vwap(df)

    logger.info(
        "📈 Teknik göstergeler eklendi | %d satır | "
        "SMA, EMA, RSI, MACD, BB, ATR, VWAP",
        len(df),
    )

    return df


# ─────────────────────────────────────────────────────────────
#  Tek başına test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Demo: rastgele fiyat verisi ile gösterge testi
    np.random.seed(42)
    n = 100
    prices = 50000 + np.cumsum(np.random.randn(n) * 500)
    demo_df = pd.DataFrame({
        "open":   prices + np.random.randn(n) * 100,
        "high":   prices + abs(np.random.randn(n) * 300),
        "low":    prices - abs(np.random.randn(n) * 300),
        "close":  prices,
        "volume": np.random.uniform(100, 1000, n),
    })

    result = add_all_indicators(demo_df)

    print("\n📊 Gösterge Hesaplama Sonucu (son 5 satır):")
    print(result[["close", "sma_20", "ema_20", "rsi_14",
                   "macd_line", "bb_upper", "bb_lower",
                   "atr_14", "vwap"]].tail().to_string())
    print(f"\nToplam sütun sayısı: {len(result.columns)}")
    print(f"NaN içermeyen satır sayısı: {result.dropna().shape[0]} / {len(result)}")
