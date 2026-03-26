# ============================================================
#  data_processor.py  —  Veri Önişleme Modülü
#  Kripto Para Trading Botu | Ahmet Yılmaz | 2-5. Hafta
#
#  İşlevler:
#    - SQLite → pandas DataFrame dönüşümü
#    - Eksik veri tespiti ve temizleme
#    - Normalizasyon / ölçeklendirme
#    - Zaman bazlı özellik çıkarımı
#    - Özellik matrisi (feature matrix) hazırlama
#    - Volatilite / hacim / pattern özellikleri (5. Hafta)
#    - Feature importance analizi (5. Hafta)
# ============================================================

import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from config import (DB_PATH, SYMBOLS, INTERVALS,
                    VOLATILITY_WINDOWS, VOLUME_MA_PERIOD, PATTERN_LOOKBACK)
from indicators import add_all_indicators

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Veritabanından Veri Yükleme
# ─────────────────────────────────────────────────────────────

def load_klines_df(symbol: str = "BTCUSDT",
                   interval: str = "1h",
                   limit: int | None = None) -> pd.DataFrame:
    """
    Kline (OHLCV) verilerini SQLite'dan pandas DataFrame olarak yükler.

    Args:
        symbol:   İşlem çifti (örn. BTCUSDT)
        interval: Zaman dilimi (1h, 4h, 1d)
        limit:    Maksimum satır sayısı (None = tümü)

    Returns:
        Kronolojik sırada DataFrame:
        open_time, open_time_dt, open, high, low, close, volume,
        quote_volume, trade_count, taker_buy_vol
    """
    query = """
        SELECT open_time, open_time_dt, open, high, low, close,
               volume, quote_volume, trade_count, taker_buy_vol
        FROM klines
        WHERE symbol = ? AND interval = ?
        ORDER BY open_time ASC
    """
    params: list = [symbol, interval]

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        logger.warning("⚠️  Veri bulunamadı: %s %s", symbol, interval)
        return df

    # Tarih sütununu datetime'a çevir
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    df.set_index("open_time_dt", inplace=True)

    logger.info("📂 %s %s: %d satır yüklendi (%s → %s)",
                symbol, interval, len(df),
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"))

    return df


def load_snapshots_df(symbol: str | None = None,
                      source: str | None = None,
                      limit: int = 1000) -> pd.DataFrame:
    """
    Anlık fiyat snapshot'larını DataFrame olarak yükler.

    Args:
        symbol: Filtrelemek için sembol (None = tümü)
        source: 'binance' | 'coingecko' | None (tümü)
        limit:  Maksimum satır sayısı

    Returns:
        Kronolojik sırada DataFrame
    """
    query = "SELECT * FROM price_snapshots WHERE 1=1"
    params: list = []

    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if source:
        query += " AND source = ?"
        params.append(source)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)

    logger.info("📂 Snapshot yüklendi: %d kayıt", len(df))
    return df


# ─────────────────────────────────────────────────────────────
#  Veri Temizleme & Kalite Kontrolü
# ─────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kline DataFrame'i temizler:
      1. Mükerrer satırları kaldır (open_time'a göre)
      2. NaN / sıfır fiyatları tespit et ve ileri doldur
      3. Negatif hacim/fiyat değerlerini NaN yap
      4. Aykırı değerleri (outlier) tespit et (IQR yöntemi)

    Returns:
        Temizlenmiş DataFrame
    """
    original_len = len(df)
    df = df.copy()

    # 1. Mükerrer open_time kaldır
    if "open_time" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["open_time"], keep="last")
        dupes = before - len(df)
        if dupes > 0:
            logger.info("🔄 %d mükerrer kayıt kaldırıldı", dupes)

    # 2. Sıfır ve NaN fiyat kontrolü
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            zero_mask = df[col] == 0
            if zero_mask.any():
                df.loc[zero_mask, col] = np.nan
                logger.warning("⚠️  %s: %d sıfır değer NaN yapıldı",
                               col, zero_mask.sum())

    # 3. Negatif değer kontrolü
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "trade_count", "taker_buy_vol"]
    for col in numeric_cols:
        if col in df.columns:
            neg_mask = df[col] < 0
            if neg_mask.any():
                df.loc[neg_mask, col] = np.nan
                logger.warning("⚠️  %s: %d negatif değer NaN yapıldı",
                               col, neg_mask.sum())

    # 4. NaN'ları ileri doldurma (forward fill) + geri doldurma (back fill)
    nan_before = df[price_cols].isna().sum().sum()
    if nan_before > 0:
        df[price_cols] = df[price_cols].ffill().bfill()
        logger.info("🔧 %d NaN değer dolduruldu (ffill + bfill)", nan_before)

    # 5. OHLC tutarlılık kontrolü
    if all(c in df.columns for c in price_cols):
        # high >= open, close, low olmalı
        invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
        if invalid_high.any():
            df.loc[invalid_high, "high"] = df.loc[
                invalid_high, ["open", "high", "close"]
            ].max(axis=1)
            logger.warning("⚠️  %d satırda high değeri düzeltildi",
                           invalid_high.sum())

        # low <= open, close, high olmalı
        invalid_low = df["low"] > df[["open", "close"]].min(axis=1)
        if invalid_low.any():
            df.loc[invalid_low, "low"] = df.loc[
                invalid_low, ["open", "low", "close"]
            ].min(axis=1)
            logger.warning("⚠️  %d satırda low değeri düzeltildi",
                           invalid_low.sum())

    cleaned = original_len - len(df)
    logger.info("🧹 Veri temizleme tamamlandı: %d → %d satır (%d kaldırıldı)",
                original_len, len(df), cleaned)

    return df


def get_data_quality_report(df: pd.DataFrame,
                            symbol: str = "",
                            interval: str = "") -> dict[str, Any]:
    """
    Veri kalitesi raporu oluşturur.

    Returns:
        dict: Aşağıdaki bilgileri içerir:
          - total_rows: Toplam satır sayısı
          - date_range: Veri aralığı (başlangıç - bitiş)
          - missing_values: Sütun bazında eksik değer sayısı
          - zero_prices: Sıfır fiyat sayıları
          - duplicates: Mükerrer kayıt sayısı
          - stats: Temel istatistikler (min, max, mean, std)
    """
    report: dict[str, Any] = {
        "symbol":       symbol,
        "interval":     interval,
        "total_rows":   len(df),
        "columns":      list(df.columns),
    }

    # Tarih aralığı
    if hasattr(df.index, "min") and not df.empty:
        report["date_range"] = {
            "start": str(df.index.min()),
            "end":   str(df.index.max()),
        }

    # Eksik değerler
    missing = df.isna().sum()
    report["missing_values"] = missing[missing > 0].to_dict()

    # Sıfır fiyatlar
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    report["zero_prices"] = {
        col: int((df[col] == 0).sum()) for col in price_cols
    }

    # Mükerrerler
    if "open_time" in df.columns:
        report["duplicates"] = int(df["open_time"].duplicated().sum())
    else:
        report["duplicates"] = int(df.index.duplicated().sum())

    # Temel istatistikler
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        report["stats"] = {}
        for col in ["close", "volume", "trade_count"]:
            if col in numeric_df.columns:
                report["stats"][col] = {
                    "min":  float(numeric_df[col].min()),
                    "max":  float(numeric_df[col].max()),
                    "mean": float(numeric_df[col].mean()),
                    "std":  float(numeric_df[col].std()),
                }

    return report


# ─────────────────────────────────────────────────────────────
#  Normalizasyon / Ölçeklendirme
# ─────────────────────────────────────────────────────────────

def normalize_min_max(df: pd.DataFrame,
                      columns: list[str] | None = None) -> pd.DataFrame:
    """
    Min-Max normalizasyonu: Değerleri [0, 1] aralığına çeker.

    Formula: x_norm = (x - min) / (max - min)
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[col] = 0.0

    logger.info("📏 Min-Max normalizasyonu uygulandı: %s", columns)
    return df


def normalize_zscore(df: pd.DataFrame,
                     columns: list[str] | None = None) -> pd.DataFrame:
    """
    Z-score normalizasyonu: Ortalama 0, standart sapma 1 olacak şekilde.

    Formula: z = (x - μ) / σ
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0

    logger.info("📏 Z-score normalizasyonu uygulandı: %s", columns)
    return df


# ─────────────────────────────────────────────────────────────
#  Zaman Bazlı Özellik Çıkarımı
# ─────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Datetime index'ten zaman bazlı özellikler çıkarır:
      - hour:        Saat (0-23)
      - day_of_week: Haftanın günü (0=Pazartesi, 6=Pazar)
      - day_of_month: Ayın günü (1-31)
      - month:       Ay (1-12)
      - is_weekend:  Haftasonu mu? (0/1)

    Not: Index DatetimeIndex olmalıdır.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("⚠️  Index DatetimeIndex değil, zaman özellikleri eklenemedi.")
        return df

    df["hour"]         = df.index.hour
    df["day_of_week"]  = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"]        = df.index.month
    df["is_weekend"]   = (df.index.dayofweek >= 5).astype(int)

    logger.info("🕐 Zaman özellikleri eklendi: hour, day_of_week, "
                "day_of_month, month, is_weekend")

    return df


# ─────────────────────────────────────────────────────────────
#  Fiyat Değişim Özellikleri
# ─────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fiyat bazlı türetilmiş özellikler ekler:
      - price_change:     Bir önceki kapanışa göre fark
      - price_change_pct: Yüzdesel değişim
      - candle_body:      Mum gövdesi (close - open)
      - candle_body_pct:  Mum gövdesi yüzdesi
      - upper_shadow:     Üst gölge uzunluğu
      - lower_shadow:     Alt gölge uzunluğu
      - range_hl:         Periyot aralığı (high - low)
      - vol_change_pct:   Hacim yüzdesel değişim
    """
    df = df.copy()

    # Fiyat değişimleri
    df["price_change"]     = df["close"].diff()
    df["price_change_pct"] = df["close"].pct_change() * 100

    # Mum gövdesi analizi
    df["candle_body"]     = df["close"] - df["open"]
    df["candle_body_pct"] = (df["candle_body"] / df["open"]) * 100

    # Gölge (shadow) analizi
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Fiyat aralığı
    df["range_hl"] = df["high"] - df["low"]

    # Hacim değişimi
    df["vol_change_pct"] = df["volume"].pct_change() * 100

    logger.info("💹 Fiyat özellikleri eklendi: price_change, candle_body, "
                "shadows, range, vol_change")

    return df


# ─────────────────────────────────────────────────────────────
#  Gelişmiş Özellik Mühendisliği (4. Hafta)
# ─────────────────────────────────────────────────────────────

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Teknik göstergelerden türetilmiş gelişmiş özellikler ekler (4. Hafta).

    Eklenen özellikler:
      - rsi_momentum:       RSI değişim hızı (RSI'nın birinci türevi)
      - macd_hist_change:   MACD histogram değişim oranı (%)
      - bb_width:           Bollinger Band genişliği (normalize)
      - rolling_mean_7/14/30:  Kapanış fiyatı hareketli ortalamaları
      - rolling_std_7/14/30:   Kapanış fiyatı hareketli standart sapmaları

    Args:
        df: Teknik göstergelerin eklenmiş olduğu DataFrame
            (rsi_14, macd_hist, bb_upper, bb_lower, bb_middle, close gerekli)

    Returns:
        Gelişmiş özellikler eklenmiş DataFrame
    """
    df = df.copy()

    # 1. RSI Momentum — RSI'nın değişim hızı
    if "rsi_14" in df.columns:
        df["rsi_momentum"] = df["rsi_14"].diff()
        logger.info("📐 RSI momentum özelliği eklendi")

    # 2. MACD Histogram Değişim Oranı
    if "macd_hist" in df.columns:
        df["macd_hist_change"] = df["macd_hist"].pct_change() * 100
        # Inf değerleri temizle (sıfırdan sıfıra geçiş)
        df["macd_hist_change"] = df["macd_hist_change"].replace(
            [np.inf, -np.inf], np.nan
        )
        logger.info("📐 MACD histogram değişim oranı özelliği eklendi")

    # 3. Bollinger Band Genişliği (normalize edilmiş)
    if all(c in df.columns for c in ["bb_upper", "bb_lower", "bb_middle"]):
        df["bb_width"] = np.where(
            df["bb_middle"] != 0,
            (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"],
            np.nan
        )
        logger.info("📐 Bollinger Band genişliği özelliği eklendi")

    # 4. Rolling Window Özellikleri (7, 14, 30 periyot)
    if "close" in df.columns:
        for window in [7, 14, 30]:
            df[f"rolling_mean_{window}"] = (
                df["close"].rolling(window=window, min_periods=window).mean()
            )
            df[f"rolling_std_{window}"] = (
                df["close"].rolling(window=window, min_periods=window).std()
            )
        logger.info("📐 Rolling window özellikleri eklendi (7, 14, 30 periyot)")

    logger.info("✅ Gelişmiş özellik mühendisliği tamamlandı (4. Hafta)")
    return df


# ─────────────────────────────────────────────────────────────
#  Volatilite Bazlı Özellikler (5. Hafta)
# ─────────────────────────────────────────────────────────────

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatilite bazlı özellikler ekler (5. Hafta).

    Eklenen özellikler:
      - atr_pct:           ATR / close (normalize volatilite)
      - rolling_std_5/10/21: Fiyat standart sapması (farklı pencereler)
      - garman_klass_vol:  Garman-Klass volatilite tahmincisi

    Returns:
        Volatilite özellikleri eklenmiş DataFrame
    """
    df = df.copy()

    # 1. ATR yüzdesi (normalize edilmiş volatilite)
    if "atr_14" in df.columns and "close" in df.columns:
        df["atr_pct"] = (df["atr_14"] / df["close"]) * 100
        logger.info("📐 ATR yüzdesi özelliği eklendi (atr_pct)")

    # 2. Rolling standard deviation (farklı pencere boyutları)
    if "close" in df.columns:
        for window in VOLATILITY_WINDOWS:
            col_name = f"vol_std_{window}"
            df[col_name] = df["close"].rolling(
                window=window, min_periods=window
            ).std()
        logger.info("📐 Rolling volatilite std eklendi: %s", VOLATILITY_WINDOWS)

    # 3. Garman-Klass volatilite tahmincisi
    #    GK = 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"]) ** 2
        df["garman_klass_vol"] = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
            .rolling(window=21, min_periods=21).mean()
        )
        logger.info("📐 Garman-Klass volatilite tahmincisi eklendi")

    logger.info("✅ Volatilite özellikleri tamamlandı (5. Hafta)")
    return df


# ─────────────────────────────────────────────────────────────
#  Hacim Bazlı Özellikler (5. Hafta)
# ─────────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hacim bazlı özellikler ekler (5. Hafta).

    Eklenen özellikler:
      - volume_ratio:    volume / SMA(volume, 20)  — Görece hacim gücü
      - volume_momentum: Volume'ün 5-periyotluk değişim hızı
      - obv:             On-Balance Volume (kümülatif hacim dengesi)

    Returns:
        Hacim özellikleri eklenmiş DataFrame
    """
    df = df.copy()

    if "volume" not in df.columns:
        logger.warning("⚠️  volume sütunu bulunamadı, hacim özellikleri eklenemedi.")
        return df

    # 1. Volume Ratio — Anlık hacim / ortalama hacim
    vol_ma = df["volume"].rolling(
        window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD
    ).mean()
    df["volume_ratio"] = df["volume"] / vol_ma
    df["volume_ratio"] = df["volume_ratio"].replace([np.inf, -np.inf], np.nan)
    logger.info("📐 Volume ratio eklendi (periyot: %d)", VOLUME_MA_PERIOD)

    # 2. Volume Momentum — Hacim değişim hızı
    df["volume_momentum"] = df["volume"].pct_change(periods=5) * 100
    df["volume_momentum"] = df["volume_momentum"].replace(
        [np.inf, -np.inf], np.nan
    )
    logger.info("📐 Volume momentum eklendi (5 periyotluk)")

    # 3. On-Balance Volume (OBV)
    #    Kapanış yükselirse +volume, düşerse -volume, kümülatif toplam
    if "close" in df.columns:
        direction = np.where(
            df["close"] > df["close"].shift(1), 1,
            np.where(df["close"] < df["close"].shift(1), -1, 0)
        )
        df["obv"] = (df["volume"] * direction).cumsum()
        logger.info("📐 On-Balance Volume (OBV) eklendi")

    logger.info("✅ Hacim özellikleri tamamlandı (5. Hafta)")
    return df


# ─────────────────────────────────────────────────────────────
#  Fiyat Pattern Özellikleri (5. Hafta)
# ─────────────────────────────────────────────────────────────

def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fiyat kalıplarına dayalı özellikler ekler (5. Hafta).

    Eklenen özellikler:
      - higher_high:   Son N barda yeni zirve oluştu mu (1/0)
      - lower_low:     Son N barda yeni dip oluştu mu (1/0)
      - trend_strength: Ardışık yükseliş/düşüş bar sayısı (+/−)
      - dist_to_high_20: 20 barlık zirveye mesafe (%)
      - dist_to_low_20:  20 barlık dibe mesafe (%)

    Returns:
        Pattern özellikleri eklenmiş DataFrame
    """
    df = df.copy()
    lb = PATTERN_LOOKBACK

    if "high" not in df.columns or "low" not in df.columns:
        logger.warning("⚠️  high/low sütunları bulunamadı, pattern özellikleri eklenemedi.")
        return df

    # 1. Higher High — high, son N bardaki max'tan büyükse 1
    rolling_max = df["high"].rolling(window=lb, min_periods=lb).max().shift(1)
    df["higher_high"] = (df["high"] > rolling_max).astype(int)
    logger.info("📐 Higher High pattern özelliği eklendi (bakış: %d)", lb)

    # 2. Lower Low — low, son N bardaki min'den küçükse 1
    rolling_min = df["low"].rolling(window=lb, min_periods=lb).min().shift(1)
    df["lower_low"] = (df["low"] < rolling_min).astype(int)
    logger.info("📐 Lower Low pattern özelliği eklendi")

    # 3. Trend Strength — Ardışık yükseliş (+) veya düşüş (−) sayısı
    if "close" in df.columns:
        direction = (df["close"].diff() > 0).astype(int)
        # Ardışık aynı yönü say
        groups = (direction != direction.shift(1)).cumsum()
        streak = direction.groupby(groups).cumcount() + 1
        # Düşüş barlarını negatif yap
        df["trend_strength"] = np.where(direction == 1, streak, -streak)
        logger.info("📐 Trend strength özelliği eklendi")

    # 4. Distance to 20-bar High (%)
    # .shift(1): mevcut barın high'ı pencereye dahil edilmez, geçmiş 20 bar kullanılır
    if "close" in df.columns:
        rolling_high_20 = df["high"].shift(1).rolling(window=20, min_periods=20).max()
        df["dist_to_high_20"] = (
            (rolling_high_20 - df["close"]) / df["close"]
        ) * 100
        logger.info("📐 20 barlık zirveye mesafe özelliği eklendi")

    # 5. Distance to 20-bar Low (%)
    # .shift(1): mevcut barın low'u pencereye dahil edilmez, geçmiş 20 bar kullanılır
    if "close" in df.columns:
        rolling_low_20 = df["low"].shift(1).rolling(window=20, min_periods=20).min()
        df["dist_to_low_20"] = (
            (df["close"] - rolling_low_20) / df["close"]
        ) * 100
        logger.info("📐 20 barlık dibe mesafe özelliği eklendi")

    logger.info("✅ Fiyat pattern özellikleri tamamlandı (5. Hafta)")
    return df


def correlation_analysis(df: pd.DataFrame,
                         target_col: str = "close",
                         top_n: int = 10) -> dict:
    """
    Özellikler arası korelasyon analizi yapar (4. Hafta).

    Hedef değişkenle en yüksek korelasyona sahip özellikleri belirler.

    Args:
        df:         Sayısal özellikler içeren DataFrame
        target_col: Hedef değişken sütunu (varsayılan: close)
        top_n:      Gösterilecek en önemli özellik sayısı

    Returns:
        dict: {
            "target": hedef sütun adı,
            "top_positive": [(özellik, korelasyon), ...],
            "top_negative": [(özellik, korelasyon), ...],
            "full_corr": tüm korelasyon dict'i,
        }
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if target_col not in numeric_df.columns:
        logger.warning("⚠️  Hedef sütun '%s' bulunamadı.", target_col)
        return {}

    # Sabit (zero-variance) sütunları kaldır → numpy divide-by-zero uyarısını önler
    non_const = numeric_df.loc[:, numeric_df.std() > 0]
    if target_col not in non_const.columns:
        return {}

    corr = non_const.corr()[target_col].drop(target_col, errors="ignore")
    corr = corr.dropna().sort_values(ascending=False)

    top_positive = list(corr.head(top_n).items())
    top_negative = list(corr.tail(top_n).items())

    logger.info("📊 Korelasyon analizi tamamlandı: %d özellik analiz edildi",
                len(corr))

    return {
        "target":       target_col,
        "top_positive": [(name, round(val, 4)) for name, val in top_positive],
        "top_negative": [(name, round(val, 4)) for name, val in top_negative],
        "full_corr":    {k: round(v, 4) for k, v in corr.items()},
    }


# ─────────────────────────────────────────────────────────────
#  Feature Importance Analizi (5. Hafta)
# ─────────────────────────────────────────────────────────────

def _corr_and_variance_importance(numeric_df: pd.DataFrame,
                                   features: list[str],
                                   target_col: str,
                                   top_n: int) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Korelasyon ve varyans bazlı önem hesaplar."""
    corr = numeric_df[features].corrwith(numeric_df[target_col]).abs()
    corr = corr.dropna().sort_values(ascending=False)

    variances = numeric_df[features].var().sort_values(ascending=False)
    threshold = variances.quantile(0.05)

    result = {
        "correlation": [(name, round(val, 4)) for name, val in corr.head(top_n).items()],
        "variance": [(name, round(val, 4)) for name, val in variances.head(top_n).items()],
        "low_variance": [name for name, val in variances.items() if val <= threshold],
    }
    return corr, variances, result


def _mi_importance(numeric_df: pd.DataFrame,
                   features: list[str],
                   target_col: str,
                   top_n: int) -> tuple[list, np.ndarray | None]:
    """Mutual Information skoru hesaplar."""
    try:
        from sklearn.feature_selection import mutual_info_regression
        X = np.asarray(numeric_df[features].values)
        y = np.asarray(numeric_df[target_col].values)
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)
        return [(name, round(val, 4)) for name, val in mi_series.head(top_n).items()], mi_scores
    except ImportError:
        logger.warning("⚠️  scikit-learn kurulu değil, MI skoru atlandı.")
        return [], None


def _combined_ranking(features: list[str], corr: pd.Series,
                      variances: pd.Series, mi_scores: np.ndarray | None,
                      top_n: int) -> list[tuple[str, float]]:
    """Üç yöntemin birleşik sıralamasını hesaplar."""
    rank_corr = {name: i for i, (name, _) in enumerate(corr.items())}
    rank_var = {name: i for i, (name, _) in enumerate(variances.items())}

    if mi_scores is not None:
        mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)
        rank_mi = {name: i for i, name in enumerate(mi_series.index)}
    else:
        rank_mi = {name: i for i, name in enumerate(features)}

    combined = {}
    for feat in features:
        ranks = [rank_corr.get(feat, len(features)),
                 rank_var.get(feat, len(features)),
                 rank_mi.get(feat, len(features))]
        combined[feat] = sum(ranks) / len(ranks)

    combined_sorted = sorted(combined.items(), key=lambda x: x[1])
    top_items = combined_sorted[:top_n]
    return [(name, round(avg_rank, 2)) for name, avg_rank in top_items]


def feature_importance_analysis(df: pd.DataFrame,
                                target_col: str = "close",
                                top_n: int = 15) -> dict[str, Any]:
    """
    Gelişmiş feature importance analizi yapar (5. Hafta).

    Üç farklı yöntemle özelliklerin önemini değerlendirir:
      1. Korelasyon bazlı önem (|pearson_r|)
      2. Varyans bazlı değerlendirme (düşük varyanslı özellik tespiti)
      3. Mutual Information skoru (doğrusal olmayan ilişkiler)

    Args:
        df:         Sayısal özellikler içeren DataFrame
        target_col: Hedef değişken sütunu
        top_n:      Gösterilecek özellik sayısı

    Returns:
        dict: {
            "correlation": [(özellik, |r|), ...],
            "variance":    [(özellik, varyans), ...],
            "mutual_info": [(özellik, mi_score), ...],
            "low_variance": [düşük varyanslı özellikler],
            "combined_rank": [(özellik, ortalama_sıra), ...],
        }
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    if target_col not in numeric_df.columns:
        logger.warning("⚠️  Hedef sütun '%s' bulunamadı.", target_col)
        return {}

    non_const_cols = numeric_df.columns[numeric_df.std() > 0]
    numeric_df = numeric_df[non_const_cols]

    features = [c for c in numeric_df.columns if c != target_col]
    if not features:
        return {}

    corr, variances, result = _corr_and_variance_importance(
        numeric_df, features, target_col, top_n)

    mi_list, mi_scores = _mi_importance(numeric_df, features, target_col, top_n)
    result["mutual_info"] = mi_list

    result["combined_rank"] = _combined_ranking(
        features, corr, variances, mi_scores, top_n)

    logger.info("📊 Feature importance analizi tamamlandı: %d özellik", len(features))
    return result


# ─────────────────────────────────────────────────────────────
#  Tam Özellik Matrisi
# ─────────────────────────────────────────────────────────────

def build_feature_matrix(symbol: str = "BTCUSDT",
                         interval: str = "1h",
                         drop_na: bool = True) -> pd.DataFrame:
    """
    Tam özellik matrisi oluşturur:
      1. Veritabanından ham veri yükle
      2. Temizle
      3. Teknik göstergeleri ekle
      4. Fiyat özelliklerini ekle
      5. Zaman özelliklerini ekle
      6. Gelişmiş özellik mühendisliği (4. Hafta)
      7. Volatilite özellikleri (5. Hafta)
      8. Hacim özellikleri (5. Hafta)
      9. Fiyat pattern özellikleri (5. Hafta)
     10. (İsteğe bağlı) NaN satırları kaldır

    Args:
        symbol:   İşlem çifti (BTCUSDT, ETHUSDT, SOLUSDT)
        interval: Zaman dilimi (1h, 4h, 1d)
        drop_na:  Gösterge hesabından kalan NaN satırları kaldır

    Returns:
        Analiz-hazır özellik matrisi (DataFrame)
    """
    logger.info("=" * 60)
    logger.info("🔨 Özellik matrisi hazırlanıyor: %s %s", symbol, interval)
    logger.info("=" * 60)

    # 1. Ham veri yükle
    df = load_klines_df(symbol, interval)
    if df.empty:
        logger.error("❌ Veri yok, matris oluşturulamadı.")
        return df

    # 2. Temizle
    df = clean_dataframe(df)

    # 3. Teknik göstergeler
    df = add_all_indicators(df)

    # 4. Fiyat özellikleri
    df = add_price_features(df)

    # 5. Zaman özellikleri
    df = add_time_features(df)

    # 6. Gelişmiş özellik mühendisliği (4. Hafta)
    df = add_advanced_features(df)

    # 7. Volatilite özellikleri (5. Hafta)
    df = add_volatility_features(df)

    # 8. Hacim özellikleri (5. Hafta)
    df = add_volume_features(df)

    # 9. Fiyat pattern özellikleri (5. Hafta)
    df = add_pattern_features(df)

    # 10. NaN kaldır
    if drop_na:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped > 0:
            logger.info("🗑️  NaN satırlar kaldırıldı: %d (kalan: %d)",
                        dropped, len(df))

    logger.info("✅ Özellik matrisi hazır: %d satır × %d sütun",
                len(df), len(df.columns))

    return df


# ─────────────────────────────────────────────────────────────
#  Tüm Semboller İçin Analiz
# ─────────────────────────────────────────────────────────────

def analyze_all_data() -> None:
    """
    Tüm sembol/interval kombinasyonları için veri kalitesi raporu
    ve özellik matrisi özeti yazdırır.
    """
    print("\n" + "═" * 60)
    print("  📊  VERİ KALİTESİ ANALİZİ")
    print("═" * 60)

    for name, symbol in SYMBOLS.items():
        for interval in INTERVALS:
            print(f"\n── {symbol} / {interval} {'─' * 35}")

            df = load_klines_df(symbol, interval)
            if df.empty:
                print("   ⚠️  Veri bulunamadı.")
                continue

            report = get_data_quality_report(df, symbol, interval)

            print(f"   📦 Toplam satır : {report['total_rows']:,}")
            if report.get("date_range"):
                print(f"   📅 Tarih aralığı: "
                      f"{report['date_range']['start']} → "
                      f"{report['date_range']['end']}")
            print(f"   🔁 Mükerrer     : {report['duplicates']}")

            if report["missing_values"]:
                print(f"   ⚠️  Eksik değerler:")
                for col, cnt in report["missing_values"].items():
                    print(f"       {col}: {cnt}")
            else:
                print("   ✅ Eksik değer yok")

            if report.get("stats") and "close" in report["stats"]:
                s = report["stats"]["close"]
                print(f"   💰 Fiyat: ${s['min']:,.2f} — ${s['max']:,.2f} "
                      f"(ort: ${s['mean']:,.2f})")

    print("\n" + "═" * 60)
    print("  Analiz tamamlandı!")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────
#  Tek başına test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Veri analizi
    analyze_all_data()

    # Örnek özellik matrisi
    print("\n\n📐 Örnek Özellik Matrisi (BTCUSDT 1h):")
    fm = build_feature_matrix("BTCUSDT", "1h")
    if not fm.empty:
        print(f"   Boyut: {fm.shape[0]} satır × {fm.shape[1]} sütun")
        print(f"   Sütunlar: {list(fm.columns)}")
        print(f"\n   Son 3 satır:\n{fm.tail(3).to_string()}")
