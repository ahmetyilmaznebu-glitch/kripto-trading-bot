# ============================================================
#  signal_generator.py  —  Alım/Satım Sinyal Üretim Modülü
#  Kripto Para Trading Botu | Ahmet Yılmaz | 3. Hafta
#
#  Stratejiler:
#    1. RSI Sinyalleri (Oversold / Overbought)
#    2. MACD Crossover Sinyalleri
#    3. Bollinger Bands Sinyalleri
#    4. SMA/EMA Crossover Sinyalleri
#    5. Kombine Ağırlıklı Sinyal
#
#  Tüm stratejiler bağımsız olarak hesaplanır ve
#  daha sonra ağırlıklı ortalama ile birleştirilir.
# ============================================================

import logging
import pandas as pd
import numpy as np

from config import SYMBOLS, INTERVALS, SIGNAL_WEIGHTS, RSI_OVERSOLD, RSI_OVERBOUGHT
from data_processor import load_klines_df, clean_dataframe
from indicators import add_all_indicators
from database import insert_signals

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  1. RSI Sinyalleri
# ─────────────────────────────────────────────────────────────

def generate_rsi_signal(df: pd.DataFrame) -> pd.Series:
    """
    RSI bazlı alım/satım sinyali üretir.

    Kurallar:
      RSI < 30 (oversold)   → BUY
      RSI > 70 (overbought) → SELL
      Diğer                 → HOLD

    Args:
        df: rsi_14 sütununu içeren DataFrame

    Returns:
        pd.Series: Her satır için 'BUY', 'SELL' veya 'HOLD'
    """
    conditions = [
        df["rsi_14"] < RSI_OVERSOLD,
        df["rsi_14"] > RSI_OVERBOUGHT,
    ]
    choices = ["BUY", "SELL"]

    signal = pd.Series(
        np.select(conditions, choices, default="HOLD"),
        index=df.index,
        name="rsi_signal",
    )

    buy_count = (signal == "BUY").sum()
    sell_count = (signal == "SELL").sum()
    logger.info("📊 RSI Sinyali: %d BUY, %d SELL, %d HOLD",
                buy_count, sell_count, len(signal) - buy_count - sell_count)

    return signal


# ─────────────────────────────────────────────────────────────
#  2. MACD Crossover Sinyalleri
# ─────────────────────────────────────────────────────────────

def generate_macd_signal(df: pd.DataFrame) -> pd.Series:
    """
    MACD line ve signal line crossover'a dayalı sinyal üretir.

    Kurallar:
      MACD line, signal line'ı yukarı keserse  → BUY  (bullish crossover)
      MACD line, signal line'ı aşağı keserse   → SELL (bearish crossover)
      Kesişim yoksa                            → HOLD

    Crossover tespiti:
      Önceki barda MACD < Signal ve şu an MACD >= Signal → BUY
      Önceki barda MACD > Signal ve şu an MACD <= Signal → SELL
    """
    macd_line = df["macd_line"]
    macd_signal = df["macd_signal"]

    # Önceki bar ile karşılaştırma
    prev_diff = (macd_line.shift(1) - macd_signal.shift(1))
    curr_diff = (macd_line - macd_signal)

    conditions = [
        (prev_diff < 0) & (curr_diff >= 0),   # Yukarı kesişim
        (prev_diff > 0) & (curr_diff <= 0),   # Aşağı kesişim
    ]
    choices = ["BUY", "SELL"]

    signal = pd.Series(
        np.select(conditions, choices, default="HOLD"),
        index=df.index,
        name="macd_signal_trade",
    )

    buy_count = (signal == "BUY").sum()
    sell_count = (signal == "SELL").sum()
    logger.info("📊 MACD Sinyali: %d BUY, %d SELL, %d HOLD",
                buy_count, sell_count, len(signal) - buy_count - sell_count)

    return signal


# ─────────────────────────────────────────────────────────────
#  3. Bollinger Bands Sinyalleri
# ─────────────────────────────────────────────────────────────

def generate_bb_signal(df: pd.DataFrame) -> pd.Series:
    """
    Bollinger Bands'a dayalı sinyal üretir.

    Kurallar:
      Fiyat alt banda temas / altına düşers   → BUY  (potansiyel dip)
      Fiyat üst banda temas / üstüne çıkarsa  → SELL (potansiyel tepe)
      Fiyat bantlar arasındaysa                → HOLD

    Not: Fiyat olarak 'close' kullanılır.
    """
    close = df["close"]
    bb_upper = df["bb_upper"]
    bb_lower = df["bb_lower"]

    conditions = [
        close <= bb_lower,   # Alt banda temas
        close >= bb_upper,   # Üst banda temas
    ]
    choices = ["BUY", "SELL"]

    signal = pd.Series(
        np.select(conditions, choices, default="HOLD"),
        index=df.index,
        name="bb_signal",
    )

    buy_count = (signal == "BUY").sum()
    sell_count = (signal == "SELL").sum()
    logger.info("📊 BB Sinyali: %d BUY, %d SELL, %d HOLD",
                buy_count, sell_count, len(signal) - buy_count - sell_count)

    return signal


# ─────────────────────────────────────────────────────────────
#  4. SMA/EMA Crossover Sinyalleri
# ─────────────────────────────────────────────────────────────

def generate_ma_crossover_signal(df: pd.DataFrame) -> pd.Series:
    """
    Fiyatın SMA/EMA hareketli ortalamalarını kesmesine dayalı sinyal.

    Kurallar:
      Fiyat EMA'nın üstüne çıkarsa  → BUY  (yükseliş trendi)
      Fiyat EMA'nın altına inerse    → SELL (düşüş trendi)
      Diğer durumlarda              → HOLD

    EMA tercih edilir çünkü son fiyatlara daha duyarlıdır.
    """
    close = df["close"]
    ema_20 = df["ema_20"]

    # Önceki bar ile karşılaştırma (crossover)
    prev_above = close.shift(1) > ema_20.shift(1)
    curr_above = close > ema_20

    prev_below = close.shift(1) < ema_20.shift(1)
    curr_below = close < ema_20

    conditions = [
        prev_below & curr_above,   # Fiyat EMA üstüne çıktı
        prev_above & curr_below,   # Fiyat EMA altına indi
    ]
    choices = ["BUY", "SELL"]

    signal = pd.Series(
        np.select(conditions, choices, default="HOLD"),
        index=df.index,
        name="ma_signal",
    )

    buy_count = (signal == "BUY").sum()
    sell_count = (signal == "SELL").sum()
    logger.info("📊 MA Crossover Sinyali: %d BUY, %d SELL, %d HOLD",
                buy_count, sell_count, len(signal) - buy_count - sell_count)

    return signal


# ─────────────────────────────────────────────────────────────
#  5. Kombine Ağırlıklı Sinyal
# ─────────────────────────────────────────────────────────────

def _signal_to_numeric(signal: pd.Series) -> pd.Series:
    """Metin sinyalini sayısal değere çevirir: BUY=+1, SELL=-1, HOLD=0."""
    mapping = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
    return signal.map(mapping).fillna(0.0)


def generate_combined_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm bireysel sinyallerin ağırlıklı ortalamasıyla nihai sinyal üretir.

    Ağırlıklar config.py'den alınır (SIGNAL_WEIGHTS).

    Skor hesaplama:
      score = Σ(sinyal_i × ağırlık_i) / Σ(ağırlık_i)

    Karar:
      score >  0.3 → BUY
      score < -0.3 → SELL
      diğer        → HOLD

    Çıkış:
      strength: |score| değeri (0-1 arası sinyal gücü)

    Returns:
        DataFrame: rsi_signal, macd_signal, bb_signal, ma_signal,
                   combined, strength sütunları
    """
    # Bireysel sinyaller
    rsi_sig  = generate_rsi_signal(df)
    macd_sig = generate_macd_signal(df)
    bb_sig   = generate_bb_signal(df)
    ma_sig   = generate_ma_crossover_signal(df)

    # Sayısal dönüşüm
    rsi_num  = _signal_to_numeric(rsi_sig)
    macd_num = _signal_to_numeric(macd_sig)
    bb_num   = _signal_to_numeric(bb_sig)
    ma_num   = _signal_to_numeric(ma_sig)

    # Ağırlıklı skor
    weights = SIGNAL_WEIGHTS
    total_weight = sum(weights.values())

    score = (
        rsi_num  * weights["rsi"]  +
        macd_num * weights["macd"] +
        bb_num   * weights["bb"]   +
        ma_num   * weights["ma"]
    ) / total_weight

    # Nihai karar
    combined = pd.Series("HOLD", index=df.index, name="combined")
    combined[score > 0.3]  = "BUY"
    combined[score < -0.3] = "SELL"

    # Sinyal gücü
    strength = score.abs().clip(0, 1)

    # Sonuç DataFrame
    result = pd.DataFrame({
        "rsi_signal":  rsi_sig,
        "macd_signal": macd_sig,
        "bb_signal":   bb_sig,
        "ma_signal":   ma_sig,
        "combined":    combined,
        "strength":    strength.round(4),
    }, index=df.index)

    buy_count  = (combined == "BUY").sum()
    sell_count = (combined == "SELL").sum()
    hold_count = (combined == "HOLD").sum()

    logger.info("🎯 Kombine Sinyal: %d BUY, %d SELL, %d HOLD",
                buy_count, sell_count, hold_count)

    return result


# ─────────────────────────────────────────────────────────────
#  Tüm Sinyalleri Üret ve Kaydet
# ─────────────────────────────────────────────────────────────

def generate_all_signals(symbol: str, interval: str) -> pd.DataFrame | None:
    """
    Belirtilen sembol/interval için:
      1. Kline verisini yükle
      2. Temizle
      3. Teknik göstergeleri hesapla
      4. Sinyalleri üret
      5. Veritabanına kaydet

    Returns:
        Sinyal DataFrame veya None (veri yoksa)
    """
    logger.info("─" * 60)
    logger.info("🔔 Sinyal üretimi: %s %s", symbol, interval)

    # 1-2. Veri yükle ve temizle
    df = load_klines_df(symbol, interval)
    if df.empty:
        logger.warning("⚠️  %s %s: Veri yok, sinyal üretilemedi.", symbol, interval)
        return None

    df = clean_dataframe(df)

    # 3. Teknik göstergeler
    df = add_all_indicators(df)
    df = df.dropna()

    if df.empty:
        logger.warning("⚠️  %s %s: Yeterli veri yok.", symbol, interval)
        return None

    # 4. Sinyalleri üret
    signals_df = generate_combined_signal(df)

    # open_time bilgisini ekle
    signals_df["open_time"] = df["open_time"]

    # 5. Veritabanına kaydet
    rows = []
    for idx, row in signals_df.iterrows():
        rows.append({
            "symbol":      symbol,
            "interval":    interval,
            "open_time":   int(row["open_time"]),
            "rsi_signal":  row["rsi_signal"],
            "macd_signal": row["macd_signal"],
            "bb_signal":   row["bb_signal"],
            "ma_signal":   row["ma_signal"],
            "combined":    row["combined"],
            "strength":    row["strength"],
        })

    saved = insert_signals(rows)
    logger.info("✅ %s %s: %d sinyal kaydedildi", symbol, interval, saved)

    return signals_df


def generate_signals_all_symbols() -> dict:
    """
    Tüm sembol/interval kombinasyonları için sinyal üretir.

    Returns:
        dict: {(symbol, interval): signals_df} eşleştirmesi
    """
    logger.info("=" * 60)
    logger.info("🔔 TÜM SEMBOLLER İÇİN SİNYAL ÜRETİMİ BAŞLIYOR")
    logger.info("=" * 60)

    all_signals = {}

    for name, symbol in SYMBOLS.items():
        for interval in INTERVALS:
            result = generate_all_signals(symbol, interval)
            if result is not None:
                all_signals[(symbol, interval)] = result

    logger.info("=" * 60)
    logger.info("✅ Sinyal üretimi tamamlandı: %d sembol/interval", len(all_signals))
    logger.info("=" * 60)

    return all_signals


# ─────────────────────────────────────────────────────────────
#  Son Sinyal Özeti
# ─────────────────────────────────────────────────────────────

def get_signal_summary(symbol: str, interval: str,
                       df: pd.DataFrame | None = None) -> dict:
    """
    Belirtilen sembol/interval için son sinyallerin özetini döndürür.

    Returns:
        dict: Son sinyaller ve istatistikler
    """
    if df is None:
        df_raw = load_klines_df(symbol, interval)
        if df_raw.empty:
            return {"symbol": symbol, "interval": interval, "status": "VERİ YOK"}

        df_raw = clean_dataframe(df_raw)
        df_raw = add_all_indicators(df_raw)
        df_raw = df_raw.dropna()

        if df_raw.empty:
            return {"symbol": symbol, "interval": interval, "status": "YETERSİZ VERİ"}

        signals = generate_combined_signal(df_raw)
    else:
        signals = df

    last = signals.iloc[-1]

    return {
        "symbol":      symbol,
        "interval":    interval,
        "rsi_signal":  last["rsi_signal"],
        "macd_signal": last["macd_signal"],
        "bb_signal":   last["bb_signal"],
        "ma_signal":   last["ma_signal"],
        "combined":    last["combined"],
        "strength":    last["strength"],
        "total_bars":  len(signals),
        "buy_count":   int((signals["combined"] == "BUY").sum()),
        "sell_count":  int((signals["combined"] == "SELL").sum()),
        "hold_count":  int((signals["combined"] == "HOLD").sum()),
    }


# ─────────────────────────────────────────────────────────────
#  Tek Başına Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n🔔 Sinyal Üretim Testi")
    print("=" * 60)

    # Demo: rastgele veri ile test
    np.random.seed(42)
    n = 200
    prices = 50000 + np.cumsum(np.random.randn(n) * 500)

    demo_df = pd.DataFrame({
        "open_time": range(n),
        "open":   prices + np.random.randn(n) * 100,
        "high":   prices + abs(np.random.randn(n) * 300),
        "low":    prices - abs(np.random.randn(n) * 300),
        "close":  prices,
        "volume": np.random.uniform(100, 1000, n),
    })

    from indicators import add_all_indicators as _add
    demo_df = _add(demo_df)
    demo_df = demo_df.dropna()

    result = generate_combined_signal(demo_df)
    print(f"\n📊 Son 10 sinyal:")
    print(result.tail(10).to_string())

    summary = get_signal_summary("DEMO", "1h", result)
    print(f"\n📋 Özet: {summary}")
