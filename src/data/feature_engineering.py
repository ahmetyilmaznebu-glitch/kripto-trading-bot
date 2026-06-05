"""
OHLCV -> model ozellikleri.

Tum ozellikler *duragan* (fiyat seviyesine bagimli degil):
- Getiri, oran, yuzde degisim kullanilir.
- Ham Open/High/Low/Close LSTM veya XGBoost'a verilmez.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from src.data.ml_config import FEATURE_COLUMNS, PRICE_COLUMN, TARGET_COLUMN, DIRECTION_THRESHOLD


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        ohlcv: open, high, low, close, volume (kucuk harf)

    Returns:
        Ozellik + hedef + close (grafik icin) iceren DataFrame.
        Son satir(lar) NaN oldugu icin dropna uygulanir.
    """
    df = ohlcv.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # --- Getiri tabanli ---
    df["log_return_1d"] = np.log(close / close.shift(1))
    df["high_low_range"] = (high - low) / close
    df["close_open_return"] = close / open_ - 1.0
    df["volume_change"] = volume.pct_change()

    # --- Teknik gostergeler (oran / normalize) ---
    df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd = ta.trend.MACD(close=close)
    df["macd_pct"] = macd.macd() / close
    df["macd_signal_pct"] = macd.macd_signal() / close

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (close - bb_lower) / width
    df["bb_width"] = width / bb_mid

    sma20 = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    df["sma_ratio_20"] = close / sma20 - 1.0
    df["ema_ratio_50"] = close / ema50 - 1.0

    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    df["atr_pct"] = atr.average_true_range() / close
    df["adx_14"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
    df["stoch_k"] = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=14
    ).stoch()

    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["obv_change"] = obv.pct_change()

    # --- Gecikmeli getiriler ---
    df["return_lag_5"] = np.log(close / close.shift(5))
    df["return_lag_10"] = np.log(close / close.shift(10))
    df["return_lag_20"] = np.log(close / close.shift(20))
    df["volatility_20"] = df["log_return_1d"].rolling(20).std()
    df["momentum_10"] = close / close.shift(10) - 1.0

    # --- Hedef: olceklemeden ONCE (gercek fiyat karsilastirmasi) ---
    next_close = close.shift(-1)
    next_return = next_close / close - 1.0
    df["next_return"] = next_return

    # Dead-zone filtreli binary hedef:
    # abs(next_return) <= DIRECTION_THRESHOLD -> -1 (gecersiz, sonra cikarilir)
    # next_return > DIRECTION_THRESHOLD  -> 1 (UP)
    # next_return < -DIRECTION_THRESHOLD -> 0 (DOWN)
    df[TARGET_COLUMN] = -1  # varsayilan: dead-zone (gecersiz)
    df.loc[next_return > DIRECTION_THRESHOLD, TARGET_COLUMN] = 1
    df.loc[next_return < -DIRECTION_THRESHOLD, TARGET_COLUMN] = 0

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    keep = FEATURE_COLUMNS + [TARGET_COLUMN, "next_return", PRICE_COLUMN]
    keep = [c for c in keep if c in df.columns]
    return df[keep]
