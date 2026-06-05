"""
Gunluk OHLCV verisi cekme — tek kaynak: Binance (1d).
Yedek: yerel SQLite (crypto_data.db) veya onceden kaydedilmis CSV.
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests

from src.data.ml_config import (
    DATA_RAW_DIR,
    FETCH_DAYS,
    PROJECT_ROOT,
    TICKER_TO_BINANCE,
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _klines_to_dataframe(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="first")]
    return df.dropna()


def fetch_binance_daily(symbol: str, days: int = FETCH_DAYS) -> pd.DataFrame | None:
    """Binance 1d kline verisi."""
    try:
        from binance.client import Client

        client = Client(api_key="", api_secret="")
        klines = client.get_historical_klines(symbol, "1d", f"{days} days ago UTC")
        if klines and len(klines) >= 100:
            df = _klines_to_dataframe(klines)
            print(f"  [Binance] {symbol}: {len(df)} gunluk mum")
            return df
    except Exception as e:
        print(f"  [Binance connector] {e}")

    try:
        import urllib3

        urllib3.disable_warnings()
        all_klines: list = []
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 60 * 60 * 1000
        current = start_ms
        while current < end_ms:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": "1d",
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=30,
                verify=False,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_klines.extend(batch)
            current = batch[-1][6] + 1
            if len(batch) < 1000:
                break
            time.sleep(0.25)
        if len(all_klines) >= 100:
            df = _klines_to_dataframe(all_klines)
            print(f"  [Binance REST] {symbol}: {len(df)} gunluk mum")
            return df
    except Exception as e:
        print(f"  [Binance REST] {e}")
    return None


def fetch_from_sqlite(ticker: str) -> pd.DataFrame | None:
    """config.SYMBOLS uzerinden SQLite klines tablosundan gunluk veri."""
    try:
        from config import DB_PATH, SYMBOLS

        symbol = TICKER_TO_BINANCE.get(ticker, ticker.replace("-USD", "USDT"))
        coin_key = None
        for k, v in SYMBOLS.items():
            if v == symbol:
                coin_key = k
                break
        if not os.path.exists(DB_PATH):
            return None

        import sqlite3

        conn = sqlite3.connect(DB_PATH)
        for interval in ("1d", "4h", "1h"):
            df = pd.read_sql_query(
                """
                SELECT open_time_dt, open, high, low, close, volume
                FROM klines
                WHERE symbol = ? AND interval = ?
                ORDER BY open_time ASC
                """,
                conn,
                params=(symbol, interval),
            )
            if len(df) >= 100:
                break
        conn.close()

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["open_time_dt"])
        df = df.set_index("date").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if interval in ("1h", "4h"):
            df = df.resample("1D").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            ).dropna()

        df = df[["open", "high", "low", "close", "volume"]]
        print(f"  [SQLite] {symbol} ({interval}): {len(df)} gun")
        return df
    except Exception as e:
        print(f"  [SQLite] {e}")
        return None


def fetch_ohlcv(ticker: str = "BTC-USD") -> pd.DataFrame:
    """
    Gunluk OHLCV DataFrame dondurur.
    Index: date (DatetimeIndex), sutunlar: open, high, low, close, volume
    """
    symbol = TICKER_TO_BINANCE.get(ticker, ticker.replace("-USD", "USDT"))
    print(f"\nOHLCV cekiliyor: {ticker} ({symbol})")

    df = fetch_binance_daily(symbol)
    if df is None or len(df) < 100:
        df = fetch_from_sqlite(ticker)

    cached = os.path.join(DATA_RAW_DIR, f"{ticker}_ohlcv.csv")
    if (df is None or len(df) < 100) and os.path.exists(cached):
        df = pd.read_csv(cached, index_col=0, parse_dates=True)
        print(f"  [Onbellek CSV] {len(df)} satir")

    if df is None or len(df) < 100:
        raise RuntimeError(
            f"{ticker} icin yeterli OHLCV verisi yok. "
            "Once 'python main.py historical' calistirin veya internet baglantinizi kontrol edin."
        )

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    df.to_csv(cached)
    # Eski dashboard uyumlulugu (Title Case)
    legacy = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    legacy.to_csv(os.path.join(DATA_RAW_DIR, f"{ticker}_raw.csv"))
    return df
