# ============================================================
#  binance_collector.py  —  Binance API Veri Toplama Modülü
#  Kripto Para Trading Botu | Ahmet Yılmaz | 1. Hafta
# ============================================================

import time
import logging
import requests
from datetime import datetime, timezone

from config import (
    BINANCE_BASE_URL, BINANCE_API_KEY,
    SYMBOLS, INTERVALS, HISTORICAL_DAYS, KLINE_LIMIT
)
from database import insert_klines, insert_snapshot, get_latest_kline_time, log_fetch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  HTTP Yardımcısı
# ─────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = None) -> dict | list:
    """Binance REST API'ye GET isteği gönderir, hata yönetimi yapar."""
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY else {}
    url = BINANCE_BASE_URL + endpoint

    for attempt in range(3):                   # 3 deneme hakkı
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)

            if resp.status_code == 429:        # Rate limit
                wait = 2 ** attempt * 5
                logger.warning("Rate limit! %ds bekleniyor...", wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            logger.error("HTTP hatası (deneme %d/3): %s", attempt + 1, e)
            time.sleep(2 ** attempt * 3)
        except requests.exceptions.ConnectionError:
            logger.error("Bağlantı hatası (deneme %d/3)", attempt + 1)
            time.sleep(2)

    raise RuntimeError(f"Binance API isteği başarısız: {endpoint}")


# ─────────────────────────────────────────────────────────────
#  Kline (OHLCV) Yardımcıları
# ─────────────────────────────────────────────────────────────

def _parse_kline(raw: list, symbol: str, interval: str) -> dict:
    """
    Binance ham kline listesini veritabanı satırına dönüştürür.
    Binance kline formatı:
      [0]  open_time (ms)
      [1]  open  [2] high  [3] low  [4] close
      [5]  volume
      [6]  close_time (ms)
      [7]  quote_asset_volume
      [8]  number_of_trades
      [9]  taker_buy_base_volume
      [10] taker_buy_quote_volume
      [11] ignore
    """
    open_ts = int(raw[0])
    open_dt = datetime.fromtimestamp(open_ts / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return {
        "symbol":        symbol,
        "interval":      interval,
        "open_time":     open_ts,
        "open_time_dt":  open_dt,
        "open":          float(raw[1]),
        "high":          float(raw[2]),
        "low":           float(raw[3]),
        "close":         float(raw[4]),
        "volume":        float(raw[5]),
        "close_time":    int(raw[6]),
        "quote_volume":  float(raw[7]),
        "trade_count":   int(raw[8]),
        "taker_buy_vol": float(raw[9]),
    }


# ─────────────────────────────────────────────────────────────
#  Tarihsel Kline Çekme
# ─────────────────────────────────────────────────────────────

def fetch_historical_klines(symbol: str, interval: str,
                            days: int = HISTORICAL_DAYS) -> int:
    """
    Belirtilen sembol ve interval için tarihsel kline verisi çeker ve kaydeder.
    Zaten var olan verileri atlar (UNIQUE kısıtı sayesinde).
    Büyük aralıklar için sayfalama (pagination) uygular.
    """
    # En son kaydedilen zaman damgasından devam et
    latest_ts = get_latest_kline_time(symbol, interval)
    if latest_ts:
        start_ms = latest_ts + 1
        logger.info("📂 %s %s: %s'den itibaren güncelleniyor",
                    symbol, interval,
                    datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"))
    else:
        start_ms = int((time.time() - days * 86400) * 1000)
        logger.info("📥 %s %s: Son %d günlük tarihsel veri çekiliyor",
                    symbol, interval, days)

    total_inserted = 0

    while True:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_ms,
            "limit":     KLINE_LIMIT,
        }
        raw_klines = _get("/api/v3/klines", params)

        if not raw_klines:
            break

        rows = [_parse_kline(k, symbol, interval) for k in raw_klines]
        inserted = insert_klines(rows)
        total_inserted += inserted

        # Son kline'ın zamanından bir sonraki sayfaya geç
        last_open_time = int(raw_klines[-1][0])
        start_ms = last_open_time + 1

        logger.debug("  → %d kline alındı, %d yeni kayıt", len(rows), inserted)

        # Binance'in son mevcut kline'ına ulaştıysak dur
        if len(raw_klines) < KLINE_LIMIT:
            break

        time.sleep(0.2)   # Rate limit'e saygı

    log_fetch("binance", symbol, f"historical_{interval}",
              "success", total_inserted)
    logger.info("✅ %s %s: toplam %d yeni kline kaydedildi",
                symbol, interval, total_inserted)
    return total_inserted


def fetch_all_historical() -> None:
    """Tüm semboller ve interval'lar için tarihsel veri çeker."""
    logger.info("=" * 60)
    logger.info("🚀 Tarihsel veri indirme başlıyor...")
    logger.info("   Semboller : %s", list(SYMBOLS.values()))
    logger.info("   Interval'lar: %s", INTERVALS)
    logger.info("=" * 60)

    for name, symbol in SYMBOLS.items():
        for interval in INTERVALS:
            try:
                fetch_historical_klines(symbol, interval)
            except Exception as e:
                logger.error("❌ %s %s hata: %s", symbol, interval, e)
                log_fetch("binance", symbol, f"historical_{interval}",
                          "error", 0, str(e))
            time.sleep(0.5)


# ─────────────────────────────────────────────────────────────
#  Gerçek Zamanlı Fiyat Verisi
# ─────────────────────────────────────────────────────────────

def fetch_ticker_24h(symbol: str) -> dict:
    """24 saatlik istatistikleri çeker (anlık fiyat dahil)."""
    data = _get("/api/v3/ticker/24hr", {"symbol": symbol})
    return {
        "symbol":           symbol,
        "price":            float(data["lastPrice"]),
        "price_change_24h": float(data["priceChange"]),
        "price_change_pct": float(data["priceChangePercent"]),
        "volume_24h":       float(data["volume"]),
        "high_24h":         float(data["highPrice"]),
        "low_24h":          float(data["lowPrice"]),
        "source":           "binance",
    }


def fetch_all_realtime() -> list[dict]:
    """Tüm sembollerin anlık fiyatlarını çeker ve veritabanına kaydeder."""
    results = []
    for name, symbol in SYMBOLS.items():
        try:
            ticker = fetch_ticker_24h(symbol)
            insert_snapshot(ticker)
            results.append(ticker)
            logger.info("💰 %s: $%.4f  (%+.2f%%  24s)",
                        symbol, ticker["price"], ticker["price_change_pct"])
        except Exception as e:
            logger.error("❌ %s fiyat hatası: %s", symbol, e)
            log_fetch("binance", symbol, "realtime", "error", 0, str(e))
    return results


# ─────────────────────────────────────────────────────────────
#  Bağlantı Testi
# ─────────────────────────────────────────────────────────────

def test_connection() -> bool:
    """Binance API bağlantısını test eder."""
    try:
        data = _get("/api/v3/ping")
        logger.info("✅ Binance API bağlantısı başarılı")
        # Sunucu saatini de kontrol et
        server = _get("/api/v3/time")
        server_time = datetime.fromtimestamp(
            server["serverTime"] / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        logger.info("   Sunucu saati: %s", server_time)
        return True
    except Exception as e:
        logger.error("❌ Binance bağlantı testi başarısız: %s", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if test_connection():
        print("\n--- Gerçek Zamanlı Fiyatlar ---")
        fetch_all_realtime()