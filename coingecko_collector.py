# ============================================================
#  coingecko_collector.py  —  CoinGecko API Veri Toplam
# ============================================================

import time
import logging
import requests
from datetime import datetime, timezone

from config import COINGECKO_BASE_URL, COINGECKO_API_KEY, COINGECKO_IDS
from database import insert_market_data, insert_snapshot, log_fetch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  HTTP Yardımcısı
# ─────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = None) -> dict | list:
    """CoinGecko REST API'ye GET isteği gönderir."""
    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-pro-api-key"] = COINGECKO_API_KEY

    url = COINGECKO_BASE_URL + endpoint
    if params is None:
        params = {}

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)

            if resp.status_code == 429:       # Rate limit (ücretsiz plan: 30/dk)
                wait = 60 if attempt == 0 else 120
                logger.warning("⚠️  CoinGecko rate limit! %ds bekleniyor...", wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            logger.error("HTTP hatası (deneme %d/3): %s", attempt + 1, e)
            time.sleep(2 ** attempt * 3)
        except requests.exceptions.ConnectionError:
            logger.error("Bağlantı hatası (deneme %d/3)", attempt + 1)
            time.sleep(5)

    raise RuntimeError(f"CoinGecko API isteği başarısız: {endpoint}")


# ─────────────────────────────────────────────────────────────
#  Piyasa Verisi (Market Data)
# ─────────────────────────────────────────────────────────────

def fetch_market_data(coin_ids: list[str] = None) -> list[dict]:
    """
    CoinGecko'dan kapsamlı piyasa verisi çeker:
    Fiyat, piyasa değeri, hacim, ATH, dolaşımdaki arz vb.
    """
    if coin_ids is None:
        coin_ids = list(COINGECKO_IDS.values())

    params = {
        "vs_currency":           "usd",
        "ids":                   ",".join(coin_ids),
        "order":                 "market_cap_desc",
        "per_page":              len(coin_ids),
        "page":                  1,
        "sparkline":             False,
        "price_change_percentage": "24h",
    }

    raw_list = _get("/coins/markets", params)
    results = []

    for coin in raw_list:
        row = {
            "coin_id":             coin["id"],
            "symbol":              coin["symbol"].upper(),
            "current_price":       coin.get("current_price"),
            "market_cap":          coin.get("market_cap"),
            "market_cap_rank":     coin.get("market_cap_rank"),
            "total_volume":        coin.get("total_volume"),
            "high_24h":            coin.get("high_24h"),
            "low_24h":             coin.get("low_24h"),
            "price_change_24h":    coin.get("price_change_24h"),
            "price_change_pct_24h":coin.get("price_change_percentage_24h"),
            "circulating_supply":  coin.get("circulating_supply"),
            "total_supply":        coin.get("total_supply"),
            "ath":                 coin.get("ath"),
            "ath_change_pct":      coin.get("ath_change_percentage"),
            "last_updated":        coin.get("last_updated"),
        }
        insert_market_data(row)
        results.append(row)

        logger.info(
            "📊 %-6s | $%-12.2f | Piyasa Değeri: $%s | Sıra: #%s",
            row["symbol"],
            row["current_price"] or 0,
            f'{row["market_cap"] or 0:,.0f}',
            row["market_cap_rank"],
        )

    log_fetch("coingecko", ",".join(coin_ids), "market_data",
              "success", len(results))
    return results


# ─────────────────────────────────────────────────────────────
#  Tarihsel OHLCV (CoinGecko alternatifi)
# ─────────────────────────────────────────────────────────────

def fetch_ohlc(coin_id: str, days: int = 30) -> list[dict]:
    """
    CoinGecko OHLC endpoint'inden mum verisi çeker.
    (Binance'i tamamlayıcı, alternatif kaynak olarak)
    Desteklenen days: 1, 7, 14, 30, 90, 180, 365
    """
    params = {"vs_currency": "usd", "days": days}
    raw = _get(f"/coins/{coin_id}/ohlc", params)

    # Format: [[timestamp_ms, open, high, low, close], ...]
    rows = []
    for item in raw:
        ts_ms = item[0]
        dt_str = datetime.fromtimestamp(
            ts_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M")
        rows.append({
            "coin_id":    coin_id,
            "timestamp":  ts_ms,
            "datetime":   dt_str,
            "open":       item[1],
            "high":       item[2],
            "low":        item[3],
            "close":      item[4],
        })

    logger.info("✅ CoinGecko OHLC | %s | %d mum | son %d gün",
                coin_id, len(rows), days)
    return rows


# ─────────────────────────────────────────────────────────────
#  Tüm Coinler İçin Anlık Fiyat (price_snapshots tablosuna)
# ─────────────────────────────────────────────────────────────

def fetch_simple_price() -> list[dict]:
    """
    CoinGecko /simple/price endpoint'inden anlık fiyatları çeker
    ve price_snapshots tablosuna kaydeder.
    """
    ids = ",".join(COINGECKO_IDS.values())
    params = {
        "ids":                 ids,
        "vs_currencies":       "usd",
        "include_24hr_change": True,
        "include_24hr_vol":    True,
    }
    raw = _get("/simple/price", params)

    results = []
    # Coin id → Binance sembolü eşleştirmesi
    id_to_symbol = {v: k for k, v in COINGECKO_IDS.items()}

    for coin_id, data in raw.items():
        base = id_to_symbol.get(coin_id)
        if base is None:
            logger.warning("⚠️  Bilinmeyen CoinGecko id: %s, atlanıyor.", coin_id)
            continue
        symbol = base + "USDT"
        snap = {
            "symbol":           symbol,
            "price":            data.get("usd", 0),
            "price_change_24h": None,
            "price_change_pct": data.get("usd_24h_change"),
            "volume_24h":       data.get("usd_24h_vol"),
            "high_24h":         None,
            "low_24h":          None,
            "source":           "coingecko",
        }
        insert_snapshot(snap)
        results.append(snap)
        logger.info("💰 [CoinGecko] %-10s $%.4f  (%+.2f%%)",
                    symbol,
                    snap["price"],
                    snap["price_change_pct"] or 0)

    log_fetch("coingecko", ids, "simple_price",
              "success", len(results))
    return results


# ─────────────────────────────────────────────────────────────
#  Bağlantı Testi
# ─────────────────────────────────────────────────────────────

def test_connection() -> bool:
    """CoinGecko API bağlantısını ve sağlık durumunu kontrol eder."""
    try:
        data = _get("/ping")
        logger.info("✅ CoinGecko API bağlantısı başarılı: %s",
                    data.get("gecko_says", "OK"))
        return True
    except Exception as e:
        logger.error("❌ CoinGecko bağlantı testi başarısız: %s", e)
        return False


# ─────────────────────────────────────────────────────────────
#  Tüm CoinGecko Verisini Çek
# ─────────────────────────────────────────────────────────────

def fetch_all_coingecko() -> None:
    """Tüm CoinGecko veri çekme işlemlerini sırayla yapar."""
    logger.info("=" * 60)
    logger.info("🦎 CoinGecko veri toplama başlıyor...")
    logger.info("=" * 60)

    if not test_connection():
        return

    # Piyasa verisi (detaylı)
    try:
        fetch_market_data()
    except Exception as e:
        logger.error("❌ Market data hatası: %s", e)

    time.sleep(2)  # Rate limit için bekle

    # Anlık fiyatlar (hızlı)
    try:
        fetch_simple_price()
    except Exception as e:
        logger.error("❌ Simple price hatası: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    fetch_all_coingecko()