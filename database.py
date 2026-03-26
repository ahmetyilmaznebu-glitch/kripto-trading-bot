# ============================================================
#  database.py  —  SQLite Veritabanı Şeması ve İşlemleri
#  Kripto Para Trading Botu | Ahmet Yılmaz | 1-4. Hafta
#
#  Temel İşlevler:
#    - Tablo oluşturma (OHLCV, göstergeler, sinyaller)
#    - Veri yazma ve okuma (INSERT, SELECT, UPDATE)
#    - Veritabanı istatistikleri
# ============================================================

import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager
from config import DB_PATH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Bağlantı Yönetimi
# ─────────────────────────────────────────────────────────────
# SQLite bağlantısı context manager ile yönetilir
# ─────────────────────────────────────────────────────────────

@contextmanager
def get_connection():
    """Thread-safe SQLite bağlantısı sağlar; hata olursa rollback yapar."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row          # Sonuçları dict gibi kullan
    conn.execute("PRAGMA journal_mode=WAL") # Eş zamanlı okuma/yazma için
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
#  Şema Oluşturma
# ─────────────────────────────────────────────────────────────

SCHEMA = """
-- ── Tablo 1: Kline (OHLCV) Verisi ──────────────────────────
CREATE TABLE IF NOT EXISTS klines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,          -- Örn: BTCUSDT
    interval        TEXT    NOT NULL,          -- Örn: 1h, 4h, 1d
    open_time       INTEGER NOT NULL,          -- Unix ms (Binance formatı)
    open_time_dt    TEXT    NOT NULL,          -- İnsan okunabilir tarih
    open            REAL    NOT NULL,
    high            REAL    NOT NULL,
    low             REAL    NOT NULL,
    close           REAL    NOT NULL,
    volume          REAL    NOT NULL,
    close_time      INTEGER NOT NULL,
    quote_volume    REAL,                      -- USDT cinsinden işlem hacmi
    trade_count     INTEGER,                   -- Periyottaki işlem sayısı
    taker_buy_vol   REAL,                      -- Alıcı tarafı hacmi
    created_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, interval, open_time)        -- Mükerrer kayıt önleme
);

-- ── Tablo 2: Gerçek Zamanlı Fiyat Anlık Görüntüleri ─────────
CREATE TABLE IF NOT EXISTS price_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    price           REAL    NOT NULL,
    price_change_24h REAL,                     -- 24s fiyat değişimi (USDT)
    price_change_pct REAL,                     -- 24s yüzde değişim
    volume_24h      REAL,                      -- 24s işlem hacmi
    high_24h        REAL,
    low_24h         REAL,
    source          TEXT    DEFAULT 'binance', -- binance | coingecko
    timestamp       TEXT    DEFAULT (datetime('now'))
);

-- ── Tablo 3: CoinGecko Piyasa Verisi ─────────────────────────
CREATE TABLE IF NOT EXISTS market_data (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    coin_id             TEXT    NOT NULL,      -- coingecko id (bitcoin vb.)
    symbol              TEXT    NOT NULL,
    current_price       REAL,
    market_cap          REAL,
    market_cap_rank     INTEGER,
    total_volume        REAL,
    high_24h            REAL,
    low_24h             REAL,
    price_change_24h    REAL,
    price_change_pct_24h REAL,
    circulating_supply  REAL,
    total_supply        REAL,
    ath                 REAL,                  -- All-time high
    ath_change_pct      REAL,
    last_updated        TEXT,
    fetched_at          TEXT    DEFAULT (datetime('now'))
);

-- ── Tablo 4: Veri Çekme Log Kaydı ───────────────────────────
CREATE TABLE IF NOT EXISTS fetch_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,   -- binance | coingecko
    symbol      TEXT    NOT NULL,
    action      TEXT    NOT NULL,   -- historical | realtime | market_data
    status      TEXT    NOT NULL,   -- success | error
    records     INTEGER DEFAULT 0,
    message     TEXT,
    timestamp   TEXT    DEFAULT (datetime('now'))
);

-- ── Tablo 5: Teknik Göstergeler (2. Hafta) ──────────────────
CREATE TABLE IF NOT EXISTS indicators (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    interval        TEXT    NOT NULL,
    open_time       INTEGER NOT NULL,
    rsi_14          REAL,
    macd_line       REAL,
    macd_signal     REAL,
    macd_hist       REAL,
    sma_20          REAL,
    ema_20          REAL,
    bb_upper        REAL,
    bb_middle       REAL,
    bb_lower        REAL,
    atr_14          REAL,
    vwap            REAL,
    calculated_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, interval, open_time)
);

-- ── Tablo 6: Alım/Satım Sinyalleri (3. Hafta) ──────────────
CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    interval     TEXT    NOT NULL,
    open_time    INTEGER NOT NULL,
    rsi_signal   TEXT,              -- BUY / SELL / HOLD
    macd_signal  TEXT,
    bb_signal    TEXT,
    ma_signal    TEXT,
    combined     TEXT,              -- Nihai karar
    strength     REAL,              -- Sinyal gücü (0-1)
    created_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, interval, open_time)
);

-- ── Tablo 7: ML Tahminleri (4. Hafta) ──────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    interval     TEXT    NOT NULL,
    model_type   TEXT    NOT NULL,
    prediction   INTEGER NOT NULL,      -- 1=UP, 0=DOWN
    probability  REAL,                  -- Tahmin olasılığı
    accuracy     REAL,                  -- Model doğruluk oranı
    created_at   TEXT    DEFAULT (datetime('now'))
);

-- ── İndeksler (Sorgu hızlandırma) ───────────────────────────
CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval
    ON klines (symbol, interval);
CREATE INDEX IF NOT EXISTS idx_klines_open_time
    ON klines (open_time);
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol
    ON price_snapshots (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_coin
    ON market_data (coin_id, fetched_at);
CREATE INDEX IF NOT EXISTS idx_indicators_symbol
    ON indicators (symbol, interval, open_time);
CREATE INDEX IF NOT EXISTS idx_signals_symbol
    ON signals (symbol, interval, open_time);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol
    ON predictions (symbol, interval, model_type);
"""


def create_tables():
    """Tüm tabloları ve indeksleri oluşturur (yoksa)."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    logger.info("✅ Veritabanı şeması hazır: %s", DB_PATH)


# ─────────────────────────────────────────────────────────────
#  Kline (OHLCV) İşlemleri
# ─────────────────────────────────────────────────────────────

def insert_klines(rows: list[dict]) -> int:
    """
    Kline listesini toplu olarak ekler; mükerrer kayıtları atlar.
    Her row: symbol, interval, open_time, open, high, low, close,
              volume, close_time, quote_volume, trade_count, taker_buy_vol
    """
    if not rows:
        return 0

    sql = """
        INSERT OR IGNORE INTO klines
            (symbol, interval, open_time, open_time_dt,
             open, high, low, close, volume,
             close_time, quote_volume, trade_count, taker_buy_vol)
        VALUES
            (:symbol, :interval, :open_time, :open_time_dt,
             :open, :high, :low, :close, :volume,
             :close_time, :quote_volume, :trade_count, :taker_buy_vol)
    """
    with get_connection() as conn:
        count_before = conn.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
        conn.executemany(sql, rows)
        count_after = conn.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
        inserted = count_after - count_before
    return inserted


def get_latest_kline_time(symbol: str, interval: str) -> int | None:
    """Belirli sembol/interval için en son open_time değerini döndürür."""
    sql = "SELECT MAX(open_time) FROM klines WHERE symbol=? AND interval=?"
    with get_connection() as conn:
        row = conn.execute(sql, (symbol, interval)).fetchone()
    return row[0] if row is not None and row[0] is not None else None


def get_klines(symbol: str, interval: str, limit: int = 500) -> list[dict]:
    """Son N adet kline'ı kronolojik sırada (eskiden yeniye) döndürür."""
    sql = """
        SELECT * FROM (
            SELECT * FROM klines
            WHERE symbol=? AND interval=?
            ORDER BY open_time DESC
            LIMIT ?
        ) sub ORDER BY open_time ASC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (symbol, interval, limit)).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
#  Anlık Fiyat İşlemleri
# ─────────────────────────────────────────────────────────────

def insert_snapshot(data: dict) -> None:
    """Anlık fiyat snapshot'ı ekler."""
    sql = """
        INSERT INTO price_snapshots
            (symbol, price, price_change_24h, price_change_pct,
             volume_24h, high_24h, low_24h, source)
        VALUES
            (:symbol, :price, :price_change_24h, :price_change_pct,
             :volume_24h, :high_24h, :low_24h, :source)
    """
    with get_connection() as conn:
        conn.execute(sql, data)


def get_latest_prices() -> list[dict]:
    """Her sembol için en son fiyat snapshot'ını döndürür (sembol başına tek satır)."""
    sql = """
        SELECT *
        FROM price_snapshots
        WHERE id IN (
            SELECT MAX(id) FROM price_snapshots GROUP BY symbol
        )
    """
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
#  CoinGecko Piyasa Verisi İşlemleri
# ─────────────────────────────────────────────────────────────

def insert_market_data(data: dict) -> None:
    """CoinGecko piyasa verisini ekler."""
    sql = """
        INSERT INTO market_data
            (coin_id, symbol, current_price, market_cap, market_cap_rank,
             total_volume, high_24h, low_24h, price_change_24h,
             price_change_pct_24h, circulating_supply, total_supply,
             ath, ath_change_pct, last_updated)
        VALUES
            (:coin_id, :symbol, :current_price, :market_cap, :market_cap_rank,
             :total_volume, :high_24h, :low_24h, :price_change_24h,
             :price_change_pct_24h, :circulating_supply, :total_supply,
             :ath, :ath_change_pct, :last_updated)
    """
    with get_connection() as conn:
        conn.execute(sql, data)


# ─────────────────────────────────────────────────────────────
#  Log İşlemleri
# ─────────────────────────────────────────────────────────────

def log_fetch(source: str, symbol: str, action: str,
              status: str, records: int = 0, message: str = "") -> None:
    """Veri çekme işlemini loglar."""
    sql = """
        INSERT INTO fetch_log (source, symbol, action, status, records, message)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (source, symbol, action, status, records, message))


# ─────────────────────────────────────────────────────────────
#  İstatistik / Özet
# ─────────────────────────────────────────────────────────────

def get_db_stats() -> dict:
    """Veritabanındaki kayıt sayılarını özetler."""
    stats = {}
    queries = {
        "klines":          "SELECT COUNT(*) FROM klines",
        "price_snapshots": "SELECT COUNT(*) FROM price_snapshots",
        "market_data":     "SELECT COUNT(*) FROM market_data",
        "fetch_log":       "SELECT COUNT(*) FROM fetch_log",
        "indicators":      "SELECT COUNT(*) FROM indicators",
        "signals":         "SELECT COUNT(*) FROM signals",
        "predictions":     "SELECT COUNT(*) FROM predictions",
    }
    with get_connection() as conn:
        for table, sql in queries.items():
            stats[table] = conn.execute(sql).fetchone()[0]

        # Sembol bazlı kline sayıları
        rows = conn.execute(
            "SELECT symbol, interval, COUNT(*) as cnt FROM klines GROUP BY symbol, interval"
        ).fetchall()
        stats["klines_detail"] = [dict(r) for r in rows]

    return stats


# ─────────────────────────────────────────────────────────────
#  Teknik Gösterge İşlemleri (2. Hafta)
# ─────────────────────────────────────────────────────────────

def insert_indicators(rows: list[dict]) -> int:
    """
    Hesaplanan teknik göstergeleri toplu olarak kaydeder.
    Mükerrer kayıtları günceller (UPSERT).
    """
    if not rows:
        return 0

    sql = """
        INSERT OR REPLACE INTO indicators
            (symbol, interval, open_time,
             rsi_14, macd_line, macd_signal, macd_hist,
             sma_20, ema_20,
             bb_upper, bb_middle, bb_lower,
             atr_14, vwap)
        VALUES
            (:symbol, :interval, :open_time,
             :rsi_14, :macd_line, :macd_signal, :macd_hist,
             :sma_20, :ema_20,
             :bb_upper, :bb_middle, :bb_lower,
             :atr_14, :vwap)
    """
    with get_connection() as conn:
        conn.executemany(sql, rows)
        inserted = len(rows)
    logger.info("📈 %d gösterge kaydı yazıldı", inserted)
    return inserted


def get_indicators(symbol: str, interval: str,
                   limit: int = 500) -> list[dict]:
    """Belirli sembol/interval için en son gösterge kayıtlarını getirir."""
    sql = """
        SELECT * FROM (
            SELECT * FROM indicators
            WHERE symbol=? AND interval=?
            ORDER BY open_time DESC
            LIMIT ?
        ) sub ORDER BY open_time ASC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (symbol, interval, limit)).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
#  Sinyal İşlemleri (3. Hafta)
# ─────────────────────────────────────────────────────────────

def insert_signals(rows: list[dict]) -> int:
    """
    Hesaplanan alım/satım sinyallerini toplu olarak kaydeder.
    Mükerrer kayıtları günceller (UPSERT).
    """
    if not rows:
        return 0

    sql = """
        INSERT OR REPLACE INTO signals
            (symbol, interval, open_time,
             rsi_signal, macd_signal, bb_signal, ma_signal,
             combined, strength)
        VALUES
            (:symbol, :interval, :open_time,
             :rsi_signal, :macd_signal, :bb_signal, :ma_signal,
             :combined, :strength)
    """
    with get_connection() as conn:
        conn.executemany(sql, rows)
        inserted = len(rows)
    logger.info("🔔 %d sinyal kaydı yazıldı", inserted)
    return inserted


def get_signals(symbol: str, interval: str,
                limit: int = 500) -> list[dict]:
    """Belirli sembol/interval için en son sinyal kayıtlarını getirir."""
    sql = """
        SELECT * FROM (
            SELECT * FROM signals
            WHERE symbol=? AND interval=?
            ORDER BY open_time DESC
            LIMIT ?
        ) sub ORDER BY open_time ASC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (symbol, interval, limit)).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
#  Tahmin İşlemleri (4. Hafta)
# ─────────────────────────────────────────────────────────────

def insert_prediction(data: dict) -> int:
    """
    ML model tahminini veritabanına kaydeder.
    """
    sql = """
        INSERT INTO predictions
            (symbol, interval, model_type, prediction, probability, accuracy)
        VALUES
            (:symbol, :interval, :model_type, :prediction, :probability, :accuracy)
    """
    with get_connection() as conn:
        conn.execute(sql, data)
    logger.info("🔮 Tahmin kaydedildi: %s %s %s → %s",
                data['symbol'], data['interval'], data['model_type'],
                'UP' if data['prediction'] == 1 else 'DOWN')
    return 1


def get_predictions(symbol: str, interval: str,
                    limit: int = 50) -> list[dict]:
    """Belirli sembol/interval için son tahminleri getirir."""
    sql = """
        SELECT * FROM predictions
        WHERE symbol=? AND interval=?
        ORDER BY created_at DESC
        LIMIT ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (symbol, interval, limit)).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    create_tables()
    stats = get_db_stats()
    print("\n📊 Veritabanı İstatistikleri:")
    for k, v in stats.items():
        if k != "klines_detail":
            print(f"   {k:20s}: {v} kayıt")