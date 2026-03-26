#!/usr/bin/env python3
# ============================================================
#  main.py  —  Ana Çalıştırıcı
#  Kripto Para Trading Botu | Ahmet Yılmaz | 1-5. Hafta + AI Pipeline
#
#  Kullanım:
#    python main.py setup        → Veritabanı kur + bağlantı testleri
#    python main.py historical   → Tarihsel veri indir (tek seferlik)
#    python main.py realtime     → Gerçek zamanlı fiyat döngüsü (Ctrl+C ile dur)
#    python main.py all          → Tam kurulum: setup + historical + realtime
#    python main.py analyze      → Veri kalitesi analizi (2. Hafta)
#    python main.py indicators   → Teknik gösterge hesaplama (2. Hafta)
#    python main.py signals      → Alım/satım sinyal üretimi (3. Hafta)
#    python main.py visualize    → Grafik ve dashboard oluşturma (3. Hafta)
#    python main.py features     → Özellik mühendisliği (4. Hafta)
#    python main.py advfeatures  → Gelişmiş özellikler & importance (5. Hafta)
#    python main.py dashboard    → Plotly interaktif dashboard (5. Hafta)
#    python main.py aidata       → AI veri hazırlığı (yfinance + teknik ind.)
#    python main.py ailstm       → LSTM/GRU fiyat tahmin modeli eğitimi
#    python main.py aiml         → XGBoost/Random Forest sinyal eğitimi
#    python main.py aiensemble   → Ensemble (Stacking Meta-Model) eğitimi
#    python main.py airl         → Deep RL (PPO) ajan eğitimi
#    python main.py aibacktest   → Backtesting ve performans raporu
#    python main.py aiall        → Tüm AI pipeline'ı sırasıyla çalıştır
# ============================================================

import sys
import os
import time
import logging
import signal

from database import create_tables, get_db_stats, insert_indicators
from binance_collector import (
    test_connection as binance_test,
    fetch_all_historical,
    fetch_all_realtime,
)
from coingecko_collector import (
    test_connection as gecko_test,
    fetch_all_coingecko,
    fetch_simple_price,
)
from config import UPDATE_INTERVAL_SECONDS, SYMBOLS, INTERVALS
from data_processor import (
    analyze_all_data,
    load_klines_df,
    build_feature_matrix,
    add_advanced_features,
    correlation_analysis,
    add_volatility_features,
    add_volume_features,
    add_pattern_features,
    feature_importance_analysis,
)
from indicators import add_all_indicators
from signal_generator import generate_signals_all_symbols, get_signal_summary
from visualizer import generate_all_charts, plot_full_dashboard
from interactive_dashboard import generate_all_interactive

# ─────────────────────────────────────────────────────────────
#  Loglama Ayarı
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("crypto_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Graceful Shutdown
# ─────────────────────────────────────────────────────────────

_running = True

def _handle_sigint(sig, frame):
    global _running
    logger.info("\n⛔ Durdurma sinyali alındı. Bot kapatılıyor...")
    _running = False

signal.signal(signal.SIGINT, _handle_sigint)


# ─────────────────────────────────────────────────────────────
#  Komutlar
# ─────────────────────────────────────────────────────────────

def cmd_setup():
    """Veritabanını kur, API bağlantılarını test et."""
    print("\n" + "═" * 60)
    print("  🔧  KURULUM AŞAMASI")
    print("═" * 60)

    # 1. Veritabanı tabloları
    print("\n[1/3] Veritabanı şeması oluşturuluyor...")
    create_tables()

    # 2. Binance bağlantısı
    print("\n[2/3] Binance API bağlantısı test ediliyor...")
    binance_ok = binance_test()

    # 3. CoinGecko bağlantısı
    print("\n[3/3] CoinGecko API bağlantısı test ediliyor...")
    gecko_ok = gecko_test()

    print("\n" + "─" * 60)
    print(f"  Binance API  : {'✅ Bağlı' if binance_ok else '❌ Bağlantı yok'}")
    print(f"  CoinGecko    : {'✅ Bağlı' if gecko_ok else '❌ Bağlantı yok'}")
    print(f"  Veritabanı   : ✅ Hazır")
    print("─" * 60)

    if not (binance_ok or gecko_ok):
        print("\n⚠️  İnternet bağlantısını kontrol edin.")
    else:
        print("\n✅ Kurulum tamamlandı! Şimdi 'historical' komutuyla")
        print("   tarihsel veriyi indirebilirsiniz.")


def cmd_historical():
    """Tüm semboller ve interval'lar için tarihsel veri indir."""
    print("\n" + "═" * 60)
    print("  📥  TARİHSEL VERİ İNDİRME")
    print(f"     Semboller  : {list(SYMBOLS.keys())}")
    print(f"     Interval'lar: {INTERVALS}")
    print("═" * 60 + "\n")

    start = time.time()
    fetch_all_historical()
    elapsed = time.time() - start

    # Özet
    stats = get_db_stats()
    print("\n" + "═" * 60)
    print("  📊  ÖZET")
    print("─" * 60)
    print(f"  Toplam kline : {stats['klines']:,}")
    print(f"  Süre         : {elapsed:.1f} saniye")
    if stats.get("klines_detail"):
        print("\n  Sembol / Interval dağılımı:")
        for row in stats["klines_detail"]:
            print(f"    {row['symbol']:<12} {row['interval']:<6} → {row['cnt']:>6,} mum")
    print("═" * 60)


def cmd_realtime():
    """Gerçek zamanlı fiyat güncelleme döngüsü."""
    global _running
    print("\n" + "═" * 60)
    print("  🔄  GERÇEK ZAMANLI FİYAT DÖNGÜSÜ")
    print(f"     Güncelleme aralığı: {UPDATE_INTERVAL_SECONDS} saniye")
    print("     Durdurmak için Ctrl+C\n")
    print("═" * 60 + "\n")

    cycle = 0
    while _running:
        cycle += 1
        print(f"\n── Döngü #{cycle} ({'─' * 40})")

        # Binance anlık fiyatlar
        try:
            fetch_all_realtime()
        except Exception as e:
            logger.error("Binance realtime hatası: %s", e)

        # CoinGecko fiyatları (her 5 döngüde bir - rate limit önlemi)
        if cycle % 5 == 1:
            try:
                fetch_simple_price()
            except Exception as e:
                logger.error("CoinGecko fiyat hatası: %s", e)

        # Bekleme (kesintili sleep: durdurma sinyaline hızlı tepki)
        for _ in range(UPDATE_INTERVAL_SECONDS):
            if not _running:
                break
            time.sleep(1)

    print("\n✅ Gerçek zamanlı döngü kapatıldı.")


def cmd_all():
    """Tam kurulum: setup + historical + realtime."""
    cmd_setup()
    print()
    cmd_historical()

    # CoinGecko piyasa verisi
    print("\n🦎 CoinGecko piyasa verisi çekiliyor...")
    try:
        fetch_all_coingecko()
    except Exception as e:
        logger.error("CoinGecko genel hata: %s", e)

    print("\n" + "═" * 60)
    print("  Tarihsel veri indirildi!")
    print("  Gerçek zamanlı takip başlatılıyor...\n")
    cmd_realtime()


# ─────────────────────────────────────────────────────────────
#  2. Hafta Komutları: Analiz & Göstergeler
# ─────────────────────────────────────────────────────────────

def cmd_analyze():
    """Veri kalitesi analizi ve önişleme raporu."""
    print("\n" + "═" * 60)
    print("  📊  VERİ KALİTESİ ANALİZİ (2. Hafta)")
    print("═" * 60)

    analyze_all_data()

    # Veritabanı özeti
    stats = get_db_stats()
    print("\n" + "─" * 60)
    print("  🗃️  Veritabanı Özeti:")
    for table, count in stats.items():
        if table != "klines_detail":
            print(f"    {table:20s}: {count:,} kayıt")
    print("═" * 60)


def cmd_indicators():
    """Tüm semboller için teknik göstergeleri hesapla ve kaydet."""
    print("\n" + "═" * 60)
    print("  📈  TEKNİK GÖSTERGE HESAPLAMA (2. Hafta)")
    print(f"     Semboller  : {list(SYMBOLS.keys())}")
    print(f"     Interval'lar: {INTERVALS}")
    print("═" * 60 + "\n")

    import time as _time
    start = _time.time()
    total_saved = 0

    for name, symbol in SYMBOLS.items():
        for interval in INTERVALS:
            print(f"\n── {symbol} / {interval} {'─' * 35}")

            df = load_klines_df(symbol, interval)
            if df.empty:
                print("   ⚠️  Veri yok, atlanıyor.")
                continue

            # Göstergeleri hesapla
            df_ind = add_all_indicators(df)
            df_ind = df_ind.dropna()

            if df_ind.empty:
                print("   ⚠️  Yeterli veri yok (gösterge hesaplanamadı).")
                continue

            # Veritabanına kaydet
            rows = []
            for idx, row in df_ind.iterrows():
                rows.append({
                    "symbol":      symbol,
                    "interval":    interval,
                    "open_time":   int(row["open_time"]),
                    "rsi_14":      round(row.get("rsi_14", 0), 4) if row.get("rsi_14") else None,
                    "macd_line":   round(row.get("macd_line", 0), 4) if row.get("macd_line") else None,
                    "macd_signal": round(row.get("macd_signal", 0), 4) if row.get("macd_signal") else None,
                    "macd_hist":   round(row.get("macd_hist", 0), 4) if row.get("macd_hist") else None,
                    "sma_20":      round(row.get("sma_20", 0), 4) if row.get("sma_20") else None,
                    "ema_20":      round(row.get("ema_20", 0), 4) if row.get("ema_20") else None,
                    "bb_upper":    round(row.get("bb_upper", 0), 4) if row.get("bb_upper") else None,
                    "bb_middle":   round(row.get("bb_middle", 0), 4) if row.get("bb_middle") else None,
                    "bb_lower":    round(row.get("bb_lower", 0), 4) if row.get("bb_lower") else None,
                    "atr_14":      round(row.get("atr_14", 0), 4) if row.get("atr_14") else None,
                    "vwap":        round(row.get("vwap", 0), 4) if row.get("vwap") else None,
                })

            saved = insert_indicators(rows)
            total_saved += saved

            # Son değerleri göster
            last = df_ind.iloc[-1]
            print(f"   RSI(14)  : {last.get('rsi_14', 0):.2f}")
            print(f"   MACD     : {last.get('macd_line', 0):.4f}")
            print(f"   SMA(20)  : ${last.get('sma_20', 0):,.2f}")
            print(f"   BB Upper : ${last.get('bb_upper', 0):,.2f}")
            print(f"   BB Lower : ${last.get('bb_lower', 0):,.2f}")
            print(f"   ATR(14)  : ${last.get('atr_14', 0):,.2f}")
            print(f"   Kaydedilen: {saved} satır")

    elapsed = _time.time() - start
    print("\n" + "═" * 60)
    print(f"  ✅ Teknik göstergeler hesaplandı!")
    print(f"     Toplam kayıt: {total_saved:,}")
    print(f"     Süre       : {elapsed:.1f} saniye")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────
#  3. Hafta Komutları: Sinyal Üretimi & Görselleştirme
# ─────────────────────────────────────────────────────────────

def cmd_signals():
    """Tüm semboller için alım/satım sinyalleri üret ve kaydet."""
    print("\n" + "═" * 60)
    print("  🔔  SİNYAL ÜRETİMİ (3. Hafta)")
    print(f"     Semboller  : {list(SYMBOLS.keys())}")
    print(f"     Interval'lar: {INTERVALS}")
    print("═" * 60 + "\n")

    import time as _time
    start = _time.time()

    all_signals = generate_signals_all_symbols()

    elapsed = _time.time() - start

    # Son sinyal özeti
    print("\n" + "═" * 60)
    print("  📊  SON SİNYAL ÖZETİ")
    print("─" * 60)

    for name, symbol in SYMBOLS.items():
        for interval in INTERVALS:
            summary = get_signal_summary(symbol, interval,
                                         all_signals.get((symbol, interval)))
            if "status" in summary:
                print(f"  {symbol:10s} {interval:4s} : {summary['status']}")
            else:
                combined = summary['combined']
                strength = summary['strength']
                emoji = "🟢" if combined == "BUY" else ("🔴" if combined == "SELL" else "⚪")
                print(f"  {emoji} {symbol:10s} {interval:4s} : "
                      f"{combined:4s} (güç: {strength:.2f}) | "
                      f"RSI={summary['rsi_signal']:4s} "
                      f"MACD={summary['macd_signal']:4s} "
                      f"BB={summary['bb_signal']:4s} "
                      f"MA={summary['ma_signal']:4s}")

    print(f"\n  Süre: {elapsed:.1f} saniye")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────
#  4. Hafta Komutları: Özellik Mühendisliği
# ─────────────────────────────────────────────────────────────

def cmd_features():
    """Gelişmiş özellik mühendisliği ve korelasyon analizi (4. Hafta)."""
    print("\n" + "═" * 60)
    print("  🧬  ÖZELLİK MÜHENDİSLİĞİ (4. Hafta)")
    print(f"     Semboller  : {list(SYMBOLS.keys())}")
    print(f"     Interval'lar: {INTERVALS}")
    print("═" * 60 + "\n")

    import time as _time
    start = _time.time()

    for name, symbol in SYMBOLS.items():
        for interval in INTERVALS:
            print(f"\n── {symbol} / {interval} {'─' * 35}")

            # Özellik matrisi oluştur (gelişmiş özellikler dahil)
            df = build_feature_matrix(symbol, interval)
            if df.empty:
                print("   ⚠️  Veri yok, atlanıyor.")
                continue

            # Gelişmiş özellikleri göster
            print(f"   📐 Toplam özellik sayısı: {len(df.columns)}")
            print(f"   📊 Toplam satır sayısı : {len(df):,}")

            # Yeni 4. hafta özelliklerini kontrol et ve göster
            new_features = ["rsi_momentum", "macd_hist_change", "bb_width",
                           "rolling_mean_7", "rolling_std_7",
                           "rolling_mean_14", "rolling_std_14",
                           "rolling_mean_30", "rolling_std_30"]
            found = [f for f in new_features if f in df.columns]
            print(f"   🧬 Gelişmiş özellikler : {len(found)}/{len(new_features)}")

            if found and not df.empty:
                last = df.iloc[-1]
                if "rsi_momentum" in df.columns:
                    print(f"   RSI Momentum    : {last.get('rsi_momentum', 0):.4f}")
                if "macd_hist_change" in df.columns:
                    print(f"   MACD Hist Δ (%) : {last.get('macd_hist_change', 0):.4f}")
                if "bb_width" in df.columns:
                    print(f"   BB Genişliği    : {last.get('bb_width', 0):.4f}")
                if "rolling_mean_7" in df.columns:
                    print(f"   Rolling Mean(7) : ${last.get('rolling_mean_7', 0):,.2f}")
                if "rolling_mean_30" in df.columns:
                    print(f"   Rolling Mean(30): ${last.get('rolling_mean_30', 0):,.2f}")

            # Korelasyon analizi
            print(f"\n   📊 Korelasyon Analizi (hedef: close):")
            corr = correlation_analysis(df, target_col="close", top_n=5)
            if corr:
                print("   En yüksek pozitif korelasyon:")
                for feat, val in corr["top_positive"][:5]:
                    bar = "█" * int(abs(val) * 20)
                    print(f"     {feat:20s} : {val:+.4f} {bar}")
                print("   En yüksek negatif korelasyon:")
                for feat, val in corr["top_negative"][:5]:
                    bar = "░" * int(abs(val) * 20)
                    print(f"     {feat:20s} : {val:+.4f} {bar}")

    elapsed = _time.time() - start
    print("\n" + "═" * 60)
    print(f"  ✅ Özellik mühendisliği tamamlandı!")
    print(f"     Süre: {elapsed:.1f} saniye")
    print("═" * 60)


def cmd_visualize():
    """Tüm semboller için grafik ve dashboard oluştur."""
    print("\n" + "═" * 60)
    print("  🎨  GRAFİK OLUŞTURMA (3. Hafta)")
    print(f"     Semboller  : {list(SYMBOLS.keys())}")
    print(f"     Interval'lar: {INTERVALS}")
    print("═" * 60 + "\n")

    import time as _time
    start = _time.time()

    files = generate_all_charts()

    elapsed = _time.time() - start

    print("\n" + "═" * 60)
    print(f"  ✅ Grafik oluşturma tamamlandı!")
    print(f"     Toplam dosya: {len(files)}")
    print(f"     Süre       : {elapsed:.1f} saniye")
    if files:
        chart_dir = os.path.dirname(files[0])
        print(f"     Klasör     : {chart_dir}")
    print("═" * 60)
# ─────────────────────────────────────────────────────────────
#  5. Hafta Komutları: Gelişmiş Özellikler & İnteraktif Dashboard
# ─────────────────────────────────────────────────────────────

def cmd_advanced_features():
    """Gelişmiş özellik çıkarımı ve feature importance analizi (5. Hafta)."""
    print("\n" + "═" * 60)
    print("  🧠  GELİŞMİŞ ÖZELLİKLER & FEATURE IMPORTANCE (5. Hafta)")
    print(f"     Semboller   : {list(SYMBOLS.keys())}")
    print(f"     Interval'lar: {INTERVALS}")
    print("═" * 60 + "\n")

    import time as _time
    start = _time.time()

    for name, symbol in SYMBOLS.items():
        for interval in INTERVALS:
            print(f"\n── {symbol} / {interval} {'─' * 35}")

            # Özellik matrisi oluştur (5. Hafta özellikleri dahil)
            df = build_feature_matrix(symbol, interval)
            if df.empty:
                print("   ⚠️  Veri yok, atlanıyor.")
                continue

            # Özellik istatistikleri
            print(f"   📐 Toplam özellik sayısı: {len(df.columns)}")
            print(f"   📊 Toplam satır sayısı : {len(df):,}")

            # 5. Hafta özelliklerini kontrol et
            week5_features = [
                "atr_pct", "vol_std_5", "vol_std_10", "vol_std_21",
                "garman_klass_vol",
                "volume_ratio", "volume_momentum", "obv",
                "higher_high", "lower_low", "trend_strength",
                "dist_to_high_20", "dist_to_low_20",
            ]
            found = [f for f in week5_features if f in df.columns]
            print(f"   🧠 5. Hafta özellikleri : {len(found)}/{len(week5_features)}")

            # Son değerleri göster
            if found and not df.empty:
                last = df.iloc[-1]
                print("   ── Volatilite Özellikleri:")
                if "atr_pct" in df.columns:
                    print(f"     ATR (%)         : {last.get('atr_pct', 0):.4f}")
                if "garman_klass_vol" in df.columns:
                    print(f"     Garman-Klass    : {last.get('garman_klass_vol', 0):.6f}")
                if "vol_std_21" in df.columns:
                    print(f"     Vol Std (21)    : ${last.get('vol_std_21', 0):,.2f}")

                print("   ── Hacim Özellikleri:")
                if "volume_ratio" in df.columns:
                    print(f"     Volume Ratio    : {last.get('volume_ratio', 0):.4f}")
                if "volume_momentum" in df.columns:
                    print(f"     Volume Momentum : {last.get('volume_momentum', 0):.2f}%")
                if "obv" in df.columns:
                    print(f"     OBV             : {last.get('obv', 0):,.0f}")

                print("   ── Pattern Özellikleri:")
                if "higher_high" in df.columns:
                    print(f"     Higher High     : {'Evet ↑' if last.get('higher_high', 0) == 1 else 'Hayır'}")
                if "lower_low" in df.columns:
                    print(f"     Lower Low       : {'Evet ↓' if last.get('lower_low', 0) == 1 else 'Hayır'}")
                if "trend_strength" in df.columns:
                    ts = int(last.get('trend_strength', 0))
                    trend_emoji = '🟢' if ts > 0 else ('🔴' if ts < 0 else '⚪')
                    print(f"     Trend Strength  : {trend_emoji} {ts:+d} bar")
                if "dist_to_high_20" in df.columns:
                    print(f"     Zirveye Mesafe  : {last.get('dist_to_high_20', 0):.2f}%")
                if "dist_to_low_20" in df.columns:
                    print(f"     Dibe Mesafe     : {last.get('dist_to_low_20', 0):.2f}%")

            # Feature Importance analizi (sadece 1d ve BTC için detaylı)
            if interval == "1d":
                print(f"\n   📊 Feature Importance Analizi:")
                importance = feature_importance_analysis(df, target_col="close", top_n=10)
                if importance:
                    print("   Top 10 Özellik (Birleşik Sıralama):")
                    for feat, rank in importance["combined_rank"][:10]:
                        bar = "█" * max(1, int((1 - rank / 50) * 20))
                        print(f"     {feat:22s} : sıra {rank:5.1f} {bar}")

                    if importance.get("low_variance"):
                        print(f"   ⚠️  Düşük varyanslı özellikler: {importance['low_variance']}")

    elapsed = _time.time() - start
    print("\n" + "═" * 60)
    print(f"  ✅ Gelişmiş özellikler ve importance analizi tamamlandı!")
    print(f"     Süre: {elapsed:.1f} saniye")
    print("═" * 60)


def cmd_dashboard():
    """İnteraktif Plotly dashboard oluştur (5. Hafta)."""
    print("\n" + "═" * 60)
    print("  🌐  İNTERAKTİF DASHBOARD (5. Hafta)")
    print(f"     Semboller   : {list(SYMBOLS.keys())}")
    print(f"     Interval'lar: {INTERVALS}")
    print("═" * 60 + "\n")

    import time as _time
    start = _time.time()

    files = generate_all_interactive(include_importance=True)

    elapsed = _time.time() - start

    print("\n" + "═" * 60)
    print(f"  ✅ İnteraktif dashboard oluşturma tamamlandı!")
    print(f"     Toplam dosya: {len(files)}")
    print(f"     Süre       : {elapsed:.1f} saniye")
    if files:
        import os
        chart_dir = os.path.dirname(files[0])
        print(f"     Klasör     : {chart_dir}")
        print(f"\n  💡 HTML dosyalarını tarayıcınızda açarak interaktif")
        print(f"     olarak inceleyebilirsiniz.")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────
#  AI Pipeline Komutları (LSTM/GRU, XGBoost/RF, Ensemble, RL, Backtest)
# ─────────────────────────────────────────────────────────────

def _run_ai_script(script_rel_path, step_name):
    """Yardımcı: Bir AI pipeline scriptini subprocess ile çalıştırır."""
    import subprocess
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, *script_rel_path.split("/"))

    print("\n" + "═" * 60)
    print(f"  🤖  {step_name}")
    print(f"     Dosya: {script_path}")
    print("═" * 60 + "\n")

    if not os.path.exists(script_path):
        print(f"  ❌ HATA: {script_path} bulunamadı!")
        return False

    result = subprocess.run([sys.executable, script_path],
                           stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"\n  ❌ HATA: {step_name} başarısız oldu (kod: {result.returncode}).")
        return False

    print(f"\n  ✅ {step_name} tamamlandı.")
    return True


def cmd_ai_data():
    """AI veri hazırlığı: yfinance ile veri çek, teknik indikatör ekle, ölçekle."""
    _run_ai_script("src/data/data_pipeline.py",
                   "AI VERİ HAZIRLAMA (yfinance + Teknik İndikatörler + Ölçekleme)")


def cmd_ai_lstm():
    """LSTM ve GRU modellerini eğit (fiyat tahmini)."""
    _run_ai_script("src/models/time_series_models.py",
                   "LSTM / GRU MODELLERİ EĞİTİMİ (Fiyat Tahmini)")


def cmd_ai_ml():
    """Random Forest ve XGBoost modellerini eğit (sinyal üretimi)."""
    _run_ai_script("src/models/ml_classification_models.py",
                   "RANDOM FOREST / XGBOOST EĞİTİMİ (Sinyal Üretimi)")


def cmd_ai_ensemble():
    """Ensemble (Stacking Meta-Model) eğitimi."""
    _run_ai_script("src/models/ensemble_model.py",
                   "ENSEMBLE META-MODEL EĞİTİMİ (Stacking)")


def cmd_ai_rl():
    """Deep RL (PPO) trading ajanı eğitimi."""
    _run_ai_script("src/rl/train_agent.py",
                   "DEEP RL (PPO) AJAN EĞİTİMİ")


def cmd_ai_backtest():
    """Backtesting ve performans raporu."""
    _run_ai_script("src/utils/backtester.py",
                   "BACKTESTING & PERFORMANS RAPORU")


def cmd_ai_all():
    """Tüm AI pipeline aşamalarını sırasıyla çalıştırır."""
    print("\n" + "═" * 60)
    print("  🚀  TÜM AI PIPELINE BAŞLATILIYOR")
    print("═" * 60)

    steps = [
        ("src/data/data_pipeline.py",              "Aşama 1: AI Veri Hazırlama"),
        ("src/models/time_series_models.py",       "Aşama 2a: LSTM / GRU Eğitimi"),
        ("src/models/ml_classification_models.py", "Aşama 2b: RF / XGBoost Eğitimi"),
        ("src/models/ensemble_model.py",           "Aşama 3: Ensemble Meta-Model"),
        ("src/rl/train_agent.py",                  "Aşama 4: Deep RL (PPO) Ajan"),
        ("src/utils/backtester.py",                "Aşama 5: Backtesting"),
    ]

    for script, name in steps:
        success = _run_ai_script(script, name)
        if not success:
            print(f"\n  ⛔ Pipeline '{name}' aşamasında durdu.")
            return

    print("\n" + "═" * 60)
    print("  🎉 TÜM AI PIPELINE AŞAMALARI BAŞARIYLA TAMAMLANDI!")
    print("     Sonuçlar: data/results/ dizininde.")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────
#  Giriş Noktası
# ─────────────────────────────────────────────────────────────

COMMANDS = {
    "setup":        cmd_setup,
    "historical":   cmd_historical,
    "realtime":     cmd_realtime,
    "all":          cmd_all,
    "analyze":      cmd_analyze,
    "indicators":   cmd_indicators,
    "signals":      cmd_signals,
    "visualize":    cmd_visualize,
    "features":     cmd_features,
    "advfeatures":  cmd_advanced_features,
    "dashboard":    cmd_dashboard,
    # AI Pipeline komutları
    "aidata":       cmd_ai_data,
    "ailstm":       cmd_ai_lstm,
    "aiml":         cmd_ai_ml,
    "aiensemble":   cmd_ai_ensemble,
    "airl":         cmd_ai_rl,
    "aibacktest":   cmd_ai_backtest,
    "aiall":        cmd_ai_all,
}

BANNER = r"""
  ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗     ██████╗  ██████╗ ████████╗
 ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗    ██╔══██╗██╔═══██╗╚══██╔══╝
 ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║    ██████╔╝██║   ██║   ██║
 ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║    ██╔══██╗██║   ██║   ██║
 ╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝    ██████╔╝╚██████╔╝   ██║
  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝     ╚═════╝  ╚═════╝    ╚═╝

  Makine Öğrenmesi Tabanlı Kripto Para Trading Botu
  Ahmet Yılmaz • 23100011075 • AI Pipeline: LSTM/GRU, XGBoost/RF, Ensemble, Deep RL
"""

# ─────────────────────────────────────────────────────────────
#  İnteraktif Menü Sistemi
# ─────────────────────────────────────────────────────────────

MENU_ITEMS = [
    # (numara, komut_adı, açıklama, hafta_grubu)
    ("1",  "setup",        "Veritabanı kur + API bağlantı testleri",           "1. Hafta: Veri Toplama"),
    ("2",  "historical",   "Tarihsel kline verisi indir (Binance)",            None),
    ("3",  "realtime",     "Gerçek zamanlı fiyat döngüsü (Ctrl+C ile dur)",    None),
    ("4",  "all",          "Tümünü çalıştır (setup + historical + realtime)",  None),
    ("5",  "analyze",      "Veri kalitesi analizi ve istatistikler",           "2. Hafta: Analiz & Göstergeler"),
    ("6",  "indicators",   "Teknik gösterge hesaplama (RSI, MACD, BB...)",     None),
    ("7",  "signals",      "Alım/satım sinyal üretimi",                       "3. Hafta: Sinyal & Görselleştirme"),
    ("8",  "visualize",    "Matplotlib grafik ve dashboard oluştur",           None),
    ("9",  "features",     "Özellik mühendisliği & korelasyon analizi",        "4. Hafta: Özellik Mühendisliği"),
    ("10", "advfeatures",  "Volatilite, hacim, pattern & feature importance",  "5. Hafta: Gelişmiş Özellikler"),
    ("11", "dashboard",    "Plotly interaktif HTML dashboard oluştur",         None),
    ("12", "aidata",       "AI veri hazırlığı (yfinance + ölçekleme)",         "AI Pipeline: Derin Öğrenme & RL"),
    ("13", "ailstm",       "LSTM / GRU fiyat tahmin modeli eğitimi",           None),
    ("14", "aiml",         "XGBoost / Random Forest sinyal eğitimi",           None),
    ("15", "aiensemble",   "Ensemble (Stacking Meta-Model) eğitimi",           None),
    ("16", "airl",         "Deep RL (PPO) ajan eğitimi",                       None),
    ("17", "aibacktest",   "Backtesting ve performans raporu",                 None),
    ("18", "aiall",        "Tüm AI pipeline'ı sırasıyla çalıştır",            None),
]


def print_menu():
    """Numaralı interaktif menüyü ekrana yazdırır."""
    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║                    ANA MENÜ                             ║")
    print("  ╠══════════════════════════════════════════════════════════╣")

    for num, cmd_name, description, group in MENU_ITEMS:
        if group:
            print(f"  ║  ── {group} {'─' * (40 - len(group))}║")
        print(f"  ║  [{num:>2}]  {description:<49}║")

    print("  ║                                                        ║")
    print("  ║  [ 0]  Çıkış                                           ║")
    print("  ╚══════════════════════════════════════════════════════════╝")


def interactive_mode():
    """Kullanıcıdan numara alarak komut çalıştırır (döngülü menü)."""
    menu_map = {item[0]: item[1] for item in MENU_ITEMS}

    while True:
        print_menu()
        try:
            choice = input("\n  Seçiminiz (0-18): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 Güle güle!\n")
            break

        if choice == "0" or choice.lower() in ("q", "quit", "exit", "çıkış"):
            print("\n  👋 Güle güle!\n")
            break

        if choice not in menu_map:
            print(f"\n  ⚠️  Geçersiz seçim: '{choice}'")
            print("  💡 0-18 arası bir numara girin veya çıkmak için 0 yazın.")
            continue

        cmd_name = menu_map[choice]
        cmd_func = COMMANDS[cmd_name]

        # Seçilen komutun bilgisini göster
        item = next(i for i in MENU_ITEMS if i[0] == choice)
        print(f"\n  ▶  [{choice}] {item[2]}")
        print(f"  {'─' * 58}")

        try:
            cmd_func()
        except KeyboardInterrupt:
            print("\n\n  ⚠️  İşlem kullanıcı tarafından iptal edildi.")
        except Exception as e:
            print(f"\n  ❌ Hata: {e}")
            logger.exception("Komut çalıştırılırken hata oluştu: %s", cmd_name)

        print(f"\n  {'═' * 58}")
        input("  ↵  Ana menüye dönmek için Enter'a basın...")


if __name__ == "__main__":
    print(BANNER)

    # Doğrudan komut modu: python main.py <komut>
    if len(sys.argv) >= 2:
        cmd = sys.argv[1].lower()
        if cmd in COMMANDS:
            COMMANDS[cmd]()
        elif cmd in ("menu", "help", "--help", "-h"):
            interactive_mode()
        else:
            print(f"  ⚠️  Bilinmeyen komut: '{cmd}'\n")
            print("  Mevcut komutlar:")
            for num, name, desc, _ in MENU_ITEMS:
                print(f"    {name:<14} → {desc}")
            print(f"\n  💡 İnteraktif menü için: python main.py")
            sys.exit(1)
    else:
        # Argüman yok → interaktif menü aç
        interactive_mode()