# ============================================================
#  visualizer.py  —  Görselleştirme Modülü
#  Kripto Para Trading Botu | Ahmet Yılmaz | 3. Hafta
#
#  Grafikler:
#    1. Fiyat + Bollinger Bands + SMA/EMA
#    2. RSI grafiği (30/70 çizgileriyle)
#    3. MACD grafiği (histogram + line + signal)
#    4. Hacim bar grafiği
#    5. Alım/Satım sinyal işaretleri
#    6. Tam dashboard (tüm paneller)
#    7. Portföy özet karşılaştırma
#
#  Bu modül matplotlib kullanarak teknik göstergeleri
#  ve alım/satım sinyallerini görsel olarak sunar.
#  Raporlar PNG ve HTML formatında kaydedilir.
# ============================================================

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # GUI olmadan çalış
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

from config import SYMBOLS, INTERVALS, REPORTS_DIR
from data_processor import load_klines_df, clean_dataframe
from indicators import add_all_indicators
from signal_generator import generate_combined_signal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Ortak Stil Ayarları
# ─────────────────────────────────────────────────────────────

DARK_BG      = "#1a1a2e"
PANEL_BG     = "#16213e"
GRID_COLOR   = "#2a2a4a"
TEXT_COLOR   = "#e0e0e0"
ACCENT_GREEN = "#00d4aa"
ACCENT_RED   = "#ff4757"
ACCENT_BLUE  = "#3742fa"
ACCENT_GOLD  = "#ffa502"
BB_FILL      = "#3742fa20"

def _apply_style():
    """Tüm grafiklere uygulanacak karanlık tema."""
    plt.rcParams.update({
        "figure.facecolor":    DARK_BG,
        "axes.facecolor":      PANEL_BG,
        "axes.edgecolor":      GRID_COLOR,
        "axes.labelcolor":     TEXT_COLOR,
        "axes.grid":           True,
        "grid.color":          GRID_COLOR,
        "grid.alpha":          0.3,
        "xtick.color":         TEXT_COLOR,
        "ytick.color":         TEXT_COLOR,
        "text.color":          TEXT_COLOR,
        "font.size":           10,
        "legend.facecolor":    PANEL_BG,
        "legend.edgecolor":    GRID_COLOR,
        "legend.fontsize":     8,
    })

# ── Aktif oturum çıktı dizini (generate_all_charts tarafından ayarlanır) ──
_SESSION_DIR: str | None = None


def _get_charts_dir() -> str:
    """Mevcut oturum dizinini döndürür; yoksa varsayılanı oluşturur."""
    if _SESSION_DIR:
        return _SESSION_DIR
    fallback = os.path.join(REPORTS_DIR, "charts")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _save_figure(fig, filename: str, output_dir: str | None = None) -> str:
    """Figürü belirtilen veya oturum dizinine kaydeder."""
    target = output_dir or _get_charts_dir()
    os.makedirs(target, exist_ok=True)
    filepath = os.path.join(target, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    logger.info("💾 Grafik kaydedildi: %s", filepath)
    return filepath


# ─────────────────────────────────────────────────────────────
#  1. Fiyat + Bollinger Bands + SMA/EMA Grafiği
# ─────────────────────────────────────────────────────────────

def plot_price_with_indicators(df: pd.DataFrame,
                               symbol: str,
                               interval: str,
                               last_n: int = 100) -> str:
    """
    Kapanış fiyatını Bollinger Bands ve hareketli ortalamalarla çizer.

    Args:
        df: Göstergeler eklenmiş DataFrame
        symbol: İşlem çifti (BTCUSDT)
        interval: Zaman dilimi (1h, 4h)
        last_n: Son kaç mum gösterilsin

    Returns:
        Kaydedilen dosyanın yolu
    """
    _apply_style()
    df_plot = df.tail(last_n).copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(df_plot))

    # Kapanış fiyatı
    ax.plot(x, df_plot["close"], color=TEXT_COLOR, linewidth=1.5,
            label="Kapanış", zorder=5)

    # Bollinger Bands
    if "bb_upper" in df_plot.columns:
        ax.plot(x, df_plot["bb_upper"], color=ACCENT_RED, linewidth=0.8,
                linestyle="--", alpha=0.7, label="BB Üst")
        ax.plot(x, df_plot["bb_lower"], color=ACCENT_GREEN, linewidth=0.8,
                linestyle="--", alpha=0.7, label="BB Alt")
        ax.fill_between(x, df_plot["bb_upper"], df_plot["bb_lower"],
                        color=ACCENT_BLUE, alpha=0.08, label="BB Bandı")

    # SMA & EMA
    if "sma_20" in df_plot.columns:
        ax.plot(x, df_plot["sma_20"], color=ACCENT_GOLD, linewidth=1,
                alpha=0.8, label="SMA(20)")
    if "ema_20" in df_plot.columns:
        ax.plot(x, df_plot["ema_20"], color=ACCENT_BLUE, linewidth=1,
                alpha=0.8, label="EMA(20)")

    ax.set_title(f"{symbol} — Fiyat & Göstergeler ({interval})",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Fiyat (USDT)")
    ax.legend(loc="upper left", framealpha=0.8)

    # X ekseni etiketleri (her 10 barda tarih)
    tick_positions = list(range(0, len(df_plot), max(len(df_plot) // 8, 1)))
    tick_labels = [str(df_plot.index[i])[:10] if hasattr(df_plot.index[i], 'strftime')
                   else str(i) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    filename = f"{symbol}_{interval}_price.png"
    return _save_figure(fig, filename)


# ─────────────────────────────────────────────────────────────
#  2. RSI Grafiği
# ─────────────────────────────────────────────────────────────

def plot_rsi(df: pd.DataFrame,
             symbol: str,
             interval: str,
             last_n: int = 100) -> str:
    """RSI grafiği: 30/70 eşikleri ve renklendirilmiş alanlar."""
    _apply_style()
    df_plot = df.tail(last_n).copy()

    fig, ax = plt.subplots(figsize=(14, 3))

    x = range(len(df_plot))
    rsi_vals = df_plot["rsi_14"]

    # RSI çizgisi
    ax.plot(x, rsi_vals, color=ACCENT_GOLD, linewidth=1.2, label="RSI(14)")

    # Eşik çizgileri
    ax.axhline(70, color=ACCENT_RED, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(30, color=ACCENT_GREEN, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(50, color=TEXT_COLOR, linestyle=":", linewidth=0.5, alpha=0.4)

    # Overbought / Oversold renkli alanlar
    ax.fill_between(x, 70, 100, color=ACCENT_RED, alpha=0.1)
    ax.fill_between(x, 0, 30, color=ACCENT_GREEN, alpha=0.1)

    ax.set_ylim(0, 100)
    ax.set_title(f"{symbol} — RSI(14) ({interval})",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("RSI")
    ax.legend(loc="upper right")

    tick_positions = list(range(0, len(df_plot), max(len(df_plot) // 8, 1)))
    tick_labels = [str(df_plot.index[i])[:10] if hasattr(df_plot.index[i], 'strftime')
                   else str(i) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    filename = f"{symbol}_{interval}_rsi.png"
    return _save_figure(fig, filename)


# ─────────────────────────────────────────────────────────────
#  3. MACD Grafiği
# ─────────────────────────────────────────────────────────────

def plot_macd(df: pd.DataFrame,
              symbol: str,
              interval: str,
              last_n: int = 100) -> str:
    """MACD grafiği: MACD line, signal line ve histogram."""
    _apply_style()
    df_plot = df.tail(last_n).copy()

    fig, ax = plt.subplots(figsize=(14, 3))

    x = range(len(df_plot))
    macd_line   = df_plot["macd_line"]
    macd_signal = df_plot["macd_signal"]
    macd_hist   = df_plot["macd_hist"]

    # Histogram (yeşil/kırmızı renkli)
    colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED for v in macd_hist]
    ax.bar(x, macd_hist, color=colors, alpha=0.5, width=0.8, label="Histogram")

    # MACD ve Signal çizgileri
    ax.plot(x, macd_line, color=ACCENT_BLUE, linewidth=1.2, label="MACD")
    ax.plot(x, macd_signal, color=ACCENT_RED, linewidth=1, linestyle="--",
            label="Signal")

    ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)

    ax.set_title(f"{symbol} — MACD ({interval})",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("MACD")
    ax.legend(loc="upper left")

    tick_positions = list(range(0, len(df_plot), max(len(df_plot) // 8, 1)))
    tick_labels = [str(df_plot.index[i])[:10] if hasattr(df_plot.index[i], 'strftime')
                   else str(i) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    filename = f"{symbol}_{interval}_macd.png"
    return _save_figure(fig, filename)


# ─────────────────────────────────────────────────────────────
#  4. Hacim Grafiği
# ─────────────────────────────────────────────────────────────

def plot_volume(df: pd.DataFrame,
                symbol: str,
                interval: str,
                last_n: int = 100) -> str:
    """Hacim bar grafiği: yeşil (yükseliş) / kırmızı (düşüş)."""
    _apply_style()
    df_plot = df.tail(last_n).copy()

    fig, ax = plt.subplots(figsize=(14, 2.5))

    x = range(len(df_plot))
    colors = [ACCENT_GREEN if c >= o else ACCENT_RED
              for c, o in zip(df_plot["close"], df_plot["open"])]

    ax.bar(x, df_plot["volume"], color=colors, alpha=0.6, width=0.8)

    ax.set_title(f"{symbol} — İşlem Hacmi ({interval})",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Hacim")

    # Hacim formatı
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    tick_positions = list(range(0, len(df_plot), max(len(df_plot) // 8, 1)))
    tick_labels = [str(df_plot.index[i])[:10] if hasattr(df_plot.index[i], 'strftime')
                   else str(i) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    filename = f"{symbol}_{interval}_volume.png"
    return _save_figure(fig, filename)


# ─────────────────────────────────────────────────────────────
#  5. Alım/Satım Sinyalleri Grafiği
# ─────────────────────────────────────────────────────────────

def plot_signals(df: pd.DataFrame,
                 signals_df: pd.DataFrame,
                 symbol: str,
                 interval: str,
                 last_n: int = 100) -> str:
    """
    Fiyat grafiği üzerinde alım (▲ yeşil) ve satım (▼ kırmızı) işaretleri.
    """
    _apply_style()

    df_plot = df.tail(last_n).copy()
    sig_plot = signals_df.tail(last_n).copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(df_plot))

    # Fiyat çizgisi
    ax.plot(x, df_plot["close"].values, color=TEXT_COLOR, linewidth=1.2,
            label="Kapanış", zorder=3)

    # Sinyal işaretleri
    buy_mask  = sig_plot["combined"].values == "BUY"
    sell_mask = sig_plot["combined"].values == "SELL"

    if buy_mask.any():
        ax.scatter(x[buy_mask], df_plot["close"].values[buy_mask],
                   marker="^", color=ACCENT_GREEN, s=80, zorder=5,
                   label=f"AL ({buy_mask.sum()})", edgecolors="white",
                   linewidth=0.5)

    if sell_mask.any():
        ax.scatter(x[sell_mask], df_plot["close"].values[sell_mask],
                   marker="v", color=ACCENT_RED, s=80, zorder=5,
                   label=f"SAT ({sell_mask.sum()})", edgecolors="white",
                   linewidth=0.5)

    ax.set_title(f"{symbol} — Alım/Satım Sinyalleri ({interval})",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Fiyat (USDT)")
    ax.legend(loc="upper left", framealpha=0.8)

    tick_positions = list(range(0, len(df_plot), max(len(df_plot) // 8, 1)))
    tick_labels = [str(df_plot.index[i])[:10] if hasattr(df_plot.index[i], 'strftime')
                   else str(i) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    filename = f"{symbol}_{interval}_signals.png"
    return _save_figure(fig, filename)


# ─────────────────────────────────────────────────────────────
#  6. Tam Dashboard (Tüm Paneller)
# ─────────────────────────────────────────────────────────────

def _draw_price_panel(ax, x, df_plot, sig_plot):
    """Fiyat + BB + MA + Sinyal paneli çizer."""
    ax.plot(x, df_plot["close"].values, color=TEXT_COLOR, linewidth=1.5,
            label="Kapanış", zorder=5)

    if "bb_upper" in df_plot.columns:
        ax.plot(x, df_plot["bb_upper"].values, color=ACCENT_RED, linewidth=0.7,
                linestyle="--", alpha=0.6)
        ax.plot(x, df_plot["bb_lower"].values, color=ACCENT_GREEN, linewidth=0.7,
                linestyle="--", alpha=0.6)
        ax.fill_between(x, df_plot["bb_upper"].values, df_plot["bb_lower"].values,
                        color=ACCENT_BLUE, alpha=0.08)

    if "sma_20" in df_plot.columns:
        ax.plot(x, df_plot["sma_20"].values, color=ACCENT_GOLD, linewidth=0.8,
                alpha=0.7, label="SMA(20)")
    if "ema_20" in df_plot.columns:
        ax.plot(x, df_plot["ema_20"].values, color=ACCENT_BLUE, linewidth=0.8,
                alpha=0.7, label="EMA(20)")

    buy_mask = sig_plot["combined"].values == "BUY"
    sell_mask = sig_plot["combined"].values == "SELL"
    if buy_mask.any():
        ax.scatter(x[buy_mask], df_plot["close"].values[buy_mask],
                   marker="^", color=ACCENT_GREEN, s=60, zorder=6,
                   edgecolors="white", linewidth=0.5)
    if sell_mask.any():
        ax.scatter(x[sell_mask], df_plot["close"].values[sell_mask],
                   marker="v", color=ACCENT_RED, s=60, zorder=6,
                   edgecolors="white", linewidth=0.5)

    ax.set_ylabel("Fiyat (USDT)")
    ax.legend(loc="upper left", framealpha=0.7, fontsize=8)


def _draw_rsi_panel(ax, x, df_plot):
    """RSI paneli çizer."""
    ax.plot(x, df_plot["rsi_14"].values, color=ACCENT_GOLD, linewidth=1)
    ax.axhline(70, color=ACCENT_RED, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axhline(30, color=ACCENT_GREEN, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.fill_between(x, 70, 100, color=ACCENT_RED, alpha=0.08)
    ax.fill_between(x, 0, 30, color=ACCENT_GREEN, alpha=0.08)
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI(14)")


def _draw_macd_panel(ax, x, df_plot):
    """MACD paneli çizer."""
    macd_hist = df_plot["macd_hist"].values
    hist_colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED for v in macd_hist]
    ax.bar(x, macd_hist, color=hist_colors, alpha=0.5, width=0.8)
    ax.plot(x, df_plot["macd_line"].values, color=ACCENT_BLUE, linewidth=1)
    ax.plot(x, df_plot["macd_signal"].values, color=ACCENT_RED, linewidth=0.8,
            linestyle="--")
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.4, alpha=0.4)
    ax.set_ylabel("MACD")


def _draw_volume_panel(ax, x, df_plot):
    """Hacim paneli çizer."""
    vol_colors = [ACCENT_GREEN if c >= o else ACCENT_RED
                  for c, o in zip(df_plot["close"].values, df_plot["open"].values)]
    ax.bar(x, df_plot["volume"].values, color=vol_colors, alpha=0.5, width=0.8)
    ax.set_ylabel("Hacim")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))


def _draw_strength_panel(ax, x, sig_plot, df_plot):
    """Sinyal gücü paneli çizer."""
    strength = sig_plot["strength"].values
    str_colors = []
    for i, val in enumerate(strength):
        combined = sig_plot["combined"].values[i]
        if combined == "BUY":
            str_colors.append(ACCENT_GREEN)
        elif combined == "SELL":
            str_colors.append(ACCENT_RED)
        else:
            str_colors.append(GRID_COLOR)

    ax.bar(x, strength, color=str_colors, alpha=0.6, width=0.8)
    ax.axhline(0.3, color=TEXT_COLOR, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Güç")
    ax.set_xlabel("Zaman")

    tick_positions = list(range(0, len(df_plot), max(len(df_plot) // 10, 1)))
    tick_labels = [str(df_plot.index[i])[:10] if hasattr(df_plot.index[i], 'strftime')
                   else str(i) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)


def plot_full_dashboard(symbol: str,
                        interval: str,
                        last_n: int = 100) -> str | None:
    """
    Tek bir figürde 5 alt panel:
      [1] Fiyat + BB + MA + Sinyaller
      [2] RSI
      [3] MACD
      [4] Hacim
      [5] Sinyal gücü

    Returns:
        Kaydedilen dosyanın yolu veya None
    """
    _apply_style()

    df = load_klines_df(symbol, interval)
    if df.empty:
        logger.warning("⚠️  %s %s: Veri yok, dashboard oluşturulamadı.", symbol, interval)
        return None

    df = clean_dataframe(df)
    df = add_all_indicators(df)
    df = df.dropna()

    if len(df) < 30:
        logger.warning("⚠️  %s %s: Yeterli veri yok (%d satır).", symbol, interval, len(df))
        return None

    signals = generate_combined_signal(df)
    df_plot = df.tail(last_n).copy()
    sig_plot = signals.tail(last_n).copy()
    x = np.arange(len(df_plot))

    fig, axes = plt.subplots(5, 1, figsize=(16, 14),
                             gridspec_kw={"height_ratios": [4, 1.5, 1.5, 1, 1]},
                             sharex=True)

    fig.suptitle(f"📊 {symbol} — Tam Dashboard ({interval})",
                 fontsize=16, fontweight="bold", y=0.98, color=ACCENT_GOLD)

    _draw_price_panel(axes[0], x, df_plot, sig_plot)
    _draw_rsi_panel(axes[1], x, df_plot)
    _draw_macd_panel(axes[2], x, df_plot)
    _draw_volume_panel(axes[3], x, df_plot)
    _draw_strength_panel(axes[4], x, sig_plot, df_plot)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"{symbol}_{interval}_dashboard.png"
    return _save_figure(fig, filename)


# ─────────────────────────────────────────────────────────────
#  7. Portföy Özet Karşılaştırma
# ─────────────────────────────────────────────────────────────

def plot_portfolio_summary(interval: str = "1d") -> str | None:
    """
    Tüm coinlerin son sinyal ve RSI durumunu karşılaştıran özet grafik.

    Yatay bar chart: her coin için RSI değeri, renklendirme sinyal bazlı.
    """
    _apply_style()

    data = []
    for name, symbol in SYMBOLS.items():
        df = load_klines_df(symbol, interval)
        if df.empty:
            continue

        df = clean_dataframe(df)
        df = add_all_indicators(df)
        df = df.dropna()

        if df.empty:
            continue

        signals = generate_combined_signal(df)

        last_row = df.iloc[-1]
        last_sig = signals.iloc[-1]

        data.append({
            "symbol":   name,
            "price":    last_row["close"],
            "rsi":      last_row["rsi_14"],
            "signal":   last_sig["combined"],
            "strength": last_sig["strength"],
        })

    if not data:
        logger.warning("⚠️  Portföy özeti için yeterli veri yok.")
        return None

    # ── Grafik ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fig.suptitle("📊 Portföy Özeti — Sinyal Karşılaştırma",
                 fontsize=14, fontweight="bold", y=1.02, color=ACCENT_GOLD)

    # Sol: RSI karşılaştırma
    ax1 = axes[0]
    symbols = [d["symbol"] for d in data]
    rsi_vals = [d["rsi"] for d in data]
    rsi_colors = []
    for d in data:
        if d["signal"] == "BUY":
            rsi_colors.append(ACCENT_GREEN)
        elif d["signal"] == "SELL":
            rsi_colors.append(ACCENT_RED)
        else:
            rsi_colors.append(ACCENT_BLUE)

    bars = ax1.barh(symbols, rsi_vals, color=rsi_colors, alpha=0.7, height=0.5)

    ax1.axvline(30, color=ACCENT_GREEN, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.axvline(70, color=ACCENT_RED, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("RSI(14)")
    ax1.set_title("RSI Durumu", fontsize=12, fontweight="bold")

    # Değer etiketleri
    for bar, val in zip(bars, rsi_vals):
        ax1.text(val + 2, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=10, color=TEXT_COLOR)

    # Sağ: Sinyal gücü
    ax2 = axes[1]
    strengths = [d["strength"] for d in data]
    str_bars = ax2.barh(symbols, strengths, color=rsi_colors, alpha=0.7, height=0.5)

    ax2.axvline(0.3, color=TEXT_COLOR, linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Sinyal Gücü")
    ax2.set_title("Sinyal Gücü & Karar", fontsize=12, fontweight="bold")

    # Sinyal etiketi
    for bar, d in zip(str_bars, data):
        label = d["signal"]
        color = ACCENT_GREEN if label == "BUY" else (ACCENT_RED if label == "SELL" else TEXT_COLOR)
        ax2.text(d["strength"] + 0.03, bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=10, color=color, fontweight="bold")

    fig.tight_layout()
    filename = "portfolio_summary.png"
    return _save_figure(fig, filename)


# ─────────────────────────────────────────────────────────────
#  Tüm Grafikleri Oluştur
# ─────────────────────────────────────────────────────────────

def generate_all_charts() -> list[str]:
    """
    Tüm sembol/interval kombinasyonları için dashboard +
    portföy özeti oluşturur.

    Her çalıştırma tarih damgalı bir alt klasöre kaydedilir:
      reports/charts/2026-02-24_170530/

    Returns:
        Oluşturulan dosya yollarının listesi
    """
    global _SESSION_DIR

    # Tarih damgalı oturum klasörü oluştur
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    _SESSION_DIR = os.path.join(REPORTS_DIR, "charts", timestamp)
    os.makedirs(_SESSION_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🎨 GRAFİK OLUŞTURMA BAŞLIYOR")
    logger.info("   Çıktı dizini: %s", _SESSION_DIR)
    logger.info("=" * 60)

    created_files = []

    for name, symbol in SYMBOLS.items():
        for interval in INTERVALS:
            print(f"\n── {symbol} / {interval} {'─' * 35}")

            result = plot_full_dashboard(symbol, interval)
            if result:
                created_files.append(result)
                print(f"   ✅ Dashboard: {result}")
            else:
                print(f"   ⚠️  Dashboard oluşturulamadı.")

    # Portföy özeti
    print(f"\n── Portföy Özeti {'─' * 40}")
    summary = plot_portfolio_summary()
    if summary:
        created_files.append(summary)
        print(f"   ✅ Özet: {summary}")

    logger.info("=" * 60)
    logger.info("✅ Grafik oluşturma tamamlandı: %d dosya", len(created_files))
    logger.info("   Klasör: %s", _SESSION_DIR)
    logger.info("=" * 60)

    _SESSION_DIR = None  # Oturumu sıfırla
    return created_files


# ─────────────────────────────────────────────────────────────
#  Tek Başına Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n🎨 Görselleştirme Testi")
    print("=" * 60)
    generate_all_charts()
