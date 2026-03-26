# ============================================================
#  interactive_dashboard.py  —  İnteraktif Dashboard Modülü
#  Kripto Para Trading Botu | Ahmet Yılmaz | 5. Hafta
#
#  Plotly ile oluşturulan interaktif HTML dashboard:
#    1. Candlestick grafiği + Bollinger Bands + SMA/EMA
#    2. RSI alt paneli (30/70 eşikleri)
#    3. MACD alt paneli (histogram + line + signal)
#    4. Hacim alt paneli (renkli bar chart)
#    5. Feature Importance bar grafiği
#
#  Çıktı: reports/ klasörüne HTML dosyası olarak kaydedilir.
# ============================================================

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import SYMBOLS, INTERVALS, REPORTS_DIR
from data_processor import (
    load_klines_df, clean_dataframe,
    build_feature_matrix, feature_importance_analysis,
)
from indicators import add_all_indicators
from signal_generator import generate_combined_signal

logger = logging.getLogger(__name__)


# ── Renk Paleti (karanlık tema) ──────────────────────────────

COLORS = {
    "bg":       "#1a1a2e",
    "panel":    "#16213e",
    "grid":     "#2a2a4a",
    "text":     "#e0e0e0",
    "green":    "#00d4aa",
    "red":      "#ff4757",
    "blue":     "#3742fa",
    "gold":     "#ffa502",
    "purple":   "#a855f7",
    "bb_fill":  "rgba(55, 66, 250, 0.08)",
}


# ─────────────────────────────────────────────────────────────
#  1. Tam İnteraktif Dashboard
# ─────────────────────────────────────────────────────────────

def _add_price_panel(fig, df_plot, sig_plot, dates):
    """Candlestick + Bollinger Bands + SMA/EMA + Sinyal işaretleri paneli."""
    fig.add_trace(go.Candlestick(
        x=dates, open=df_plot["open"], high=df_plot["high"],
        low=df_plot["low"], close=df_plot["close"],
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        name="OHLC", showlegend=True,
    ), row=1, col=1)

    if "bb_upper" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df_plot["bb_upper"],
            line=dict(color=COLORS["red"], width=1, dash="dash"),
            name="BB Üst", opacity=0.6,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=df_plot["bb_lower"],
            line=dict(color=COLORS["green"], width=1, dash="dash"),
            name="BB Alt", opacity=0.6,
            fill="tonexty", fillcolor=COLORS["bb_fill"],
        ), row=1, col=1)

    if "sma_20" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df_plot["sma_20"],
            line=dict(color=COLORS["gold"], width=1),
            name="SMA(20)", opacity=0.8,
        ), row=1, col=1)
    if "ema_20" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df_plot["ema_20"],
            line=dict(color=COLORS["blue"], width=1),
            name="EMA(20)", opacity=0.8,
        ), row=1, col=1)

    buy_mask = sig_plot["combined"] == "BUY"
    sell_mask = sig_plot["combined"] == "SELL"

    if buy_mask.any():
        fig.add_trace(go.Scatter(
            x=dates[buy_mask], y=df_plot.loc[buy_mask, "low"] * 0.998,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10,
                        color=COLORS["green"], line=dict(width=1, color="white")),
            name=f"AL ({buy_mask.sum()})",
        ), row=1, col=1)

    if sell_mask.any():
        fig.add_trace(go.Scatter(
            x=dates[sell_mask], y=df_plot.loc[sell_mask, "high"] * 1.002,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10,
                        color=COLORS["red"], line=dict(width=1, color="white")),
            name=f"SAT ({sell_mask.sum()})",
        ), row=1, col=1)


def _add_rsi_panel(fig, df_plot, dates):
    """RSI alt paneli (30/70 eşikleri)."""
    if "rsi_14" not in df_plot.columns:
        return
    fig.add_trace(go.Scatter(
        x=dates, y=df_plot["rsi_14"],
        line=dict(color=COLORS["gold"], width=1.5),
        name="RSI(14)", showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"],
                  line_width=0.8, opacity=0.6, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"],
                  line_width=0.8, opacity=0.6, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["text"],
                  line_width=0.5, opacity=0.3, row=2, col=1)

    fig.add_hrect(y0=70, y1=100, fillcolor=COLORS["red"],
                  opacity=0.05, line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["green"],
                  opacity=0.05, line_width=0, row=2, col=1)


def _add_macd_panel(fig, df_plot, dates):
    """MACD alt paneli (histogram + line + signal)."""
    if not all(c in df_plot.columns for c in ["macd_line", "macd_signal", "macd_hist"]):
        return
    hist_colors = [COLORS["green"] if v >= 0 else COLORS["red"]
                   for v in df_plot["macd_hist"]]
    fig.add_trace(go.Bar(
        x=dates, y=df_plot["macd_hist"],
        marker_color=hist_colors, opacity=0.5,
        name="Histogram", showlegend=False,
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=df_plot["macd_line"],
        line=dict(color=COLORS["blue"], width=1.5),
        name="MACD", showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=df_plot["macd_signal"],
        line=dict(color=COLORS["red"], width=1, dash="dash"),
        name="Signal", showlegend=False,
    ), row=3, col=1)

    fig.add_hline(y=0, line_color=COLORS["text"],
                  line_width=0.5, opacity=0.4, row=3, col=1)


def _add_volume_and_obv_panels(fig, df_plot, dates):
    """Hacim ve OBV alt panelleri."""
    vol_colors = [COLORS["green"] if c >= o else COLORS["red"]
                  for c, o in zip(df_plot["close"], df_plot["open"])]
    fig.add_trace(go.Bar(
        x=dates, y=df_plot["volume"],
        marker_color=vol_colors, opacity=0.5,
        name="Hacim", showlegend=False,
    ), row=4, col=1)

    direction = np.where(
        df_plot["close"] > df_plot["close"].shift(1), 1,
        np.where(df_plot["close"] < df_plot["close"].shift(1), -1, 0)
    )
    obv = (df_plot["volume"] * direction).cumsum()
    fig.add_trace(go.Scatter(
        x=dates, y=obv,
        line=dict(color=COLORS["purple"], width=1.5),
        fill="tozeroy", fillcolor="rgba(168, 85, 247, 0.1)",
        name="OBV", showlegend=False,
    ), row=5, col=1)


def _apply_dashboard_layout(fig, symbol, interval):
    """Dashboard genel layout ayarlarını uygular."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], size=11),
        title=dict(
            text=f"📊 {symbol} — İnteraktif Dashboard ({interval})",
            font=dict(size=18, color=COLORS["gold"]),
            x=0.5,
        ),
        height=1100,
        showlegend=True,
        legend=dict(
            orientation="h", x=0.5, xanchor="center",
            y=1.02, yanchor="bottom",
            bgcolor="rgba(22, 33, 62, 0.8)",
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Fiyat (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Hacim", row=4, col=1)
    fig.update_yaxes(title_text="OBV", row=5, col=1)
    fig.update_xaxes(title_text="Tarih", row=5, col=1)
    fig.update_xaxes(gridcolor=COLORS["grid"], gridwidth=0.5)
    fig.update_yaxes(gridcolor=COLORS["grid"], gridwidth=0.5)


def create_interactive_dashboard(symbol: str,
                                  interval: str,
                                  last_n: int = 200) -> str | None:
    """
    Belirtilen sembol/interval için Plotly ile interaktif HTML dashboard oluşturur.

    5 alt panel:
      [1] Candlestick + Bollinger Bands + SMA/EMA + Sinyaller
      [2] RSI (30/70 eşikleri)
      [3] MACD (histogram + line + signal)
      [4] Hacim (renkli bar chart)
      [5] On-Balance Volume (OBV)

    Returns:
        Kaydedilen HTML dosyasının yolu veya None
    """
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
    dates = df_plot.index

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.40, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=[
            f"📈 {symbol} — Fiyat & Göstergeler ({interval})",
            "RSI (14)", "MACD", "İşlem Hacmi", "On-Balance Volume (OBV)",
        ],
    )

    _add_price_panel(fig, df_plot, sig_plot, dates)
    _add_rsi_panel(fig, df_plot, dates)
    _add_macd_panel(fig, df_plot, dates)
    _add_volume_and_obv_panels(fig, df_plot, dates)
    _apply_dashboard_layout(fig, symbol, interval)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(REPORTS_DIR, "interactive", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{symbol}_{interval}_dashboard.html"
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath, include_plotlyjs=True)

    logger.info("💾 İnteraktif dashboard kaydedildi: %s", filepath)
    return filepath


# ─────────────────────────────────────────────────────────────
#  2. Feature Importance Grafiği
# ─────────────────────────────────────────────────────────────

def create_feature_importance_chart(symbol: str = "BTCUSDT",
                                     interval: str = "1h") -> str | None:
    """
    Feature importance analizi sonuçlarını interaktif bar chart olarak gösterir.

    3 panel:
      [1] Korelasyon bazlı önem (|Pearson r|)
      [2] Mutual Information skoru
      [3] Birleşik sıralama

    Returns:
        Kaydedilen HTML dosyasının yolu veya None
    """
    # Özellik matrisi oluştur
    df = build_feature_matrix(symbol, interval)
    if df.empty:
        logger.warning("⚠️  Özellik matrisi boş, importance grafiği oluşturulamadı.")
        return None

    # Feature importance analizi
    importance = feature_importance_analysis(df, target_col="close", top_n=15)
    if not importance:
        return None

    # ── Figür oluştur ──
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Korelasyon Bazlı (|r|)",
            "Mutual Information",
            "Birleşik Sıralama",
        ],
        horizontal_spacing=0.08,
    )

    # Panel 1: Korelasyon
    if importance.get("correlation"):
        names = [x[0] for x in reversed(importance["correlation"])]
        values = [x[1] for x in reversed(importance["correlation"])]
        fig.add_trace(go.Bar(
            y=names, x=values, orientation="h",
            marker_color=COLORS["gold"], opacity=0.8,
            name="Korelasyon", showlegend=False,
        ), row=1, col=1)

    # Panel 2: Mutual Information
    if importance.get("mutual_info"):
        names = [x[0] for x in reversed(importance["mutual_info"])]
        values = [x[1] for x in reversed(importance["mutual_info"])]
        fig.add_trace(go.Bar(
            y=names, x=values, orientation="h",
            marker_color=COLORS["purple"], opacity=0.8,
            name="MI Score", showlegend=False,
        ), row=1, col=2)

    # Panel 3: Birleşik sıralama (düşük rank = daha önemli)
    if importance.get("combined_rank"):
        # Rank'ı tersine çevir (düşük rank → yüksek bar)
        max_rank = max(x[1] for x in importance["combined_rank"]) + 1
        names = [x[0] for x in reversed(importance["combined_rank"])]
        values = [max_rank - x[1] for x in reversed(importance["combined_rank"])]
        fig.add_trace(go.Bar(
            y=names, x=values, orientation="h",
            marker_color=COLORS["green"], opacity=0.8,
            name="Birleşik Skor", showlegend=False,
        ), row=1, col=3)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], size=10),
        title=dict(
            text=f"🧬 {symbol} ({interval}) — Feature Importance Analizi",
            font=dict(size=16, color=COLORS["gold"]),
            x=0.5,
        ),
        height=600,
        showlegend=False,
    )

    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])

    # ── Kaydet ──
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(REPORTS_DIR, "interactive", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{symbol}_{interval}_feature_importance.html"
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath, include_plotlyjs=True)

    logger.info("💾 Feature importance grafiği kaydedildi: %s", filepath)
    return filepath


# ─────────────────────────────────────────────────────────────
#  3. Tüm Dashboardları Oluştur
# ─────────────────────────────────────────────────────────────

def generate_all_interactive(include_importance: bool = True) -> list[str]:
    """
    Tüm sembol/interval kombinasyonları için interaktif dashboard oluşturur.

    Returns:
        Oluşturulan HTML dosya yollarının listesi
    """
    logger.info("=" * 60)
    logger.info("🎨 İNTERAKTİF DASHBOARD OLUŞTURMA BAŞLIYOR (5. Hafta)")
    logger.info("=" * 60)

    created_files = []

    for name, symbol in SYMBOLS.items():
        for interval in INTERVALS:
            print(f"\n── {symbol} / {interval} {'─' * 35}")

            result = create_interactive_dashboard(symbol, interval)
            if result:
                created_files.append(result)
                print(f"   ✅ Dashboard: {result}")
            else:
                print(f"   ⚠️  Dashboard oluşturulamadı.")

    # Feature importance (sadece 1d interval için)
    if include_importance:
        print(f"\n── Feature Importance {'─' * 36}")
        for name, symbol in SYMBOLS.items():
            result = create_feature_importance_chart(symbol, "1d")
            if result:
                created_files.append(result)
                print(f"   ✅ {symbol}: {result}")

    logger.info("=" * 60)
    logger.info("✅ İnteraktif dashboard oluşturma tamamlandı: %d dosya",
                len(created_files))
    logger.info("=" * 60)

    return created_files


# ─────────────────────────────────────────────────────────────
#  Tek Başına Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n🎨 İnteraktif Dashboard Testi (5. Hafta)")
    print("=" * 60)
    generate_all_interactive()
