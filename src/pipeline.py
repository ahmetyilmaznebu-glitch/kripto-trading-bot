"""
Ana pipeline yoneticisi — Kripto Karar Destek Sistemi.

Adimlar:
    data      → Veri cek + feature engineering + dataset build
    dl        → LSTM egitimi (GRU opsiyonel)
    ml        → RF + XGBoost egitimi
    eval      → Validation uzerinde agirlik secimi + test metrikleri
    backtest  → final_prob ile backtest
    all       → Tum adimlari sirayla calistir

Kullanim:
    python -m src.pipeline --step all --coin BTC-USD --skip-rl
    python -m src.pipeline --step eval --coin BTC-USD
"""
import argparse
import os
import sys

# Proje kokunu sys.path'e ekle; hem 'python src/pipeline.py' hem
# 'python -m src.pipeline' ile calisabilmek icin gerekli.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.build_dataset import build as build_dataset
from src.models import time_series_models, ml_classification_models
from src.utils import backtester

# Desteklenen coin listesi
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]


def run_data_pipeline(ticker="BTC-USD"):
    """Veri cekme + feature engineering + dataset build."""
    build_dataset(ticker=ticker)


def run_time_series_models(ticker="BTC-USD"):
    """LSTM (ve opsiyonel GRU) egitimi."""
    time_series_models.main(ticker=ticker)


def run_ml_models(ticker="BTC-USD"):
    """RF + XGBoost egitimi."""
    ml_classification_models.main(ticker=ticker)


def run_eval(ticker="BTC-USD"):
    """Validation uzerinde agirlik secimi + test metrikleri."""
    try:
        import joblib
        import torch
        from src.evaluation.weight_selector import evaluate_weight_configs, get_model_predictions
        from src.evaluation.metrics import compute_classification_metrics, save_metrics
        from src.models.weighted_hybrid import compute_weighted_hybrid
        from src.data.feature_store import FeatureStore
        from src.models.time_series_models import TimeSeriesNet

        store = FeatureStore(ticker)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "src", "models", "saved_models")

        # Modelleri yukle
        rf_model = joblib.load(os.path.join(models_dir, f"{ticker}_rf_classifier.pkl"))
        xgb_model = joblib.load(os.path.join(models_dir, f"{ticker}_xgb_classifier.pkl"))

        # LSTM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_path = os.path.join(models_dir, f"{ticker}_lstm_best.pth")
        input_size = len(store.feature_columns)
        lstm_sd = torch.load(lstm_path, map_location=device, weights_only=True)
        num_layers = max(len([k for k in lstm_sd.keys() if k.startswith('rnn.weight_ih')]), 1)
        # hidden_size'i state_dict'ten otomatik tespit et
        hidden_size = lstm_sd['rnn.weight_ih_l0'].shape[0] // 4  # LSTM: 4 * hidden_size
        lstm_model = TimeSeriesNet(input_size, hidden_size=hidden_size, num_layers=num_layers,
                                   output_size=1, model_type="LSTM", use_attention=True)
        lstm_model.load_state_dict(lstm_sd)
        lstm_model.to(device)
        lstm_model.eval()

        result = evaluate_weight_configs(store, rf_model, xgb_model, lstm_model, ticker=ticker)
        bw = result["best_weights"]
        print(f"\n  {ticker} En iyi agirliklar: RF={bw[0]:.2f}, "
              f"XGB={bw[1]:.2f}, LSTM={bw[2]:.2f}")

        # Test seti tahminlerini al
        rf_test_probs, xgb_test_probs, lstm_test_probs, y_test = get_model_predictions(
            store, rf_model, xgb_model, lstm_model, "test"
        )

        # Weighted hybrid
        final_test_probs = compute_weighted_hybrid(
            rf_test_probs, xgb_test_probs, lstm_test_probs,
            w_rf=bw[0], w_xgb=bw[1], w_lstm=bw[2]
        )

        # Test tahminleri (varsayilan 0.50 threshold)
        y_test_pred = (final_test_probs > 0.50).astype(int)

        # Metrikleri hesapla
        metrics = compute_classification_metrics(y_test, y_test_pred, final_test_probs)

        # Agirlik ve dejenere bilgisini ekle
        metrics["best_weights"] = {"rf": bw[0], "xgb": bw[1], "lstm": bw[2]}
        metrics["lstm_degenerate"] = result["lstm_degenerate"]

        # Metrikleri kaydet
        out_path = os.path.join(base_dir, "outputs", "metrics", f"{ticker}_evaluation_metrics.json")
        save_metrics(metrics, out_path)
    except Exception as e:
        print(f"\n  [UYARI] {ticker} eval asamasi basarisiz: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Devam ediliyor (varsayilan agirliklar kullanilacak)...")


def run_backtest(ticker="BTC-USD"):
    """Backtest motoru (final_prob ile)."""
    backtester.main(ticker=ticker)


def run_all_for_ticker(ticker, skip_rl=False):
    """Tek bir coin icin tum pipeline adimlarini calistirir."""
    print(f"\n{'#'*60}")
    print(f"  🚀 {ticker} PIPELINE BASLIYOR")
    print(f"{'#'*60}")

    print(f"\n=== {ticker} 1/5: Veri pipeline ===")
    run_data_pipeline(ticker)

    print(f"\n=== {ticker} 2/5: Zaman serisi (LSTM) modeli ===")
    run_time_series_models(ticker)

    print(f"\n=== {ticker} 3/5: Tabular ML (RF / XGBoost) modelleri ===")
    run_ml_models(ticker)

    print(f"\n=== {ticker} 4/5: Evaluation (agirlik secimi + metrikler) ===")
    run_eval(ticker)

    print(f"\n=== {ticker} 5/5: Backtest ===")
    run_backtest(ticker)

    print(f"\n{'#'*60}")
    print(f"  🎉 {ticker} PIPELINE TAMAMLANDI!")
    print(f"{'#'*60}")


def run_all(skip_rl=False, coin=None):
    """Tum coinler veya tek coin icin pipeline calistirir."""
    if coin:
        tickers = [coin]
    else:
        tickers = TICKERS

    for ticker in tickers:
        run_all_for_ticker(ticker, skip_rl=skip_rl)

    print(f"\n{'='*60}")
    print(f"  🎉 TUM COIN PIPELINE'LARI TAMAMLANDI ({len(tickers)} coin)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Kripto karar destek sistemi pipeline yoneticisi."
    )
    parser.add_argument(
        "--step",
        choices=["data", "dl", "ml", "eval", "backtest", "all"],
        default="all",
        help="Calistirilacak adim.",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Geriye uyumluluk icin korunuyor (RL artik varsayilan olarak atlanir).",
    )
    parser.add_argument(
        "--coin",
        choices=TICKERS,
        default=None,
        help="Sadece belirli bir coin icin calistir (orn: BTC-USD, ETH-USD, SOL-USD).",
    )

    args = parser.parse_args()

    # Proje kok dizininde calisildigindan emin ol
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    ticker = args.coin or "BTC-USD"

    if args.step == "data":
        run_data_pipeline(ticker)
    elif args.step == "dl":
        run_time_series_models(ticker)
    elif args.step == "ml":
        run_ml_models(ticker)
    elif args.step == "eval":
        run_eval(ticker)
    elif args.step == "backtest":
        run_backtest(ticker)
    elif args.step == "all":
        run_all(skip_rl=args.skip_rl, coin=args.coin)


if __name__ == "__main__":
    main()
