import argparse
import os
import sys

# Proje kokunu sys.path'e ekle; hem 'python src/pipeline.py' hem
# 'python -m src.pipeline' ile calisabilmek icin gerekli.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data import data_pipeline
from src.models import time_series_models, ml_classification_models, ensemble_model
from src.rl import train_agent as rl_train
from src.utils import backtester

# Desteklenen coin listesi
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]


def run_data_pipeline(ticker="BTC-USD"):
    data_pipeline.main(ticker=ticker)


def run_time_series_models(ticker="BTC-USD"):
    time_series_models.main(ticker=ticker)


def run_ml_models(ticker="BTC-USD"):
    ml_classification_models.main(ticker=ticker)


def run_ensemble_model(ticker="BTC-USD"):
    ensemble_model.main(ticker=ticker)


def run_rl_agent(ticker="BTC-USD"):
    rl_train.train_rl_agent(ticker=ticker)


def run_backtest(ticker="BTC-USD"):
    backtester.main(ticker=ticker)


def run_all_for_ticker(ticker, skip_rl=False):
    """Tek bir coin icin tum pipeline adimlarini calistirir."""
    print(f"\n{'#'*60}")
    print(f"  🚀 {ticker} PIPELINE BASLIYOR")
    print(f"{'#'*60}")

    print(f"\n=== {ticker} 1/6: Veri pipeline ===")
    run_data_pipeline(ticker)

    print(f"\n=== {ticker} 2/6: Zaman serisi (LSTM/GRU) modelleri ===")
    run_time_series_models(ticker)

    print(f"\n=== {ticker} 3/6: Tabular ML (RF / XGBoost) modelleri ===")
    run_ml_models(ticker)

    print(f"\n=== {ticker} 4/6: Ensemble meta-model ===")
    run_ensemble_model(ticker)

    if not skip_rl:
        print(f"\n=== {ticker} 5/6: RL ajan egitimi ===")
        run_rl_agent(ticker)
    else:
        print(f"\n>>> {ticker} RL egitimi atlandi (skip-rl=True).")

    print(f"\n=== {ticker} 6/6: Backtest ===")
    run_backtest(ticker)

    print(f"\n{'#'*60}")
    print(f"  🎉 {ticker} PIPELINE TAMAMLANDI!")
    print(f"{'#'*60}")


def run_all(skip_rl=False, coin=None):
    """Tum coinler veya tek coin icin pipeline calistirir."""
    if coin:
        # Tek coin
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
        description="Kripto trading projesi pipeline yoneticisi."
    )
    parser.add_argument(
        "--step",
        choices=["data", "dl", "ml", "ensemble", "rl", "backtest", "all"],
        default="all",
        help="Calistirilacak adim.",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="'all' modunda RL egitimini atla (hizli calistirmalar icin).",
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
    elif args.step == "ensemble":
        run_ensemble_model(ticker)
    elif args.step == "rl":
        run_rl_agent(ticker)
    elif args.step == "backtest":
        run_backtest(ticker)
    elif args.step == "all":
        run_all(skip_rl=args.skip_rl, coin=args.coin)


if __name__ == "__main__":
    main()
