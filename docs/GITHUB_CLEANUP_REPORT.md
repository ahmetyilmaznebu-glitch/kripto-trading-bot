# GitHub Cleanup Report

Date: 2026-06-05

## Summary

This cleanup pass focused on GitHub readiness without changing production code paths.
Only cache and temporary files were deleted. Active data and active production model
files are now intended to be included in GitHub. Archive, experiment, cache, log,
virtual environment, duplicate raw, and old output files remain excluded.

## Inventory Categories

| Category | Paths | GitHub action | Risk note |
|---|---|---|---|
| KEEP_PRODUCTION | `ai_dashboard.py`, `config.py`, `src/data/`, `src/models/meta_inference.py`, `src/models/weighted_hybrid.py`, `src/models/ml_classification_models.py`, `src/models/time_series_models.py`, `src/evaluation/`, `src/utils/backtester.py`, `src/pipeline.py` | Keep / stage selectively | Active dashboard, data pipeline, model inference, evaluation, and backtest flow. |
| KEEP_ACTIVE_DATA | `data/raw/*_ohlcv.csv`, `data/ml/BTC-USD/`, `data/ml/ETH-USD/`, `data/ml/SOL-USD/` | Keep / include | Active dashboard and FeatureStore data. |
| KEEP_ACTIVE_MODELS | `src/models/saved_models/*_rf_classifier.pkl`, `src/models/saved_models/*_xgb_classifier.pkl`, `src/models/saved_models/*_lstm_best.pth`, `src/models/saved_models/*_dl_thresholds.json` | Keep / include | Active Weighted Hybrid production model files. |
| KEEP_TESTS | `tests/` | Keep | Active unit tests. |
| KEEP_DOCS | `README.md`, `docs/` | Keep | Documentation should be included; previous `docs/` ignore rule was removed. |
| KEEP_SMALL_OUTPUT | `outputs/metrics/*.json`, `outputs/charts/*.png` | Keep | Dashboard reads these for metrics/charts fallback. |
| GITIGNORE_NONACTIVE_DATA | `data/raw/*_raw.csv`, `data/raw/archive_*/`, `data/ml/archive_*/`, `data/processed/`, `data/exported/`, `data/results/`, `*.db`, `*.sqlite*` | Ignore, do not delete | Duplicate, archive, legacy, or generated data outside current FeatureStore flow. |
| GITIGNORE_NONACTIVE_MODELS | `src/models/saved_models/ml_experiments/`, `src/models/saved_models/lstm_experiments/`, `src/models/saved_models/*_ensemble_meta_model.pkl`, `src/models/saved_models/*_gru_best.pth`, `src/models/saved_models/*_lstm_best_old64.pth`, `src/rl/saved_agents/`, `outputs/models/` | Ignore, do not delete | Experiment/legacy model artifacts, not active Weighted Hybrid production files. |
| KEEP_EXPERIMENT_LOCAL | `src/models/saved_models/ml_experiments/`, `src/models/saved_models/lstm_experiments/`, `src/rl/`, `requirements-rl.txt` | Keep local / document | Valuable experiment artifacts, not production paths. |
| DELETE_CACHE | `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ipynb_checkpoints/`, `*.pyc`, `*.pyo`, `*.log` | Deleted if outside `.venv` and `.git` | Safe generated files. |
| UNKNOWN_NEEDS_REVIEW | `main.py`, `database.py`, `binance_collector.py`, `coingecko_collector.py`, `signal_generator.py`, `indicators.py`, `data_processor.py`, `interactive_dashboard.py`, `visualizer.py`, `save_and_analyze.py`, `test_suite.py`, `analysis_v2.py` | Not moved | Legacy bot/test/analysis files have internal imports; moving could break old workflows. |
| GITIGNORE_ARCHIVES | `backup_models_*/`, `zip/`, `*.rar`, `*.7z`, `proje_ciktilari.zip`, `analysis_v2_output.txt` | Ignore, do not delete | Large or duplicate/archive outputs. |

## Import Notes

Legacy bot files are internally connected:

| File/group | Observed use |
|---|---|
| `main.py` | Imports `database`, collectors, processor, indicators, signal generator, visualizer, interactive dashboard. |
| `binance_collector.py`, `coingecko_collector.py` | Import `database`. |
| `interactive_dashboard.py`, `visualizer.py`, `signal_generator.py` | Import legacy processor/indicator modules. |
| `test_suite.py` | Imports legacy bot modules. |

These files are not used by the current academic dashboard routing, but they were not
moved in this pass because their internal dependencies are still coupled.

## Deleted Temporary Files

- Removed project-local `__pycache__/` directories.
- Removed project-local `*.pyc` files.
- Removed `crypto_bot.log`.
- Did not touch `.venv/` or `.git/`.

## Updated Git Ignore Policy

The `.gitignore` now excludes:

- virtual environments and cache files
- logs
- duplicate raw, archive, processed, exported, and generated result data
- database files
- experiment/legacy model binaries and saved model experiment directories
- RL saved agents
- backups, archives, duplicate zip folders
- temporary analysis outputs

It explicitly keeps:

- `data/raw/*_ohlcv.csv`
- `data/ml/BTC-USD/`, `data/ml/ETH-USD/`, `data/ml/SOL-USD/`
- `src/models/saved_models/*_rf_classifier.pkl`
- `src/models/saved_models/*_xgb_classifier.pkl`
- `src/models/saved_models/*_lstm_best.pth`
- `src/models/saved_models/*_dl_thresholds.json`
- `outputs/metrics/*.json`
- `outputs/charts/*.png`
- `docs/`

## Recommended GitHub Staging Set

Stage selectively:

```powershell
git add .gitignore README.md requirements.txt requirements-rl.txt ai_dashboard.py src tests docs outputs/metrics outputs/charts
git add data/raw/*_ohlcv.csv data/ml/BTC-USD data/ml/ETH-USD data/ml/SOL-USD
git add src/models/saved_models/*_rf_classifier.pkl src/models/saved_models/*_xgb_classifier.pkl src/models/saved_models/*_lstm_best.pth src/models/saved_models/*_dl_thresholds.json
git status
```

Review before commit:

```powershell
git status --short
git diff --cached --stat
```
