# Experiment Notes

## RF/XGBoost Retraining Decision - 2026-06-05

Context: the project is now positioned as a Crypto Decision Support System. The production decision model remains the Weighted Hybrid:

```text
final_prob = 0.40 * RF_prob + 0.40 * XGB_prob + 0.20 * LSTM_prob
```

The current production RF/XGBoost models are feature-compatible with the final `data/ml` datasets (`n_features_in_ = 20` for RF and XGBoost), but their saved timestamps are older than the final `data/ml` manifests. A retraining experiment was therefore run without deleting or overwriting production models.

### Experiment Location

New experiment artifacts were saved under:

```text
src/models/saved_models/ml_experiments/20260605_162854/
```

The summary report is:

```text
src/models/saved_models/ml_experiments/20260605_162854/ml_experiment_summary.json
```

These files are archived experiment outputs only. They are not production model paths.

### Experiment Setup

- Training used the final `FeatureStore` `train` split.
- Validation was used for model selection.
- Test was used only for final reporting.
- Existing production models were kept unchanged.
- LSTM models, LSTM loader behavior, backtest `FeatureStore + compute_final_probs` flow, and dashboard production paths were not changed.

Tested model families:

- Experiment A: retrain RF/XGBoost with the existing hyperparameter pattern.
- Experiment B: controlled Random Forest variants (`n_estimators=300`, `max_depth=5/8`, `min_samples_leaf=5`, `class_weight="balanced"`).
- Experiment C: controlled XGBoost (`n_estimators=300`, `max_depth=3`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=2`, `reg_alpha=0.1`).
- Experiment D: RF-only, XGB-only, RF+XGB average, and Weighted Hybrid comparisons.

### Results Summary

| Coin | Old RF+XGB Val F1 | New RF+XGB Val F1 | New Weighted Hybrid Val F1 | Decision |
|---|---:|---:|---:|---|
| BTC-USD | 0.464 | 0.535 | 0.563 | Experiment improved validation but failed generalization checks |
| ETH-USD | 0.559 | 0.447 | 0.462 | Rejected |
| SOL-USD | 0.622 | 0.441 | 0.458 | Rejected |

BTC validation improved, but the new BTC test metrics and backtest weakened:

- Old RF+XGB test F1: `0.582`, AUC: `0.621`
- New RF+XGB test F1: `0.509`, AUC: `0.524`
- New Weighted Hybrid test F1: `0.535`, AUC: `0.525`
- New Weighted Hybrid backtest return: `-12.52%`

ETH and SOL new models were worse on validation than the existing production models. Backtests also showed weaker behavior across all coins.

### Final Decision

The new RF/XGBoost experiment models are **not promoted to production**.

Production continues to use:

```text
src/models/saved_models/{ticker}_rf_classifier.pkl
src/models/saved_models/{ticker}_xgb_classifier.pkl
```

The dashboard and pipeline continue to read the current production model paths. The Weighted Hybrid weights remain:

```text
RF = 0.40
XGBoost = 0.40
LSTM = 0.20
```

The `ml_experiments/` directory is retained as an archive for the retraining attempt and model selection decision.

### Notes And Risks

- The experiment models are useful evidence that newer training alone does not guarantee better generalization.
- BTC improved on validation but failed the broader acceptance checks on test F1/AUC and backtest behavior.
- ETH and SOL did not meet the validation criterion.
- Future retraining should use time-series cross-validation and keep validation-only model selection. Test data must remain final-report-only.
