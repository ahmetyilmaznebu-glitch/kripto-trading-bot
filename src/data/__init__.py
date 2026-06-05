"""ML veri paketi — Unified FeatureStore ile veri erisimi."""
from src.data.feature_store import FeatureStore
from src.data.ml_config import FEATURE_COLUMNS, TICKERS, WINDOW_SIZE

# Geriye uyumluluk: loader hala mevcut ama kullanimi onerilen degil
try:
    from src.data.loader import MLDataset, load_dataset
except ImportError:
    MLDataset = None
    load_dataset = None

__all__ = [
    "FeatureStore",
    "FEATURE_COLUMNS",
    "TICKERS",
    "WINDOW_SIZE",
    "load_dataset",
    "MLDataset",
]
