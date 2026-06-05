# Kripto Para Yön Tahmini — Karar Destek Sistemi

> **Proje:** Ahmet Yılmaz — 23100011075 — Necmettin Erbakan Üniversitesi 
> **Danışman:** Doç. Dr. Muhammed Karaaltun  
> **Ders:** Bilgisayar Mühendisliği Uygulama Tasarımı

## Proje Hakkında

Bu proje, makine öğrenmesi yöntemleriyle kripto para (BTC, ETH, SOL) fiyat yönü
tahmini yapan bir **karar destek / araştırma prototipi**dir.

> ⚠️ Bu sistem bir yatırım tavsiyesi aracı değildir. Akademik araştırma amacıyla
> geliştirilmiştir ve canlı trading için kullanılmamalıdır.

## Ana Model — Weighted Hybrid Ensemble

**RF + XGBoost Ağırlıklı, LSTM Destekli Hibrit Model**

```
final_prob = 0.40 × RF_prob + 0.40 × XGBoost_prob + 0.20 × LSTM_prob
```

| Model | Rol | Ağırlık |
|-------|-----|---------|
| **Random Forest** | Ana sınıflandırıcı, feature importance | %40 |
| **XGBoost** | Gradient boosting ile güçlendirilmiş | %40 |
| **LSTM** | Zaman serisi destekleyici | %20 |

### Fallback Mekanizması

| Durum | Formül |
|-------|--------|
| LSTM makul (val F1 > 0.40) | 0.40 RF + 0.40 XGB + 0.20 LSTM |
| LSTM zayıf (F1 ≤ 0.40) | 0.45 RF + 0.45 XGB + 0.10 LSTM |
| LSTM dejenere (tek sınıf) | 0.50 RF + 0.50 XGB + 0.00 LSTM |

## Teknolojiler

- Python 3.10+, PyTorch, scikit-learn, XGBoost
- Pandas, NumPy, ta (teknik analiz kütüphanesi)
- Streamlit (interaktif dashboard), Matplotlib (grafikler)

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
# Tam pipeline (veri + eğitim + evaluation + backtest)
python -m src.pipeline --step all --coin BTC-USD

# Sadece belirli adımlar
python -m src.pipeline --step data --coin BTC-USD     # Veri hazırlama
python -m src.pipeline --step dl --coin BTC-USD       # LSTM eğitimi
python -m src.pipeline --step ml --coin BTC-USD       # RF + XGBoost
python -m src.pipeline --step eval --coin BTC-USD     # Evaluation
python -m src.pipeline --step backtest --coin BTC-USD # Backtest

# Tüm coinler için
python -m src.pipeline --step all

# Dashboard
streamlit run ai_dashboard.py
```

## Pipeline Adımları

```
1. Veri çek           → fetch_ohlcv.py
2. Feature üret       → feature_engineering.py (20 stationary feature)
3. Dataset build      → build_dataset.py (dead-zone filtre + scaler + split)
4. LSTM eğit          → time_series_models.py
5. RF + XGBoost eğit  → ml_classification_models.py
6. Evaluation         → weight_selector.py (val üzerinde ağırlık seçimi)
7. Backtest           → backtester.py (final_prob ile)
```

## Klasör Yapısı

```
project_root/
├── src/                    — Ana AI pipeline kaynak kodu
│   ├── data/               — Veri işleme (FeatureStore, build_dataset)
│   ├── models/             — Modeller (RF, XGB, LSTM, weighted_hybrid)
│   ├── evaluation/         — Metrikler ve ağırlık seçimi
│   └── utils/              — Backtest motoru
├── data/                   — Veri dosyaları
│   ├── raw/                — Ham OHLCV CSV'ler
│   └── ml/                 — Build edilmiş dataset (manifest v3)
├── outputs/                — Çıktılar (metrikler, grafikler, tahminler)
├── docs/                   — Dokümantasyon ve raporlar
├── legacy/                 — Eski bot ve deneysel modüller
└── ai_dashboard.py         — Streamlit dashboard
```

## Veri

- **Coinler:** BTC-USD, ETH-USD, SOL-USD
- **Kaynak:** Yahoo Finance (yfinance), Binance API fallback
- **Periyot:** Günlük OHLCV
- **Feature sayısı:** 20 durağan (stationary) teknik gösterge
- **Split:** %70 train / %15 validation / %15 test (purge gap = 60 gün)
- **Dead-zone filtresi:** ±%0.15'ten küçük hareketler çıkarılır

## Sınırlamalar

- ~300 günlük veri ile eğitilmiştir (LSTM için yetersiz)
- Dead-zone filtresi sonrası ~250-330 örnek kalmaktadır
- Geçmiş performans gelecek performansı garanti etmez
- Komisyon modeli basitleştirilmiştir (%0.1 sabit)

## Etik Uyarı

Bu proje kripto para piyasalarında kesin kazanç sağlamayı amaçlamamaktadır.
Yön tahmini yapan bir araştırma prototipidir. Gerçek yatırım kararları için
profesyonel finansal danışmanlık alınmalıdır.

## Deneysel Modüller (Ana Pipeline Dışı)

Aşağıdaki modüller araştırma amaçlı geliştirilmiş olup **ana karar mekanizmasının parçası değildir:**

- **PPO/RL Agent:** Pekiştirmeli öğrenme ile dinamik pozisyon yönetimi denenmiştir (`src/rl/`)
- **GRU:** LSTM alternatifi olarak değerlendirilmiştir
- **Full Stacking Ensemble:** 4-model GradientBoosting meta-learner denenmiştir (`src/models/ensemble_model.py`)

> ⚠️ RL bağımlılıkları (stable-baselines3, gymnasium) ana `requirements.txt`'te yer almaz.
> RL deneyleri için: `pip install -r requirements-rl.txt`
