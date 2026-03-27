# Makine Öğrenmesi Tabanlı Kripto Para Trading Botu

**Öğrenci:** Ahmet Yılmaz — 23100011075
**Danışman:** Doç. Dr. Muhammed Karaaltun
**Ders:** Bilgisayar Mühendisliği Uygulama Tasarımı — Near East University

---

## Proje Hakkında

BTC-USD, ETH-USD ve SOL-USD kripto paralar için makine öğrenmesi ve derin öğrenme tabanlı fiyat yönü tahmin sistemi. Model; LSTM/GRU derin öğrenme modelleri, Random Forest ve XGBoost sınıflandırıcılarını bir ensemble yapıda birleştirerek "UP / DOWN" kararı üretir.

---

## Mimari

```
Veri Toplama (Binance / CoinGecko / yfinance)
        ↓
Teknik Gösterge Hesaplama  (17+ gösterge: RSI, MACD, BB, ATR, ADX …)
        ↓
AI Veri Pipeline  (MinMaxScaler → 60-günlük kayan pencere)
        ↓
┌────────────────────────────────────────────────────┐
│            Derin Öğrenme Modelleri                  │
│   LSTM (Attention + Focal Loss)                    │
│   GRU  (Attention + Focal Loss)                    │
└───────────────────┬────────────────────────────────┘
                    │
┌───────────────────▼────────────────────────────────┐
│          ML Sınıflandırıcıları                      │
│   Random Forest  |  XGBoost                        │
└───────────────────┬────────────────────────────────┘
                    │
┌───────────────────▼────────────────────────────────┐
│        Ensemble Meta-Model (Stacking)               │
│       Logistic Regression meta-learner             │
└───────────────────┬────────────────────────────────┘
                    ↓
              UP / DOWN Kararı
```

---

## Özellikler

- **Çoklu Fallback Veri Mimarisi:** SQLite önbellek → Binance REST → Yahoo Finance → yfinance → CoinGecko → CSV
- **Look-Ahead Bias Önlemi:** Scaler yalnızca train setine fit; Train/Val/Test split kronolojik; 60-günlük purging gap
- **2-Fazlı Focal Loss Eğitimi:** Warmup (BCE) → Ana eğitim (Focal Loss, γ=1.0) ile sınıf dengesizliği yönetimi
- **Temporal Attention:** LSTM/GRU çıktılarında hangi zaman adımının kritik olduğunu öğrenir
- **Karar Eşiği Optimizasyonu:** Her model için F1 maksimizasyonu ile optimal threshold belirlenir
- **İnteraktif Dashboard:** Streamlit + Plotly ile mum grafiği, RSI/MACD/BB görselleştirmeleri

---

## Kurulum

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

---

## Kullanım

### Veri Pipeline

```bash
python -m src.data.data_pipeline
```

### Model Eğitimi

```bash
# Derin öğrenme modelleri (LSTM + GRU)
python -m src.models.time_series_models

# ML sınıflandırıcıları (RF + XGBoost)
python -m src.models.ml_classification_models

# Ensemble meta-model
python -m src.models.ensemble_model
```

### Dashboard

```bash
streamlit run ai_dashboard.py
```

---

## Proje Yapısı

```
├── ai_dashboard.py              # Streamlit dashboard
├── data/
│   ├── raw/                     # Ham OHLCV CSV dosyaları (gitignore)
│   └── processed/               # Ölçeklenmiş CSV + window npy (gitignore)
└── src/
    ├── data/
    │   └── data_pipeline.py     # Veri toplama, gösterge, ölçekleme, pencere
    └── models/
        ├── time_series_models.py        # LSTM / GRU + Attention
        ├── ml_classification_models.py  # Random Forest + XGBoost
        ├── ensemble_model.py            # Stacking meta-model
        ├── meta_inference.py            # Çıkarım (inference) yardımcısı
        └── saved_models/                # Eğitilmiş model dosyaları (*.pth gitignore)
```

---

## Teknik Yığın

| Kategori | Araçlar |
|----------|---------|
| Dil | Python 3.11 |
| Derin Öğrenme | PyTorch 2.0 |
| ML | scikit-learn, XGBoost |
| Veri İşleme | NumPy, Pandas, pandas-ta |
| Görselleştirme | Plotly, Streamlit |
| API | Binance REST, CoinGecko |
| Veritabanı | SQLite |

---

## İlerleme Durumu

| Aşama | Konu | Durum |
|-------|------|-------|
| IP 1 | Akademik literatür taraması | ✅ |
| IP 2 | Veri toplama, SQLite, gerçek zamanlı akış | ✅ |
| IP 3 | Teknik göstergeler, interaktif dashboard | ✅ |
| IP 4 | AI veri pipeline, TimeSeriesNet mimarisi | ✅ |
| IP 5 | LSTM/GRU model eğitimi | ✅ |
| IP 6 | ML sınıflandırıcıları (RF + XGBoost) | ✅ |
| IP 7 | Ensemble meta-model (stacking) | ✅ |
| IP 8 | Pekiştirmeli öğrenme (PPO) | 🔄 Devam ediyor |
| IP 9 | Backtesting ve performans analizi | 🔜 |
| IP 10 | Sistem entegrasyonu ve test paketi | 🔜 |
| IP 11 | Final sunumu | 🔜 |
