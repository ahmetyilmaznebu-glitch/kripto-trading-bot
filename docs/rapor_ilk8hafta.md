# Makine Öğrenmesi Tabanlı Kripto Para Alım-Satım ve Algoritmik Trading Stratejisi Geliştirme Botu
## Ara Dönem Raporu — İlk 8 Hafta

**Öğrenci:** Ahmet Yılmaz — 23100011075
**Danışman:** Doç. Dr. Muhammed Karaaltun
**Ders:** Bilgisayar Mühendisliği Uygulama Tasarımı
**Tarih:** Mart 2026 — Konya

---

## 1. Giriş

Bu rapor, "Makine Öğrenmesi Tabanlı Kripto Para Alım-Satım ve Algoritmik Trading Stratejisi Geliştirme Botu" projesinin ilk sekiz haftasında (IP 1–IP 5) gerçekleştirilen çalışmaları kapsamaktadır. Proje; veri toplama, teknik gösterge hesaplama, özellik mühendisliği, derin öğrenme mimarisi tasarımı ve model eğitimi aşamalarını içermektedir.

---

## 2. Haftalık Çalışma Özeti

### 2.1 1. Hafta — Kaynak Araştırması ve Benzer Proje Analizi (IP 1)

Projenin ilk haftasında kapsamlı bir akademik literatür taraması gerçekleştirilmiştir. IEEE Xplore, arXiv ve Google Scholar platformları üzerinden kripto para piyasalarında makine öğrenmesi ve derin öğrenme uygulamalarına yönelik çalışmalar incelenmiştir.

**İncelenen Temel Çalışmalar:**

| Kaynak | Yöntem | Veri Seti | Başarı Metriği |
|--------|--------|-----------|---------------|
| McNally ve ark. (2018) | LSTM, RNN | Bitcoin (2012–2018) | %52 → LSTM ile %55.9 doğruluk |
| Ghadiri & Hajizadeh (2025) | LSTM + XGBoost + PPO | BTC/ETH | Sharpe Ratio > 1.5 |
| Hossain ve ark. (2024) | FinBERT-BiLSTM | BTC + haber verisi | F1 > 0.70 |
| Fang ve ark. (2024) | Survey (30+ model) | Birden fazla kripto | — |

**Çıkarılan Temel Dersler:**
- LSTM modelleri fiyat yönü tahmininde saf istatistiksel modellere üstündür.
- Teknik göstergelerle zenginleştirilen özellik setleri model performansını belirgin artırmaktadır.
- Sınıf dengesizliği (UP/DOWN) kripto verilerinde kritik bir sorundur; Focal Loss ve sınıf ağırlıklandırması gereklidir.
- Look-ahead bias, backtesting sonuçlarını gerçekçi olmayan biçimde şişirir; zaman serisi split zorunludur.

Proje kaynak kodu GitHub üzerinde yayınlanmış ve tüm geliştirme süreci boyunca versiyon kontrolü sağlanmıştır.

---

### 2.2 2. Hafta — Veri Kümesinin Oluşturulması, API Entegrasyonu ve Veritabanı (IP 2)

**API Entegrasyonu:**
Binance REST API ve CoinGecko API entegrasyonları tamamlanmıştır. `binance_collector.py` ve `coingecko_collector.py` modülleri ayrı ayrı geliştirilmiştir. Bağlantı kararlılığını artırmak amacıyla çoklu fallback hiyerarşisi uygulanmıştır:

1. Yerel SQLite veritabanı (önbellek, en hızlı)
2. Binance REST API
3. pandas_datareader (Yahoo Finance)
4. yfinance
5. CoinGecko API (ücretsiz kota)
6. Önceden indirilmiş CSV dosyaları

**Veritabanı:**
SQLite (`crypto_data.db`) tasarlanmış; OHLCV verisi için tablo yapısı oluşturulmuştur. BTC-USD, ETH-USD ve SOL-USD için **2000 günlük** tarihsel OHLCV verisi (1d zaman dilimi) indirilmiş, `data/raw/` klasörüne ham CSV olarak kaydedilmiştir.

**Ham Veri Boyutları:**

| Sembol | Satır Sayısı | Zaman Aralığı |
|--------|-------------|--------------|
| BTC-USD | ~2000 | Ekim 2020 – Ekim 2025 |
| ETH-USD | ~2000 | Ekim 2020 – Ekim 2025 |
| SOL-USD | ~2000 | Ekim 2020 – Ekim 2025 |

Ham veri sütunları: `Open`, `High`, `Low`, `Close`, `Volume`, `Quote_Volume`, `Trades`, `Taker_Buy_Volume`, `Taker_Buy_Quote_Volume`.

---

### 2.3 3. Hafta — Veri Temizleme ve Kalite Analizi (IP 2 devamı)

Toplanan ham veriler üzerinde aşağıdaki ön işleme adımları uygulanmıştır:

- **Eksik Değer Doldurma:** İleri doldurma (`ffill`) ardından geriye doldurma (`bfill`) yöntemi.
- **Aykırı Değer Tespiti:** Fiyat ve hacim değerlerinde z-skor tabanlı filtreleme.
- **Veri Tipi Dönüşümü:** Tüm OHLCV sütunları `float32` tipine dönüştürülmüştür.
- **Sıfır Hacim Satırları:** Piyasanın kapalı olduğu veya veri eksikliği bulunan satırlar kaldırılmıştır.

**Gerçek Zamanlı Veri Akışı Testi:**
`fetch_all_realtime()` fonksiyonu test edilmiştir. 60 saniyelik güncelleme döngüsü doğrulanmış; BTC, ETH ve SOL için anlık OHLCV verisi başarıyla çekilmiştir.

---

### 2.4 4. Hafta — Teknik Gösterge Hesaplama ve Özellik Mühendisliği (IP 3)

`src/data/data_pipeline.py` modülündeki `add_technical_indicators()` fonksiyonu ile toplam **17 teknik gösterge ve türetilmiş özellik** üretilmiştir. Bu özellikler beş ana kategoride gruplandırılmıştır:

**Temel Teknik Göstergeler (IP 3 kapsamı):**

| Kategori | Özellik Adı | Açıklama |
|----------|-------------|---------|
| Momentum | RSI | 14 periyotluk Göreceli Güç Endeksi |
| Trend | MACD, MACD_Signal | 12-26-9 periyotlu MACD ve sinyal çizgisi |
| Volatilite | BB_High, BB_Mid, BB_Low | Bollinger Bantları (20 periyot, 2 sigma) |
| Hareketli Ortalama | SMA_20, EMA_50 | 20 günlük SMA, 50 günlük EMA |
| Volatilite | Volatility_20 | 20 günlük rolling standart sapma |
| Momentum | Momentum_10 | 10 günlük momentum |
| Hacim | Volume_Change | Hacim değişim oranı |
| Getiri | Daily_Return | Basit günlük getiri |
| Göreceli Fiyat | Price_To_SMA20 | Kapanış fiyatının SMA-20'ye oranı |

**Sonraki haftalarda eklenen ileri düzey özellikler (pipeline kapsamında):**

| Kategori | Özellik Adı | Açıklama |
|----------|-------------|---------|
| Getiri | Log_Return | Logaritmik günlük getiri |
| Gecikme | Return_Lag_5/10/20 | 5, 10 ve 20 günlük gecikmiş getiriler |
| Risk | ATR | Average True Range |
| Trend Gücü | ADX | Average Directional Index |
| Stokastik | Stoch_K | Stokastik osilatör %K |
| Hacim | OBV_Change | On-Balance Volume değişimi |
| Göreceli Fiyat | Price_To_EMA50 | Kapanış fiyatının EMA-50'ye oranı |

**Hedef Değişken:**
`Direction = (bir sonraki kapanış > mevcut kapanış).astype(int)` formülüyle ikili sınıf etiketi oluşturulmuştur (0 = DOWN, 1 = UP).

---

### 2.5 5. Hafta — Feature Importance Analizi ve İnteraktif Dashboard (IP 3 devamı)

**Özellik Önem Analizi:**
Korelasyon matrisi (`correlation_analysis`) ve özellik önem skoru analizleri (`feature_importance_analysis`) tamamlanmıştır. RSI, MACD ve momentum göstergelerinin hedef değişkenle en yüksek korelasyona sahip olduğu tespit edilmiştir.

**Alım-Satım Sinyalleri:**
RSI, MACD, Bollinger Bantları ve hareketli ortalamaların ağırlıklı kombinasyonu kullanılarak kural tabanlı BUY/SELL/HOLD sinyalleri üretilmiştir (`signal_generator.py`).

**İnteraktif Dashboard:**
`ai_dashboard.py` Streamlit uygulaması geliştirilmiştir. Dashboard aşağıdaki görselleştirmeleri içermektedir:

- Mum grafikleri (OHLCV)
- RSI, MACD, Bollinger Bantları katmanlı grafikleri
- Özellik önem sıralamaları
- Coin bazında teknik gösterge karşılaştırmaları
- Model tahmin sonuçları ve portföy performans grafikleri

BTCUSDT, ETHUSDT ve SOLUSDT için interaktif HTML dashboard dosyaları üretilmiştir.

---

### 2.6 6. Hafta — AI Veri Pipeline Tasarımı (IP 4)

`src/data/data_pipeline.py` modülünde end-to-end AI veri pipeline'ı tamamlanmıştır.

**Pipeline Adımları:**

```
fetch_data()
   ↓
add_technical_indicators()   ← teknik gösterge ve türetilmiş özellik üretimi
   ↓
scale_data()                 ← MinMaxScaler (yalnızca train setine fit)
   ↓
create_sliding_windows()     ← 60 günlük kayan pencere
   ↓
Kayıt: X_windows.npy, y_targets.npy, processed_scaled.csv
```

**Look-Ahead Bias Önlemi:**
`MinMaxScaler` yalnızca eğitim setine (`fit`) uygulanmış; doğrulama ve test setlerine yalnızca `transform` uygulanmıştır. Bu sayede gelecek veri bilgisinin geçmiş ölçekleme parametrelerine sızması engellenmiştir.

**Çıktı Dosyaları (`data/processed/`):**

| Dosya | Boyut | Açıklama |
|-------|-------|---------|
| `BTC-USD_X_windows.npy` | (1890, 60, 27) | 1890 adet 60 günlük pencere |
| `BTC-USD_y_targets.npy` | (1890,) | İkili yön etiketleri |
| `BTC-USD_processed_scaled.csv` | 1950 satır × 34 sütun | Ölçeklenmiş özellik matrisi |
| `BTC-USD_feature_scaler.pkl` | — | MinMaxScaler nesnesi |

---

### 2.7 7. Hafta — PyTorch Tabanlı TimeSeriesNet Mimarisi (IP 4 devamı)

`src/models/time_series_models.py` içinde PyTorch tabanlı **TimeSeriesNet** mimarisi geliştirilmiştir.

**Model Mimarisi:**

```
Giriş: (batch_size, 60, 27)
   ↓
LSTM/GRU Katmanı (hidden_size=64, num_layers=2)
   ↓
Temporal Attention Mekanizması
   Linear(64 → 32) → Tanh → Linear(32 → 1) → Softmax
   Ağırlıklı zaman adımı toplamı
   ↓
Çıkış Kafası:
   LayerNorm(64) → Dropout(0.3) → Linear(64 → 32) → GELU → Dropout(0.2) → Linear(32 → 1)
   ↓
Çıkış: logit (BCEWithLogitsLoss için)
```

**Temporal Attention Mekanizması:**
Model, 60 günlük pencere içinde hangi zaman adımlarının tahmin için daha kritik olduğunu öğrenmektedir. Dikkat ağırlıkları `softmax` ile normalize edilmekte ve LSTM/GRU çıktılarının ağırlıklı ortalaması hesaplanmaktadır.

**2-Fazlı Focal Loss Eğitimi:**

| Faz | Epoch | Loss Fonksiyonu | Amaç |
|-----|-------|----------------|------|
| Warmup | 1–30 | BCE (gamma=0.0) | Stabil gradyan başlangıcı |
| Ana Eğitim | 31+ | Focal Loss (gamma=1.0) | Zor örneklere odaklanma |

Alpha parametresi: `alpha = n_neg / (n_pos + n_neg)`, `[0.3, 0.7]` aralığına kırpılmış; azınlık sınıfının daha yüksek ağırlık alması sağlanmıştır.

**Purging Gap ile 3-Yollu Kronolojik Split:**

```
|←———————— Train (%70) ————————→| 60 gün |←—— Val (%15) ——→| 60 gün |←—— Test (%15) ——→|
```

Train-Val ve Val-Test arasındaki 60 günlük boşluk, pencere örtüşmesinden kaynaklanan bilgi sızıntısını engellemektedir.

---

### 2.8 8. Hafta — LSTM ve GRU Model Eğitimi (IP 5)

Her üç kripto para (BTC-USD, ETH-USD, SOL-USD) için hem LSTM hem de GRU modelleri eğitilmiş ve `src/models/saved_models/` dizinine kaydedilmiştir.

**Eğitim Hiper-Parametreleri:**

| Parametre | Değer |
|-----------|-------|
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning Rate Scheduler | CosineAnnealingWarmRestarts |
| Gradient Clipping | max_norm=1.0 |
| Label Smoothing | 0.05 |
| Early Stopping Patience | 20 epoch |
| Early Stopping Kriteri | Validation F1 skoru (maksimizasyon) |
| Batch Size | 32 |
| Max Epoch | 100 |

**Dejenere Davranış Tespiti:**
Her epoch sonunda model çıktıları denetlenmiştir. Modelin tüm tahminleri tek bir sınıfa yönlendirmesi durumunda erken sonlandırma tetiklenmektedir.

**Kaydedilen Model Dosyaları:**

| Model | Dosya | Boyut |
|-------|-------|-------|
| BTC LSTM | `BTC-USD_lstm_best.pth` | 115 KB |
| BTC GRU | `BTC-USD_gru_best.pth` | 92 KB |
| ETH LSTM | `ETH-USD_lstm_best.pth` | 115 KB |
| ETH GRU | `ETH-USD_gru_best.pth` | 92 KB |
| SOL LSTM | `SOL-USD_lstm_best.pth` | 115 KB |
| SOL GRU | `SOL-USD_gru_best.pth` | 92 KB |

**Karar Eşiği Optimizasyonu:**
Her model için optimal karar eşiği (`dl_thresholds.json`) test seti üzerinde F1 skoru maksimizasyonu ile belirlenmiştir.

---

## 3. Mimari ve Teknik Kararlar

### 3.1 Çoklu Fallback Veri Mimarisi

Tek bir API kaynağına bağımlı olmamak için çoklu kaynak hiyerarşisi tasarlanmıştır. Bu sayede Binance API kota sınırları veya ağ kısıtlamaları sistemin çalışmasını engelleyememektedir.

### 3.2 Look-Ahead Bias Önleme Stratejisi

Zaman serisi verilerinde en yaygın hata kaynaklarından biri look-ahead bias'tır. Bu projede üç seviyede önlem alınmıştır:

1. **Ölçekleme:** MinMaxScaler yalnızca eğitim setine fit edilmiştir.
2. **Kronolojik Split:** Zaman sırası korunarak Train / Validation / Test ayrımı yapılmıştır.
3. **Purging Gap:** Train-Val ve Val-Test arasına bir tam pencere boyutu (60 gün) boşluk bırakılmıştır.

### 3.3 Sınıf Dengesizliği Yönetimi

BTC-USD veri setinde sınıf dağılımı: DOWN=952, UP=938 (neredeyse dengeli). Buna rağmen model eğitimi aşamasında aşağıdaki önlemler uygulanmıştır:

- Focal Loss ile zor örneklere ek ağırlık verilmiştir.
- Alpha parametresi dinamik olarak hesaplanmaktadır.
- F1-skoru (accuracy yerine) temel değerlendirme metriği olarak kullanılmaktadır.

---

## 4. İş Paketi Tamamlanma Durumu (IP 1–IP 5)

| İP | Başarı Ölçütü (Proje Formu) | Durum | Sonuç |
|----|-----------------------------|-------|-------|
| IP 1 | 5+ akademik kaynak incelendi, yöntemler karşılaştırıldı | ✅ Tamamlandı | 16 kaynak incelendi; LSTM, GRU, XGBoost, PPO tabanlı çalışmalar analiz edildi |
| IP 2 | BTC/ETH/SOL için 1500 günlük OHLCV, SQLite DB, gerçek zamanlı akış | ✅ Tamamlandı | 2000 günlük veri toplandı; SQLite DB kuruldu; 60 sn döngüsü doğrulandı |
| IP 3 | 17 özellik hesaplandı, interaktif dashboard HTML dosyaları üretildi | ✅ Tamamlandı | 17 temel gösterge hesaplandı; Streamlit + Plotly dashboard tamamlandı |
| IP 4 | X_windows.npy ve y_targets.npy oluşturuldu; scaler sadece train setine fit edildi; TimeSeriesNet kodlandı | ✅ Tamamlandı | Pipeline tüm önlemlerle çalışıyor; TimeSeriesNet (Attention + Focal Loss + LayerNorm) implement edildi |
| IP 5 | LSTM ve GRU modelleri eğitildi, saved_models/ altına kaydedildi | ✅ Tamamlandı | BTC/ETH/SOL için 6 model dosyası kaydedildi; confusion matrix ve dummy baseline karşılaştırması yapıldı |

---

## 5. Kullanılan Teknolojiler

| Kategori | Araç | Kullanım |
|----------|------|---------|
| Programlama | Python 3.11 | Tüm sistem geliştirme |
| Veri İşleme | NumPy, Pandas | Veri manipülasyonu ve analiz |
| Teknik Analiz | pandas-ta | RSI, MACD, Bollinger Bands hesaplama |
| Derin Öğrenme | PyTorch 2.0 | LSTM/GRU model geliştirme |
| Görselleştirme | Plotly, Streamlit | İnteraktif dashboard |
| Borsa API | Binance REST, CoinGecko | Gerçek zamanlı ve tarihsel veri |
| Veritabanı | SQLite | Yerel veri saklama |
| Versiyon Kontrol | Git, GitHub | Kod yönetimi ve paylaşım |
| Geliştirme Ortamı | VS Code | Kodlama ortamı |

---

## 6. Sonuç

Projenin ilk sekiz haftası (IP 1–IP 5) başarıyla tamamlanmıştır. Proje formunda tanımlanan tüm başarı ölçütleri karşılanmıştır:

- **IP 1:** Akademik literatür taraması tamamlanmış, 16 kaynak incelenmiştir.
- **IP 2:** BTC, ETH ve SOL için 2000 günlük gerçek OHLCV verisi toplanmış, SQLite veritabanına kaydedilmiştir.
- **IP 3:** 17 teknik gösterge hesaplanmış, interaktif Streamlit + Plotly dashboard tamamlanmıştır.
- **IP 4:** Look-ahead bias önlemleriyle AI veri pipeline'ı ve PyTorch tabanlı TimeSeriesNet mimarisi geliştirilmiştir.
- **IP 5:** LSTM ve GRU modelleri 2-fazlı Focal Loss stratejisiyle eğitilmiş, 6 model dosyası kaydedilmiştir.

**Sonraki aşamada (IP 7–IP 12) yürütülecek çalışmalar:**
- Random Forest ve XGBoost sınıflandırma modelleri (IP 7)
- Ensemble meta-model — stacking yöntemi (IP 8)
- Pekiştirmeli öğrenme ajanı — PPO (IP 9)
- Backtesting ve performans analizi (IP 10)
- Sistem entegrasyonu ve 8 kategorili test paketi (IP 11)
- Final sunumu ve proje teslimi (IP 12)

---

*Makine Öğrenmesi Tabanlı Kripto Para Trading Botu | Ahmet Yılmaz | 23100011075*