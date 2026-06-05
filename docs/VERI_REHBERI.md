# Veri Rehberi (Baslangic Seviyesi)

Bu dokuman, projedeki veri yapisini sifirdan anlaman icin yazildi.

## Ne tahmin ediyoruz?

Her **gun** icin model su soruyu cevaplar:

> **Yarin kapanis fiyati, bugunku kapanistan yukari mi olacak?**

- `direction = 1` → Yukari (AL)
- `direction = 0` → Asagi (SAT / bekle)

Bu bir **siniflandirma** problemidir.

## Veri akisi (3 katman)

```
1. OHLCV (ham)          →  Binance gunluk mumlar
2. Ozellikler           →  Getiri, RSI, MACD... (fiyat seviyesi YOK)
3. Model girdileri      →  LSTM penceresi + XGBoost tek satir
```

### Katman 1: OHLCV

| Sutun | Anlam |
|-------|--------|
| open | Acilis |
| high | En yuksek |
| low | En dusuk |
| close | Kapanis |
| volume | Islem hacmi |

Dosya: `data/raw/{COIN}_ohlcv.csv`

### Katman 2: Ozellikler

Model **ham fiyat** ($50.000 gibi) gormez. Bunun yerine:

- `log_return_1d` — bugunku yuzde degisim
- `rsi_14` — asiri alim/satim gostergesi
- `bb_position` — fiyatin Bollinger bandindaki yeri (0–1)

Tam liste: `src/data/ml_config.py` → `FEATURE_COLUMNS`

Dosya: `data/ml/{COIN}/features.csv`

### Katman 3: Model formatlari

| Model | Girdi sekli | Olcekleme |
|-------|-------------|-----------|
| **LSTM** | Son 60 gun x 20 ozellik | StandardScaler (egitim gunlerinde fit) |
| **XGBoost** | Tek gun, 20 ozellik | Olcek yok (agac modeli) |

Her iki model **ayni gun** icin **ayni hedefi** kullanir — bu onemli.

Dosyalar:
- `data/ml/{COIN}/X_lstm.npy`, `y_lstm.npy`
- `data/ml/{COIN}/X_xgb.npy`, `y_xgb.npy`
- `data/ml/{COIN}/manifest.json` — metadata

## Nasil calistirilir?

```bash
# 1) (Istege bagli) SQLite'a tarihsel veri
python main.py setup
python main.py historical

# 2) ML veri setini olustur
python -m src.data.build_dataset --ticker BTC-USD

# Tum coinler
python -m src.data.build_dataset --all

# 3) Modelleri egit
python -m src.models.time_series_models      # LSTM + GRU
python -m src.models.ml_classification_models  # RF + XGBoost
```

## Veri analizi ciktisi

Pipeline calisinca konsolda gorursun:

- Tarih araligi ve satir sayisi
- Hedef dagilimi (% yukari / % asagi)
- Eksik veri kontrolu
- Ornek ozellik istatistikleri

## Onemli kurallar (veri sizintisi)

1. **Gelecegi kullanma** — ozellikler sadece bugune kadar olan bilgiyi kullanir.
2. **Hedef once** — `direction` olceklemeden once hesaplanir.
3. **Scaler sadece egitim** — ilk %70 gun uzerinde fit, sonra tum veriye uygulanir.
4. **Zaman sirasi** — train/val/test karistirilmaz; gelecekten gecmise bakilmaz.

## Klasor yapisi

```
data/
  raw/                    # Ham OHLCV
  ml/
    BTC-USD/
      features.csv        # Tum ozellikler + hedef
      X_lstm.npy          # LSTM girdisi
      X_xgb.npy           # XGBoost girdisi
      manifest.json       # Ozet bilgi
  processed/              # Eski kod uyumlulugu (otomatik uretilir)
```

## Sik sorulan sorular

**S: Neden iki farkli veri yolu vardi?**  
Eski projede SQLite botu ile AI pipeline ayri gelismisti. Yeni yapi tek merkezden uretir.

**S: XGBoost neden olceklenmiyor?**  
Agac modelleri fiyat seviyesinden etkilenmez; duragan ozellikler yeterli.

**S: LSTM neden 60 gun?**  
Son 2 ay civarindaki paternleri ogrenir. `ml_config.py` → `WINDOW_SIZE` ile degistirilebilir.
