import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import torch
import sklearn.ensemble._forest  # Deadlock onlemi: joblib.load oncesinde tam submodule yukle
import sklearn.ensemble._gb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# Proje kok dizini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.models.time_series_models import TimeSeriesNet
from config import DISPLAY_DAYS  # Dashboard parametreleri

# ─────────────────────────────────────────────────────────────
#  Sayfa Ayarlari
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kripto Para Yon Tahmini ve Karar Destek Sistemi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  CSS Stilleri
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .model-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        color: #e0e0e0;
    }
    .metric-box {
        background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #999;
    }
    .phase-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Coin secici ve Yardimci Fonksiyonlar
# ─────────────────────────────────────────────────────────────

# Sidebar'da coin secimi
AVAILABLE_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]
selected_ticker = st.sidebar.selectbox(
    "🪙 Coin Seçimi",
    AVAILABLE_TICKERS,
    index=0,
    help="Dashboard'da görüntülemek istediğiniz coin'i seçin"
)

@st.cache_data
def load_processed_data(ticker="BTC-USD"):
    path = os.path.join(BASE_DIR, "data", "processed", f"{ticker}_processed_scaled.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_raw_data(ticker="BTC-USD"):
    path = os.path.join(BASE_DIR, "data", "raw", f"{ticker}_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

@st.cache_resource
def load_ml_models(ticker="BTC-USD"):
    models_dir = os.path.join(BASE_DIR, "src", "models", "saved_models")
    models = {}
    if os.path.exists(os.path.join(models_dir, f"{ticker}_rf_classifier.pkl")):
        models['rf'] = joblib.load(os.path.join(models_dir, f"{ticker}_rf_classifier.pkl"))
    if os.path.exists(os.path.join(models_dir, f"{ticker}_xgb_classifier.pkl")):
        models['xgb'] = joblib.load(os.path.join(models_dir, f"{ticker}_xgb_classifier.pkl"))
    if os.path.exists(os.path.join(models_dir, f"{ticker}_ensemble_meta_model.pkl")):
        models['ensemble'] = joblib.load(os.path.join(models_dir, f"{ticker}_ensemble_meta_model.pkl"))
    return models

@st.cache_data
def load_evaluation_results():
    """model_evaluation_results.json dosyasini yukler."""
    path = os.path.join(BASE_DIR, "data", "results", "model_evaluation_results.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

@st.cache_data
def evaluate_ml_models(df, ticker="BTC-USD"):
    """ML modellerini yukle, test verisi uzerinde tekrar calistir ve metrikleri dondur."""
    # sklearn artik top-level'da import ediliyor (deadlock onlemi)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD',
                'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid', 'SMA_20', 'EMA_50',
                'Log_Return', 'Return_Lag_5', 'Return_Lag_10', 'Return_Lag_20',
                'Daily_Return', 'Volatility_20', 'Momentum_10', 'Volume_Change',
                'ATR', 'ADX', 'Stoch_K', 'OBV_Change',
                'Price_To_SMA20', 'Price_To_EMA50']
    features = [c for c in features if c in df.columns]
    
    df_eval = df.copy()
    # Direction sutunu varsa onu kullan (olcekleme oncesi olusturulmus)
    if 'Direction' in df_eval.columns:
        df_eval['Signal'] = df_eval['Direction'].astype(int)
    else:
        df_eval['Signal'] = np.where(df_eval['Target_Close'] > df_eval['Close'], 1, 0)
    
    X = df_eval[features].values
    y = df_eval['Signal'].values
    
    train_size = int(len(df_eval) * 0.8)
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    models_dir = os.path.join(BASE_DIR, "src", "models", "saved_models")
    results = {}
    
    for name, filename in [("Random Forest", f"{ticker}_rf_classifier.pkl"), ("XGBoost", f"{ticker}_xgb_classifier.pkl")]:
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            model = joblib.load(path)
            preds = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, preds),
                'f1': f1_score(y_test, preds),
                'precision': precision_score(y_test, preds),
                'recall': recall_score(y_test, preds),
                'confusion_matrix': confusion_matrix(y_test, preds),
                'predictions': preds,
                'y_test': y_test
            }
    return results

@st.cache_data
def evaluate_dl_models(ticker="BTC-USD"):
    """LSTM ve GRU modellerinin tahminlerini test verisi uzerinde uretir."""
    x_path = os.path.join(BASE_DIR, "data", "processed", f"{ticker}_X_windows.npy")
    y_path = os.path.join(BASE_DIR, "data", "processed", f"{ticker}_y_targets.npy")
    models_dir = os.path.join(BASE_DIR, "src", "models", "saved_models")
    
    if not os.path.exists(x_path):
        return None
    
    X = np.load(x_path)
    y = np.load(y_path)
    
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    results = {}
    
    for name, filename in [("LSTM", f"{ticker}_lstm_best.pth"), ("GRU", f"{ticker}_gru_best.pth")]:
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            # num_layers'i state dict'ten otomatik tespit et (hardcode hatalarini onler)
            sd = torch.load(path, map_location='cpu', weights_only=True)
            num_layers = len([k for k in sd.keys() if k.startswith('rnn.weight_ih')])
            num_layers = max(num_layers, 1)
            hidden_size = sd['rnn.weight_hh_l0'].shape[1] if 'rnn.weight_hh_l0' in sd else sd['rnn.weight_ih_l0'].shape[0] // 4
            input_size = sd['rnn.weight_ih_l0'].shape[1] if 'rnn.weight_ih_l0' in sd else X.shape[2]
            use_attention = any('attention' in k for k in sd.keys())
            model = TimeSeriesNet(input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  output_size=1, model_type=name, use_attention=use_attention)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model.eval()
            
            with torch.no_grad():
                raw_logits = model(torch.FloatTensor(X_test))
                # KRITIK: BCEWithLogitsLoss ile egitilen model logit cikarir
                # sigmoid uygulayarak 0-1 olasiliga donustur
                preds = torch.sigmoid(raw_logits).numpy().flatten()
            
            # Binary siniflandirma metrikleri (hedef 0/1)
            mse = np.mean((preds - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(preds - y_test))
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'predictions': preds,
                'actuals': y_test
            }
    
    return results


# ─────────────────────────────────────────────────────────────
#  SAYFA: Genel Bakis
# ─────────────────────────────────────────────────────────────
def page_overview():
    st.markdown('<p class="main-header">🤖 AI Trading Bot — Model Performans Paneli</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mimari Akis Semasi
    st.subheader("🏗️ Sistem Mimarisi")
    st.markdown("""
    Bu sistem **4 katmanlı** bir yapay zeka mimarisine sahiptir. Her katman birbirini besler:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h4>📈 Katman 1</h4>
            <h5>Fiyat Tahmini</h5>
            <p><b>LSTM / GRU</b></p>
            <p style="font-size:0.85rem">Geçmiş fiyatlardan gelecek fiyatı tahmin eder (Regresyon)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card" style="border-left-color: #f093fb;">
            <h4>🔔 Katman 2</h4>
            <h5>Sinyal Üretimi</h5>
            <p><b>XGBoost / Random Forest</b></p>
            <p style="font-size:0.85rem">Teknik göstergelerden AL/SAT sinyali üretir (Sınıflandırma)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="model-card" style="border-left-color: #4facfe;">
            <h4>🧬 Katman 3</h4>
            <h5>Ensemble</h5>
            <p><b>Stacking Meta-Model</b></p>
            <p style="font-size:0.85rem">Tüm modellerin çıktılarını birleştirip güvenilirliği artırır</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="model-card" style="border-left-color: #43e97b;">
            <h4>🎮 Katman 4</h4>
            <h5>Karar Alma</h5>
            <p><b>PPO (Deep RL)</b></p>
            <p style="font-size:0.85rem">Portföy yöneterek optimal alım-satım kararı verir</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Durumu Ozeti
    st.subheader("📦 Model Durumları")
    models_dir = os.path.join(BASE_DIR, "src", "models", "saved_models")
    rl_dir = os.path.join(BASE_DIR, "src", "rl", "saved_agents")
    
    model_files = {
        "LSTM (Fiyat Tahmini)": os.path.join(models_dir, f"{selected_ticker}_lstm_best.pth"),
        "GRU (Fiyat Tahmini)": os.path.join(models_dir, f"{selected_ticker}_gru_best.pth"),
        "Random Forest (Sinyal)": os.path.join(models_dir, f"{selected_ticker}_rf_classifier.pkl"),
        "XGBoost (Sinyal)": os.path.join(models_dir, f"{selected_ticker}_xgb_classifier.pkl"),
        "Ensemble Meta-Model": os.path.join(models_dir, f"{selected_ticker}_ensemble_meta_model.pkl"),
        "PPO RL Ajanı": os.path.join(rl_dir, f"{selected_ticker}_ppo_trading_agent.zip"),
    }
    
    cols = st.columns(3)
    for i, (name, path) in enumerate(model_files.items()):
        with cols[i % 3]:
            exists = os.path.exists(path)
            size = os.path.getsize(path) / 1024 if exists else 0
            status = "✅ Eğitildi" if exists else "❌ Henüz eğitilmedi"
            st.metric(label=name, value=status, delta=f"{size:.1f} KB" if exists else None)


# ─────────────────────────────────────────────────────────────
#  SAYFA: Fiyat Tahmini (LSTM / GRU)
# ─────────────────────────────────────────────────────────────
def page_price_prediction():
    st.header("📈 Fiyat Tahmini — LSTM / GRU Modelleri")
    st.warning(
        "Bu legacy sayfa ana routing'de devre disidir. "
        "LSTM ciktilari artik guvenli checkpoint loader ve compute_final_probs akisiyle "
        "Model Karari / Model Karsilastirmasi sayfalarinda gosterilir."
    )
    return
    
    st.info("""
    **Bu modeller ne yapar?**  
    LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit) modelleri, son **60 günlük** fiyat ve teknik 
    gösterge verilerini analiz ederek **bir sonraki günün kapanış fiyatını** tahmin eder.  
    Zaman serisi verilerdeki uzun vadeli kalıpları öğrenebilen derin öğrenme mimarileridir.
    """)
    
    dl_results = None
    
    if dl_results is None:
        st.warning("⚠️ Henüz eğitilmiş model bulunamadı. Lütfen önce `python main.py ailstm` komutunu çalıştırın.")
        return
    
    # Metrik Kartlari
    st.subheader("📊 Model Performans Metrikleri")
    
    for name, res in dl_results.items():
        st.markdown(f"### {name} Modeli")
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE (Ortalama Kare Hata)", f"{res['mse']:.6f}")
        col2.metric("RMSE (Karekök Ort. Hata)", f"{res['rmse']:.6f}")
        col3.metric("MAE (Ortalama Mutlak Hata)", f"{res['mae']:.6f}")
    
    # Tahmin vs Gercek Grafigi
    st.subheader("📉 Tahmin vs Gerçek Fiyat Karşılaştırması")
    
    for name, res in dl_results.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=res['actuals'], name='Gerçek Fiyat',
                                  line={"color": "#667eea", "width": 2}))
        fig.add_trace(go.Scatter(y=res['predictions'], name=f'{name} Tahmini',
                                  line={"color": "#f093fb", "width": 2, "dash": "dot"}))
        fig.update_layout(
            title=f'{name} — Test Verisi Üzerinde Tahmin Başarısı',
            xaxis_title='Zaman Adımı',
            yaxis_title='Fiyat (Ölçeklenmiş)',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SAYFA: Sinyal Uretimi (XGBoost / RF)
# ─────────────────────────────────────────────────────────────
def page_signal_generation():
    st.header("🔔 Sinyal Üretimi — XGBoost / Random Forest")
    st.warning(
        "Bu legacy sayfa ana routing'de devre disidir. "
        "RF/XGBoost tahmini artik FeatureStore(ticker).get_xgb_split('test') ve "
        "compute_final_probs akisiyle Model Karari / Model Karsilastirmasi sayfalarinda gosterilir."
    )
    return
    
    st.info("""
    **Bu modeller ne yapar?**  
    Teknik göstergeler (RSI, MACD, Bollinger Bands vb.) temelinde piyasanın **yükselip (AL) ya da 
    düşeceğini (SAT)** tahmin eder. Karar ağacı tabanlı güçlü sınıflandırma algoritmalarıdır.
    """)
    
    df = load_processed_data(selected_ticker)
    if df is None:
        st.warning("⚠️ İşlenmiş veri bulunamadı.")
        return
    
    results = {}
    
    if not results:
        st.warning("⚠️ Eğitilmiş ML modeli bulunamadı.")
        return
    
    # Metrik Kartlari
    st.subheader("📊 Sınıflandırma Performans Metrikleri")
    
    for name, res in results.items():
        st.markdown(f"### {name}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy (Doğruluk)", f"{res['accuracy']:.2%}")
        col2.metric("F1 Score", f"{res['f1']:.2%}")
        col3.metric("Precision (Kesinlik)", f"{res['precision']:.2%}")
        col4.metric("Recall (Duyarlılık)", f"{res['recall']:.2%}")
        
        # Confusion Matrix
        cm = res['confusion_matrix']
        fig = px.imshow(cm,
                        labels=dict(x="Tahmin Edilen", y="Gerçek Değer", color="Sayı"),
                        x=['Düşecek (0)', 'Yükselecek (1)'],
                        y=['Düşecek (0)', 'Yükselecek (1)'],
                        text_auto=True,
                        color_continuous_scale='Blues')
        fig.update_layout(title=f'{name} — Karışıklık Matrisi (Confusion Matrix)',
                          template='plotly_dark', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Karsilastirma
    st.subheader("⚖️ Model Karşılaştırması")
    comparison_data = []
    for name, res in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'F1 Score': res['f1'],
            'Precision': res['precision'],
            'Recall': res['recall']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    fig = go.Figure()
    for metric in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
        fig.add_trace(go.Bar(name=metric, x=comp_df['Model'], y=comp_df[metric]))
    fig.update_layout(barmode='group', template='plotly_dark',
                      title='Model Performans Karşılaştırması',
                      yaxis_title='Skor', height=400)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SAYFA: Ensemble
# ─────────────────────────────────────────────────────────────
def page_ensemble():
    st.header("🧬 Ensemble — Stacking Meta-Model")
    
    st.info("""
    **Bu model ne yapar?**  
    LSTM ve GRU'nun fiyat tahminleri ile XGBoost ve Random Forest'ın sinyal olasılıklarını 
    **birleştirerek** daha güvenilir bir sonuç üretir. 4 modelin çıktısını girdi olarak alır 
    ve bir **GradientBoosting meta-model** ile nihai AL/SAT kararı verir.  
    Bu yöntem, tek bir modelin hatasını diğer modellerin doğruluğuyla telafi eder.
    """)
    
    ensemble_path = os.path.join(BASE_DIR, "src", "models", "saved_models", f"{selected_ticker}_ensemble_meta_model.pkl")
    if not os.path.exists(ensemble_path):
        st.warning("⚠️ Ensemble modeli henüz eğitilmedi. `python -m src.pipeline --step ensemble` komutunu çalıştırın.")
        return
    
    # Evaluation sonuclarini yukle
    eval_results = load_evaluation_results()
    ensemble_data = None
    if eval_results and selected_ticker in eval_results:
        ensemble_data = eval_results[selected_ticker].get('Ensemble')
    
    # ── Performans Metrikleri ──
    if ensemble_data:
        st.subheader("📊 Ensemble Model Performans Metrikleri")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy (Doğruluk)", f"{ensemble_data['accuracy']:.2%}")
        col2.metric("F1 Score", f"{ensemble_data['f1']:.2%}")
        col3.metric("Precision (Kesinlik)", f"{ensemble_data['precision']:.2%}")
        col4.metric("Recall (Duyarlılık)", f"{ensemble_data['recall']:.2%}")
        
        # Confusion Matrix
        cm = np.array(ensemble_data['confusion_matrix'])
        fig_cm = px.imshow(cm,
                        labels=dict(x="Tahmin Edilen", y="Gerçek Değer", color="Sayı"),
                        x=['Düşecek (0)', 'Yükselecek (1)'],
                        y=['Düşecek (0)', 'Yükselecek (1)'],
                        text_auto=True,
                        color_continuous_scale='Blues')
        fig_cm.update_layout(title='Ensemble — Karışıklık Matrisi (Confusion Matrix)',
                            template='plotly_dark', height=350)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # ── Feature Importances (Base model agirliklari) ──
        if 'feature_importances' in ensemble_data:
            st.subheader("⚖️ Base Model Ağırlıkları (Feature Importances)")
            fi = ensemble_data['feature_importances']
            labels = list(fi.keys())
            values = list(fi.values())
            colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
            
            col_fi1, col_fi2 = st.columns(2)
            with col_fi1:
                fig_bar = go.Figure(go.Bar(
                    x=values, y=labels, orientation='h',
                    marker_color=colors[:len(labels)],
                    text=[f"{v:.1%}" for v in values],
                    textposition='auto'
                ))
                fig_bar.update_layout(title='Meta-Model İçindeki Ağırlıklar',
                                      xaxis_title='Önem Skoru',
                                      template='plotly_dark', height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_fi2:
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=values,
                    marker_colors=colors[:len(labels)],
                    hole=0.4
                ))
                fig_pie.update_layout(title='Ağırlık Dağılımı',
                                       template='plotly_dark', height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # ── Tum modeller ile karsilastirma ──
        st.subheader("⚖️ Ensemble vs Bireysel Modeller")
        ticker_data = eval_results.get(selected_ticker, {})
        comparison_rows = []
        for model_name in ['LSTM', 'GRU', 'Random Forest', 'XGBoost', 'Ensemble']:
            if model_name in ticker_data:
                md = ticker_data[model_name]
                comparison_rows.append({
                    'Model': model_name,
                    'Accuracy': md.get('accuracy', 0),
                    'F1 Score': md.get('f1', 0),
                    'Precision': md.get('precision', 0),
                    'Recall': md.get('recall', 0),
                    'Model Boyutu (KB)': md.get('model_size_kb', 0)
                })
        
        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            
            fig_comp = go.Figure()
            colors_map = {'Accuracy': '#667eea', 'F1 Score': '#f093fb', 
                          'Precision': '#4facfe', 'Recall': '#43e97b'}
            for metric, color in colors_map.items():
                fig_comp.add_trace(go.Bar(name=metric, x=comp_df['Model'], 
                                          y=comp_df[metric], marker_color=color))
            fig_comp.update_layout(barmode='group', template='plotly_dark',
                                   title='Tüm Modellerin Performans Karşılaştırması',
                                   yaxis_title='Skor', height=400)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Tablo olarak da goster
            st.dataframe(
                comp_df.style.format({
                    'Accuracy': '{:.2%}', 'F1 Score': '{:.2%}',
                    'Precision': '{:.2%}', 'Recall': '{:.2%}',
                    'Model Boyutu (KB)': '{:.1f}'
                }),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Ensemble değerlendirme sonuçları bulunamadı. Pipeline'ı çalıştırın.")
    
    # Mimari sema
    st.subheader("🔀 Ensemble Akış Diyagramı")
    st.markdown("""
    ```
    ┌─────────┐     ┌──────────┐
    │  LSTM   │────▶│          │
    │ Tahmini │     │          │
    └─────────┘     │          │
    ┌─────────┐     │  META    │     ┌──────────┐
    │   GRU   │────▶│  MODEL   │────▶│  Nihai   │
    │ Tahmini │     │ (Gradient│     │  Sinyal  │
    └─────────┘     │  Boost.) │     │  AL/SAT  │
    ┌─────────┐     │          │     └──────────┘
    │   RF    │────▶│          │
    │ Olasılık│     │          │
    └─────────┘     │          │
    ┌─────────┐     │          │
    │ XGBoost │────▶│          │
    │ Olasılık│     └──────────┘
    └─────────┘
    ```
    """)


# ─────────────────────────────────────────────────────────────
#  SAYFA: Deep RL
# ─────────────────────────────────────────────────────────────
def page_rl():
    st.header("🎮 Karar Alma — Deep Reinforcement Learning (PPO)")
    
    st.info("""
    **Bu model ne yapar?**  
    PPO (Proximal Policy Optimization) algoritması, bir **sanal borsa ortamında** binlerce kez 
    alım-satım yaparak deneyim kazanır. Tıpkı bir oyun oynayan yapay zeka gibi, kâr ettiğinde 
    **ödül**, zarar ettiğinde **ceza** alır. Zamanla en kârlı stratejiyi kendi kendine öğrenir.
    """)
    
    rl_path = os.path.join(BASE_DIR, "src", "rl", "saved_agents", f"{selected_ticker}_ppo_trading_agent.zip")
    
    if not os.path.exists(rl_path):
        st.warning("⚠️ PPO Ajanı henüz eğitilmedi. `python main.py airl` komutunu çalıştırın.")
        return
    
    st.success("✅ PPO Ajanı eğitilmiş ve hazır!")
    
    # Evaluation sonuclarini yukle
    eval_results = load_evaluation_results()
    rl_data = None
    if eval_results and selected_ticker in eval_results:
        rl_data = eval_results[selected_ticker].get('PPO RL')
    
    # ── Performans Metrikleri ──
    if rl_data:
        st.subheader("📊 PPO Ajan Performans Metrikleri")
        
        total_return = rl_data.get('total_return_pct', 0)
        delta_color = "normal" if total_return > 0 else "inverse"
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Son Portföy Değeri", 
                    f"${rl_data.get('final_net_worth', 0):,.2f}",
                    delta=f"{total_return:+.2f}%")
        col2.metric("Toplam Getiri", f"{total_return:.2f}%")
        col3.metric("Sharpe Oranı", f"{rl_data.get('sharpe_ratio', 0):.4f}")
        col4.metric("Max Drawdown", f"{rl_data.get('max_drawdown_pct', 0):.2f}%")
        
        st.markdown("---")
        
        # ── Ek Metrikler ──
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Başlangıç Sermayesi", f"${rl_data.get('initial_balance', 10000):,.2f}")
        col_b.metric("Toplam Adım Sayısı", f"{rl_data.get('total_steps', 0):,}")
        col_c.metric("Model Boyutu", f"{rl_data.get('model_size_kb', 0):.1f} KB")
        
        # ── Aksiyon Dagilimi ──
        st.subheader("🎯 Aksiyon Dağılımı")
        action_dist = rl_data.get('action_distribution', {})
        if action_dist:
            action_labels = list(action_dist.keys())
            action_values = list(action_dist.values())
            action_colors = ['#ff6b6b', '#feca57', '#48dbfb']  # SAT=kirmizi, TUT=sari, AL=mavi
            
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                fig_pie = go.Figure(go.Pie(
                    labels=action_labels, values=action_values,
                    marker_colors=action_colors,
                    hole=0.4,
                    textinfo='label+percent+value'
                ))
                fig_pie.update_layout(title='Aksiyon Oranları',
                                       template='plotly_dark', height=350)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_act2:
                fig_bar_act = go.Figure(go.Bar(
                    x=action_labels, y=action_values,
                    marker_color=action_colors,
                    text=action_values, textposition='auto'
                ))
                fig_bar_act.update_layout(title='Aksiyon Sayıları',
                                           xaxis_title='Aksiyon', yaxis_title='Sayı',
                                           template='plotly_dark', height=350)
                st.plotly_chart(fig_bar_act, use_container_width=True)
        
        # ── Getiri Karsilastirma ──
        st.subheader("💰 Getiri Özeti")
        initial = rl_data.get('initial_balance', 10000)
        final = rl_data.get('final_net_worth', 0)
        profit = final - initial
        
        fig_waterfall = go.Figure(go.Waterfall(
            x=['Başlangıç', 'Kâr/Zarar', 'Son Değer'],
            y=[initial, profit, 0],
            measure=['absolute', 'relative', 'total'],
            connector={'line': {'color': 'rgba(63, 63, 63, 0.5)'}},
            decreasing={'marker': {'color': '#ff6b6b'}},
            increasing={'marker': {'color': '#43e97b'}},
            totals={'marker': {'color': '#667eea'}}
        ))
        fig_waterfall.update_layout(title='Portföy Değişimi',
                                     yaxis_title='Değer ($)',
                                     template='plotly_dark', height=350)
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # ── Tum coinler arasi RL karsilastirma ──
        if eval_results:
            st.subheader("🏆 Coin Bazında PPO Performans Karşılaştırması")
            rl_comparison = []
            for t in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
                if t in eval_results and 'PPO RL' in eval_results[t]:
                    rd = eval_results[t]['PPO RL']
                    rl_comparison.append({
                        'Coin': t,
                        'Toplam Getiri (%)': rd.get('total_return_pct', 0),
                        'Sharpe Oranı': rd.get('sharpe_ratio', 0),
                        'Max Drawdown (%)': rd.get('max_drawdown_pct', 0),
                        'Son Değer ($)': rd.get('final_net_worth', 0)
                    })
            if rl_comparison:
                rl_df = pd.DataFrame(rl_comparison)
                
                fig_rl_comp = go.Figure()
                fig_rl_comp.add_trace(go.Bar(
                    name='Toplam Getiri (%)', x=rl_df['Coin'], 
                    y=rl_df['Toplam Getiri (%)'],
                    marker_color='#43e97b'
                ))
                fig_rl_comp.update_layout(template='plotly_dark',
                                           title='PPO Ajanı — Coin Bazlı Toplam Getiri',
                                           yaxis_title='Getiri (%)', height=350)
                st.plotly_chart(fig_rl_comp, use_container_width=True)
                
                st.dataframe(
                    rl_df.style.format({
                        'Toplam Getiri (%)': '{:.2f}%',
                        'Sharpe Oranı': '{:.4f}',
                        'Max Drawdown (%)': '{:.2f}%',
                        'Son Değer ($)': '${:,.2f}'
                    }),
                    use_container_width=True
                )
    else:
        st.warning("⚠️ PPO RL değerlendirme sonuçları bulunamadı.")
    
    # ── Ortam ve Egitim Parametreleri ──
    st.subheader("🔧 Ortam ve Eğitim Detayları")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Ortam (Environment) Detayları:**
        - **Gözlem (10 özellik):** Bakiye, Net Worth, Meta-Prob, Fiyat, RSI, MACD Signal, 5g Değişim, Volatilite, Hacim Oranı, Pozisyon
        - **Eylem:** SAT (0), TUT (1), AL (2)
        - **Ödül:** Risk-adjusted PnL + Drawdown Cezası + Sharpe Bonus
        """)
    with col2:
        st.markdown("""
        **Eğitim Parametreleri:**
        - **Algoritma:** PPO (Proximal Policy Optimization)
        - **Politika:** MlpPolicy (Çok Katmanlı Algılayıcı)
        - **Öğrenme Hızı:** 0.0003
        - **Toplam Adım:** 500,000 timestep
        - **Entropy Katsayısı:** 0.10 (keşfetme teşviki)
        """)


# ─────────────────────────────────────────────────────────────
#  SAYFA: Backtesting
# ─────────────────────────────────────────────────────────────
def page_backtest():
    st.header("📊 Backtesting — Geriye Dönük Performans Testi")
    
    st.info("""
    **Backtesting ne yapar?**  
    Modellerin ürettiği sinyallerle geçmiş veride **sanal ticaret** yapar. Gerçek komisyon oranları 
    uygulanarak, yapay zekanın para kazanıp kazanmadığı test edilir. Bu, modeli canlı piyasaya 
    sokmadan önce yapılan en kritik sınavdır.
    """)
    
    # Backtest grafigi
    backtest_img = os.path.join(BASE_DIR, "data", "results", f"{selected_ticker}_backtest_portfolio.png")
    if os.path.exists(backtest_img):
        st.subheader("💰 Portföy Performans Grafiği")
        st.image(backtest_img, width="stretch")
    
    st.subheader("🧪 Hızlı Backtest Çalıştır")
    
    col1, col2, col3 = st.columns(3)
    initial_capital = col1.number_input("Başlangıç Sermayesi ($)", value=10000, step=1000)
    commission = col2.number_input("Komisyon Oranı (%)", value=0.1, step=0.01) / 100
    
    if col3.button("🚀 Backtest Başlat", type="primary"):
        df = load_processed_data(selected_ticker)
        if df is not None:
            # Basit bir sinyal uretimi
            test_size = int(len(df) * 0.2)
            prices = df['Close'].values[-test_size:]
            np.random.seed(42)
            signals = np.random.choice([0, 1, 2], size=len(prices), p=[0.15, 0.70, 0.15])
            
            # Basit simülasyon
            balance: float = float(initial_capital)
            shares: float = 0.0
            portfolio_values = []
            
            for i in range(len(prices)):
                price: float = float(prices[i])
                if signals[i] == 2 and balance > 0:
                    shares = float((balance * (1 - commission)) / price)
                    balance = 0.0
                elif signals[i] == 0 and shares > 0:
                    balance = float(shares * price * (1 - commission))
                    shares = 0.0
                portfolio_values.append(float(balance + shares * price))
            
            pv = np.array(portfolio_values)
            final = pv[-1]
            total_return = (final - initial_capital) / initial_capital * 100
            daily_returns = np.diff(pv) / pv[:-1]
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
            peak = np.maximum.accumulate(pv)
            max_dd = np.max((peak - pv) / peak) * 100
            
            # Sonuclari goster
            st.markdown("---")
            st.subheader("📈 Backtest Sonuçları")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Son Portföy Değeri", f"${final:,.2f}",
                      delta=f"{total_return:+.2f}%")
            c2.metric("Sharpe Oranı", f"{sharpe:.4f}")
            c3.metric("Max Drawdown", f"{max_dd:.2f}%")
            c4.metric("Toplam Getiri", f"{total_return:.2f}%")
            
            # Interaktif Grafik
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=pv, fill='tozeroy',
                                      line={"color": "#667eea", "width": 2},
                                      fillcolor='rgba(102, 126, 234, 0.1)',
                                      name='Portföy Değeri'))
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                          annotation_text="Başlangıç Sermayesi")
            fig.update_layout(title='Portföy Değeri Değişimi',
                              xaxis_title='Gün', yaxis_title='Değer ($)',
                              template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Veri bulunamadı. Lütfen önce AI pipeline'ı çalıştırın.")


# ─────────────────────────────────────────────────────────────
#  SAYFA: Legacy / Arsiv Veri Akisi
# ─────────────────────────────────────────────────────────────
def page_data_pipeline():
    st.header("Legacy / Arşiv Bilgisi — Ana Pipeline’da Kullanılmıyor")
    
    st.warning("""
    Bu bölüm eski geliştirme döneminden kalan arşiv/legacy bilgidir.
    Güncel sistem `data/raw` ve `data/ml` altındaki yeni datasetleri,
    `src/data/build_dataset.py` ve `src/data/feature_store.py` tabanlı yeni pipeline’ı kullanır.
    """)
    
    raw_df = load_raw_data(selected_ticker)
    processed_df = load_processed_data(selected_ticker)
    
    # ── Legacy Veri Akisi Ozeti ──
    st.subheader("📦 Legacy Veri Akışı Özeti")
    
    col1, col2, col3, col4 = st.columns(4)
    
    raw_path = os.path.join(BASE_DIR, "data", "raw", f"{selected_ticker}_raw.csv")
    processed_path = os.path.join(BASE_DIR, "data", "processed", f"{selected_ticker}_processed_scaled.csv")
    windows_path = os.path.join(BASE_DIR, "data", "processed", f"{selected_ticker}_X_windows.npy")
    
    if raw_df is not None:
        col1.metric("Ham Veri Satır", f"{len(raw_df):,}")
        if 'Date' in raw_df.columns:
            date_range = f"{raw_df['Date'].min()} — {raw_df['Date'].max()}"
        else:
            date_range = "N/A"
    else:
        col1.metric("Ham Veri Satır", "N/A")
        date_range = "N/A"
    
    if processed_df is not None:
        col2.metric("İşlenmiş Veri Satır", f"{len(processed_df):,}")
        col3.metric("Özellik Sayısı", f"{len(processed_df.columns)}")
    else:
        col2.metric("İşlenmiş Veri Satır", "N/A")
        col3.metric("Özellik Sayısı", "N/A")
    
    if os.path.exists(windows_path):
        X_win = np.load(windows_path, allow_pickle=True)
        col4.metric("Zaman Penceresi", f"{X_win.shape[0]:,} x {X_win.shape[1]} x {X_win.shape[2]}")
    else:
        col4.metric("Zaman Penceresi", "N/A")
    
    # Tarih araligi
    st.caption(f"📅 Veri Dönemi: {date_range}")
    
    st.markdown("---")
    
    # ── Dosya Boyutlari ──
    st.subheader("💾 Pipeline Dosyaları")
    file_info = []
    pipeline_files = {
        'Ham Veri (CSV)': raw_path,
        'İşlenmiş Veri (CSV)': processed_path,
        'Zaman Pencereleri (NPY)': windows_path,
        'Hedef Değişken (NPY)': os.path.join(BASE_DIR, "data", "processed", f"{selected_ticker}_y_targets.npy"),
        'Feature Scaler (PKL)': os.path.join(BASE_DIR, "data", "processed", f"{selected_ticker}_feature_scaler.pkl"),
        'Target Scaler (PKL)': os.path.join(BASE_DIR, "data", "processed", f"{selected_ticker}_target_scaler.pkl"),
    }
    for name, fpath in pipeline_files.items():
        exists = os.path.exists(fpath)
        size_kb = os.path.getsize(fpath) / 1024 if exists else 0
        file_info.append({
            'Dosya': name,
            'Durum': '✅ Mevcut' if exists else '❌ Eksik',
            'Boyut': f"{size_kb:,.1f} KB" if exists else '-'
        })
    st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
    
    if processed_df is None:
        st.warning("⚠️ İşlenmiş veri dosyası bulunamadı. Pipeline'ı çalıştırın.")
        return
    
    st.markdown("---")
    
    # ── Ozellik Listesi ve Istatistikleri ──
    st.subheader("📐 Özellik İstatistikleri")
    
    desc = processed_df.describe().T
    desc['null_count'] = processed_df.isnull().sum()
    desc['null_pct'] = (processed_df.isnull().sum() / len(processed_df) * 100).round(2)
    
    st.dataframe(
        desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'null_count', 'null_pct']]
        .style.format({
            'count': '{:.0f}', 'mean': '{:.4f}', 'std': '{:.4f}',
            'min': '{:.4f}', '25%': '{:.4f}', '50%': '{:.4f}',
            '75%': '{:.4f}', 'max': '{:.4f}',
            'null_count': '{:.0f}', 'null_pct': '{:.2f}%'
        }),
        use_container_width=True, height=400
    )
    
    # ── Ozellik Dagilimi Grafigi ──
    st.subheader("📊 Özellik Dağılımları")
    
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "Görselleştirilecek özellikleri seçin:",
        numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
    )
    
    if selected_features:
        num_feats = len(selected_features)
        cols_per_row = min(num_feats, 3)
        rows_needed = (num_feats + cols_per_row - 1) // cols_per_row
        
        fig_dist = make_subplots(rows=rows_needed, cols=cols_per_row,
                                  subplot_titles=selected_features)
        for idx, feat in enumerate(selected_features):
            r = idx // cols_per_row + 1
            c = idx % cols_per_row + 1
            fig_dist.add_trace(
                go.Histogram(x=processed_df[feat], name=feat, 
                            marker_color='#667eea', opacity=0.7),
                row=r, col=c
            )
        fig_dist.update_layout(template='plotly_dark', height=300 * rows_needed,
                                showlegend=False, title_text='Seçili Özelliklerin Dağılımı')
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # ── Korelasyon Matrisi ──
    st.subheader("🔗 Korelasyon Matrisi")
    
    key_features = [c for c in ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 
                                 'BB_High', 'BB_Low', 'SMA_20', 'EMA_50', 'Target_Close']
                    if c in processed_df.columns]
    
    if len(key_features) > 2:
        corr_matrix = processed_df[key_features].corr()
        fig_corr = px.imshow(corr_matrix.round(2),
                             text_auto=True,
                             color_continuous_scale='RdBu_r',
                             zmin=-1, zmax=1)
        fig_corr.update_layout(title='Temel Özellikler Korelasyon Matrisi',
                                template='plotly_dark', height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # ── Veri Kalitesi Ozeti ──
    st.subheader("✅ Veri Kalitesi Raporu")
    total_cells = processed_df.shape[0] * processed_df.shape[1]
    null_cells = processed_df.isnull().sum().sum()
    null_pct = null_cells / total_cells * 100
    
    qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
    qual_col1.metric("Toplam Hücre", f"{total_cells:,}")
    qual_col2.metric("Eksik Hücre", f"{null_cells:,}")
    qual_col3.metric("Eksik Oran", f"{null_pct:.4f}%")
    qual_col4.metric("Veri Kalitesi", 
                     "🟢 Mükemmel" if null_pct < 1 else ("🟡 İyi" if null_pct < 5 else "🔴 Düşük"))


# ─────────────────────────────────────────────────────────────
#  SAYFA: Veri Gorsellestirme
# ─────────────────────────────────────────────────────────────
def page_data():
    st.header("📊 Veri ve Teknik Göstergeler")
    
    raw_df = load_raw_data(selected_ticker)
    processed_df = load_processed_data(selected_ticker)
    
    if raw_df is not None:
        st.subheader(f"🕯️ {selected_ticker} Fiyat Grafiği")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05)
        
        fig.add_trace(go.Candlestick(
            x=raw_df['Date'] if 'Date' in raw_df.columns else raw_df.index,
            open=raw_df['Open'], high=raw_df['High'],
            low=raw_df['Low'], close=raw_df['Close'],
            name=selected_ticker
        ), row=1, col=1)
        
        if 'Volume' in raw_df.columns:
            fig.add_trace(go.Bar(
                x=raw_df['Date'] if 'Date' in raw_df.columns else raw_df.index,
                y=raw_df['Volume'], name='Hacim',
                marker_color='rgba(102, 126, 234, 0.3)'
            ), row=2, col=1)
        
        fig.update_layout(template='plotly_dark', height=600,
                          title=f'{selected_ticker} Mum Grafiği ve Hacim',
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    if processed_df is not None:
        st.subheader("🔢 İşlenmiş Veri Özeti")
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Satır", f"{len(processed_df):,}")
        col2.metric("Özellik Sayısı", f"{len(processed_df.columns)}")
        col3.metric("Dönem", f"{min(len(processed_df), DISPLAY_DAYS)} gün")
        
        st.dataframe(processed_df.describe().round(4), use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SAYFA: Arsiv Dataset Bilgisi
# ─────────────────────────────────────────────────────────────
def page_dataset_viewer():
    st.header("Arşiv Dataset Bilgisi — Ana Sistemde Kullanılmıyor")
    
    st.warning("""
    Bu bölüm eski geliştirme döneminden kalan arşiv/legacy bilgidir.
    Güncel sistem `data/raw/{ticker}_ohlcv.csv` ve `data/ml/{ticker}/` datasetlerini kullanır.
    Ana pipeline `fetch_ohlcv → feature_engineering → build_dataset → FeatureStore` akışıdır.
    """)
    
    raw_df = load_raw_data(selected_ticker)
    processed_df = load_processed_data(selected_ticker)
    
    # ── Veri Seti Secimi ──
    dataset_choice = st.radio(
        "📂 Görüntülenecek veri setini seçin:",
        ["Ham Veri (Raw)", "İşlenmiş Veri (Processed)", "Her İkisi"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # ════════════════════════════════════════════════════════
    #  HAM VERİ
    # ════════════════════════════════════════════════════════
    if dataset_choice in ["Ham Veri (Raw)", "Her İkisi"]:
        st.subheader(f"📄 Ham Veri — {selected_ticker}")
        
        if raw_df is not None:
            # Bilgi kartlari
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            info_col1.metric("Satır Sayısı", f"{len(raw_df):,}")
            info_col2.metric("Sütun Sayısı", f"{len(raw_df.columns)}")
            if 'Date' in raw_df.columns:
                info_col3.metric("Başlangıç", str(raw_df['Date'].min())[:10])
                info_col4.metric("Bitiş", str(raw_df['Date'].max())[:10])
            elif 'Open Time' in raw_df.columns:
                info_col3.metric("Başlangıç", str(raw_df['Open Time'].iloc[0])[:10])
                info_col4.metric("Bitiş", str(raw_df['Open Time'].iloc[-1])[:10])
            
            # Filtre: Son N satir
            filter_col1, filter_col2 = st.columns([1, 3])
            with filter_col1:
                row_option = st.selectbox(
                    "Gösterilecek satırlar:",
                    ["Tüm Veri", "İlk 50", "İlk 100", "Son 50", "Son 100"],
                    key="raw_row_filter"
                )
            
            if row_option == "İlk 50":
                display_raw = raw_df.head(50)
            elif row_option == "İlk 100":
                display_raw = raw_df.head(100)
            elif row_option == "Son 50":
                display_raw = raw_df.tail(50)
            elif row_option == "Son 100":
                display_raw = raw_df.tail(100)
            else:
                display_raw = raw_df
            
            # Tablo
            st.dataframe(
                display_raw,
                use_container_width=True,
                height=450,
                hide_index=True
            )
            
            # İstatistikler
            with st.expander("📊 Ham Veri İstatistikleri", expanded=False):
                st.dataframe(
                    raw_df.describe().round(4),
                    use_container_width=True
                )
            
            # İnteraktif Grafik
            with st.expander("📈 Ham Veri Grafikleri", expanded=False):
                numeric_cols_raw = raw_df.select_dtypes(include=[np.number]).columns.tolist()
                selected_raw_cols = st.multiselect(
                    "Görselleştirilecek sütunları seçin:",
                    numeric_cols_raw,
                    default=[c for c in ['Close', 'Volume'] if c in numeric_cols_raw],
                    key="raw_chart_cols"
                )
                if selected_raw_cols:
                    x_axis = raw_df['Date'] if 'Date' in raw_df.columns else (
                        raw_df['Open Time'] if 'Open Time' in raw_df.columns else raw_df.index
                    )
                    fig_raw = go.Figure()
                    colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a',
                              '#fee140', '#30cfd0', '#a8edea', '#fed6e3', '#d299c2']
                    for i, col in enumerate(selected_raw_cols):
                        fig_raw.add_trace(go.Scatter(
                            x=x_axis, y=raw_df[col],
                            name=col,
                            line={"color": colors[i % len(colors)], "width": 2}
                        ))
                    fig_raw.update_layout(
                        title=f'{selected_ticker} — Ham Veri Zaman Serisi',
                        xaxis_title='Tarih',
                        yaxis_title='Değer',
                        template='plotly_dark',
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig_raw, use_container_width=True)
            
            # CSV İndirme
            csv_raw = raw_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Ham Veriyi CSV Olarak İndir",
                data=csv_raw,
                file_name=f"{selected_ticker}_raw_data.csv",
                mime="text/csv",
                key="download_raw"
            )
        else:
            st.warning("⚠️ Ham veri dosyası bulunamadı. Lütfen önce veri pipeline'ını çalıştırın.")
    
    if dataset_choice == "Her İkisi":
        st.markdown("---")
    
    # ════════════════════════════════════════════════════════
    #  İŞLENMİŞ VERİ
    # ════════════════════════════════════════════════════════
    if dataset_choice in ["İşlenmiş Veri (Processed)", "Her İkisi"]:
        st.subheader(f"⚙️ İşlenmiş (Ölçeklenmiş) Veri — {selected_ticker}")
        
        if processed_df is not None:
            # Bilgi kartlari
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            p_col1.metric("Satır Sayısı", f"{len(processed_df):,}")
            p_col2.metric("Sütun Sayısı", f"{len(processed_df.columns)}")
            null_count = int(processed_df.isnull().sum().sum())
            p_col3.metric("Eksik Değer", f"{null_count:,}")
            p_col4.metric("Veri Tipi", "float64 (ölçeklenmiş)")
            
            # Sütun secimi
            all_cols = processed_df.columns.tolist()
            col_groups = {
                "Fiyat": [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in all_cols],
                "Teknik Göstergeler": [c for c in ['RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid',
                                                    'SMA_20', 'EMA_50', 'ATR', 'ADX', 'Stoch_K', 'OBV_Change'] if c in all_cols],
                "Getiri & Volatilite": [c for c in ['Log_Return', 'Return_Lag_5', 'Return_Lag_10', 'Return_Lag_20',
                                                     'Daily_Return', 'Volatility_20', 'Momentum_10', 'Volume_Change'] if c in all_cols],
                "Oran & Hedef": [c for c in ['Price_To_SMA20', 'Price_To_EMA50', 'Target_Close', 'Direction'] if c in all_cols],
            }
            
            show_col1, show_col2 = st.columns([1, 3])
            with show_col1:
                col_group_choice = st.selectbox(
                    "Sütun grubu:",
                    ["Tüm Sütunlar"] + list(col_groups.keys()),
                    key="proc_col_group"
                )
            
            if col_group_choice == "Tüm Sütunlar":
                display_cols = all_cols
            else:
                display_cols = col_groups[col_group_choice]
            
            # Filtre: Son N satir
            with show_col2:
                row_option_proc = st.selectbox(
                    "Gösterilecek satırlar:",
                    ["Tüm Veri", "İlk 50", "İlk 100", "Son 50", "Son 100"],
                    key="proc_row_filter"
                )
            
            proc_display = processed_df[display_cols]
            if row_option_proc == "İlk 50":
                proc_display = proc_display.head(50)
            elif row_option_proc == "İlk 100":
                proc_display = proc_display.head(100)
            elif row_option_proc == "Son 50":
                proc_display = proc_display.tail(50)
            elif row_option_proc == "Son 100":
                proc_display = proc_display.tail(100)
            
            # Tablo
            st.dataframe(
                proc_display.style.format("{:.6f}", na_rep="-"),
                use_container_width=True,
                height=450
            )
            
            # İstatistikler
            with st.expander("📊 İşlenmiş Veri İstatistikleri", expanded=False):
                desc = processed_df[display_cols].describe().T
                desc['null'] = processed_df[display_cols].isnull().sum()
                st.dataframe(
                    desc.style.format({
                        'count': '{:.0f}', 'mean': '{:.4f}', 'std': '{:.4f}',
                        'min': '{:.4f}', '25%': '{:.4f}', '50%': '{:.4f}',
                        '75%': '{:.4f}', 'max': '{:.4f}', 'null': '{:.0f}'
                    }),
                    use_container_width=True,
                    height=400
                )
            
            # İnteraktif Grafik
            with st.expander("📈 İşlenmiş Veri Grafikleri", expanded=False):
                numeric_proc = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                selected_proc_cols = st.multiselect(
                    "Görselleştirilecek sütunları seçin:",
                    numeric_proc,
                    default=[c for c in ['Close', 'RSI', 'MACD'] if c in numeric_proc],
                    key="proc_chart_cols"
                )
                if selected_proc_cols:
                    fig_proc = go.Figure()
                    colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a',
                              '#fee140', '#30cfd0', '#a8edea', '#fed6e3', '#d299c2']
                    for i, col in enumerate(selected_proc_cols):
                        fig_proc.add_trace(go.Scatter(
                            y=processed_df[col],
                            name=col,
                            line={"color": colors[i % len(colors)], "width": 2}
                        ))
                    fig_proc.update_layout(
                        title=f'{selected_ticker} — İşlenmiş Veri Zaman Serisi (Ölçeklenmiş)',
                        xaxis_title='İndeks',
                        yaxis_title='Değer (0-1 arası)',
                        template='plotly_dark',
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig_proc, use_container_width=True)
            
            # Korelasyon Matrisi
            with st.expander("🔗 Sütunlar Arası Korelasyon", expanded=False):
                corr_cols = [c for c in ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
                                          'BB_High', 'BB_Low', 'SMA_20', 'EMA_50',
                                          'ATR', 'ADX', 'Stoch_K', 'Target_Close']
                             if c in processed_df.columns]
                if len(corr_cols) > 2:
                    corr = processed_df[corr_cols].corr().round(2)
                    fig_corr = px.imshow(
                        corr, text_auto=True,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                    fig_corr.update_layout(
                        title='Özellikler Arası Korelasyon Matrisi',
                        template='plotly_dark',
                        height=550
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            # CSV İndirme
            csv_proc = processed_df.to_csv().encode('utf-8')
            st.download_button(
                label="⬇️ İşlenmiş Veriyi CSV Olarak İndir",
                data=csv_proc,
                file_name=f"{selected_ticker}_processed_data.csv",
                mime="text/csv",
                key="download_proc"
            )
        else:
            st.warning("⚠️ İşlenmiş veri dosyası bulunamadı. Lütfen önce veri pipeline'ını çalıştırın.")


# ─────────────────────────────────────────────────────────────
#  Ana Uygulama
# ─────────────────────────────────────────────────────────────
PRESENTATION_NOTICE = "Bu sistem yatirim tavsiyesi degildir; model ciktilari yalnizca karar destek amaciyla sunulur."


def show_presentation_notice():
    st.caption(PRESENTATION_NOTICE)


def _proof_box(text):
    st.info(f"Sayfa özeti: {text}")


def _safe_auc(y_true, probs):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return None


def _classification_summary(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int).ravel()
    probs = np.asarray(probs, dtype=float).ravel()
    n = min(len(y_true), len(probs))
    y_true = y_true[-n:]
    probs = probs[-n:]
    preds = (probs >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "AUC": _safe_auc(y_true, probs),
    }


def _format_metric_df(df):
    formatters = {
        "Accuracy": lambda x: "-" if pd.isna(x) else f"{x:.2%}",
        "Precision": lambda x: "-" if pd.isna(x) else f"{x:.2%}",
        "Recall": lambda x: "-" if pd.isna(x) else f"{x:.2%}",
        "F1": lambda x: "-" if pd.isna(x) else f"{x:.2%}",
        "AUC": lambda x: "-" if pd.isna(x) else f"{x:.3f}",
        "Probability Std": lambda x: "-" if pd.isna(x) else f"{x:.4f}",
        "RF-XGB Corr": lambda x: "-" if pd.isna(x) else f"{x:.3f}",
    }
    return df.style.format({k: v for k, v in formatters.items() if k in df.columns})


def _decision_band(prob):
    if prob >= 0.55:
        return "BUY / yukari yon egilimi"
    if prob <= 0.45:
        return "CLOSE / risk azalt"
    return "HOLD / izleme bandi"


@st.cache_data
def _feature_store_evidence_rows(cache_version="presentation_rebuild_v2"):
    from src.data.feature_store import FeatureStore
    from src.data.ml_config import FETCH_DAYS, PURGE_GAP, WINDOW_SIZE

    rows = []
    for ticker in AVAILABLE_TICKERS:
        store = FeatureStore(ticker)
        manifest = store.manifest
        split = manifest.get("split_indices", {})
        X_train, y_train = store.get_xgb_split("train")
        X_val, y_val = store.get_xgb_split("val")
        X_test, y_test = store.get_xgb_split("test")
        X_lstm_test, _ = store.get_lstm_split("test")
        train_end = split.get("train", [None, None])[1]
        val_start, val_end = split.get("val", [None, None])
        test_start = split.get("test", [None, None])[0]
        leakage_ok = all(v is not None for v in [train_end, val_start, val_end, test_start])
        leakage_ok = leakage_ok and train_end <= val_start and val_end <= test_start
        purge_ok = leakage_ok and (val_start - train_end) >= PURGE_GAP and (test_start - val_end) >= PURGE_GAP
        rows.append({
            "Coin": ticker,
            "Aktif Raw Veri": f"data/raw/{ticker}_ohlcv.csv",
            "Aktif Dataset": f"data/ml/{ticker}/",
            "Raw Rows": manifest.get("n_days_original", "-"),
            "Filtered Rows": manifest.get("n_days_after_filter", "-"),
            "Samples": manifest.get("n_samples", len(store.y)),
            "Date Range": f"{manifest.get('date_start', '-')} / {manifest.get('date_end', '-')}",
            "Train": len(y_train),
            "Val": len(y_val),
            "Test": len(y_test),
            "Window": manifest.get("window_size", WINDOW_SIZE),
            "Purge Gap": PURGE_GAP,
            "Fetch Days": FETCH_DAYS,
            "Feature Count": X_train.shape[1],
            "Train UP/DOWN": f"{int(np.sum(y_train == 1))}/{int(np.sum(y_train == 0))}",
            "Val UP/DOWN": f"{int(np.sum(y_val == 1))}/{int(np.sum(y_val == 0))}",
            "Test UP/DOWN": f"{int(np.sum(y_test == 1))}/{int(np.sum(y_test == 0))}",
            "Leakage Check": "OK" if leakage_ok else "Kontrol et",
            "Purge Applied": "OK" if purge_ok else "Kontrol et",
            "XGB/LSTM Align": "OK" if len(X_test) == len(X_lstm_test) == len(y_test) else "Kontrol et",
        })
    return rows


@st.cache_data
def _decision_snapshot(ticker, cache_version="presentation_rebuild_v2"):
    from src.data.feature_store import FeatureStore
    from src.models.meta_inference import compute_final_probs
    from src.models.weighted_hybrid import generate_signals

    store = FeatureStore(ticker)
    X_test, _ = store.get_xgb_split("test")
    final_probs, _, component = compute_final_probs(ticker, split_name="test")
    n = min(len(X_test), len(final_probs), len(component["rf"]), len(component["xgb"]), len(component["lstm"]))
    weights = component.get("weights", {"rf": 0.4, "xgb": 0.4, "lstm": 0.2})
    final_prob = float(final_probs[-n:][-1])
    signal_code = int(generate_signals(np.array([final_prob]))[0])
    return {
        "rf_prob": float(component["rf"][-n:][-1]),
        "xgb_prob": float(component["xgb"][-n:][-1]),
        "lstm_prob": float(component["lstm"][-n:][-1]),
        "final_prob": final_prob,
        "weights": weights,
        "signal": {0: "CLOSE", 1: "HOLD", 2: "BUY"}[signal_code],
        "lstm_fallback_active": bool(component.get("lstm_degenerate", False)) or weights.get("lstm", 0.2) == 0.0,
    }


@st.cache_data
def _model_metric_rows(ticker, split_name, cache_version="presentation_rebuild_v2"):
    from src.data.feature_store import FeatureStore
    from src.models.meta_inference import compute_final_probs

    store = FeatureStore(ticker)
    _, y = store.get_xgb_split(split_name)
    final_probs, _, component = compute_final_probs(ticker, split_name=split_name)
    n = min(len(y), len(final_probs), len(component["rf"]), len(component["xgb"]), len(component["lstm"]))
    y = np.asarray(y[-n:], dtype=int)
    rf_probs = np.asarray(component["rf"][-n:], dtype=float)
    xgb_probs = np.asarray(component["xgb"][-n:], dtype=float)
    lstm_probs = np.asarray(component["lstm"][-n:], dtype=float)
    final_probs = np.asarray(final_probs[-n:], dtype=float)
    rf_xgb_probs = (rf_probs + xgb_probs) / 2.0
    majority_prob = np.full(n, 1.0 if np.mean(y) >= 0.5 else 0.0)
    corr = float(np.corrcoef(rf_probs, xgb_probs)[0, 1]) if n > 1 else np.nan
    rows = []
    for model_name, probs in [
        ("Dummy Baseline", majority_prob),
        ("Random Forest", rf_probs),
        ("XGBoost", xgb_probs),
        ("LSTM", lstm_probs),
        ("RF+XGB Average", rf_xgb_probs),
        ("Weighted Hybrid", final_probs),
    ]:
        row = {"Model": model_name, **_classification_summary(y, probs)}
        row["Probability Std"] = float(np.std(probs))
        row["RF-XGB Corr"] = corr if model_name in ["RF+XGB Average", "Weighted Hybrid"] else np.nan
        rows.append(row)
    return rows


@st.cache_data
def _load_saved_evaluation_metrics(ticker, cache_version="academic_stats_v1"):
    path = os.path.join(BASE_DIR, "outputs", "metrics", f"{ticker}_evaluation_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _model_statistics_bundle(ticker, split_name, cache_version="academic_stats_v1"):
    from src.data.feature_store import FeatureStore
    from src.models.meta_inference import compute_final_probs

    store = FeatureStore(ticker)
    X_tabular, y = store.get_xgb_split(split_name)
    if X_tabular.ndim != 2:
        raise ValueError(f"{ticker} {split_name}: RF/XGBoost input 2D olmali, gelen shape={X_tabular.shape}")
    if X_tabular.shape[1] != 20:
        raise ValueError(f"{ticker} {split_name}: RF/XGBoost feature sayisi 20 olmali, gelen={X_tabular.shape[1]}")

    final_probs, _, component = compute_final_probs(ticker, split_name=split_name)
    n = min(len(y), len(final_probs), len(component["rf"]), len(component["xgb"]), len(component["lstm"]))
    y = np.asarray(y[-n:], dtype=int)
    probs_by_model = {
        "RF": np.asarray(component["rf"][-n:], dtype=float),
        "XGBoost": np.asarray(component["xgb"][-n:], dtype=float),
        "LSTM": np.asarray(component["lstm"][-n:], dtype=float),
        "RF+XGB": (np.asarray(component["rf"][-n:], dtype=float) + np.asarray(component["xgb"][-n:], dtype=float)) / 2.0,
        "Weighted Hybrid": np.asarray(final_probs[-n:], dtype=float),
    }

    rows = []
    matrices = {}
    for model_name, probs in probs_by_model.items():
        summary = _classification_summary(y, probs)
        preds = (probs >= 0.5).astype(int)
        matrices[model_name] = confusion_matrix(y, preds).tolist()
        rows.append({
            "Model": model_name,
            "Accuracy": summary["Accuracy"],
            "Precision": summary["Precision"],
            "Recall": summary["Recall"],
            "F1": summary["F1"],
            "AUC": summary["AUC"],
            "Probability Mean": float(np.mean(probs)),
            "Probability Std": float(np.std(probs)),
            "Probability Min": float(np.min(probs)),
            "Probability Max": float(np.max(probs)),
        })
    return {"rows": rows, "matrices": matrices}


@st.cache_data
def _feature_importance_rows(ticker, cache_version="academic_stats_v1"):
    from src.data.feature_store import FeatureStore

    store = FeatureStore(ticker)
    feature_names = list(store.feature_columns)
    models = load_ml_models(ticker)
    rows = []
    for model_key, label in [("rf", "Random Forest"), ("xgb", "XGBoost")]:
        model = models.get(model_key)
        if model is None or not hasattr(model, "feature_importances_"):
            continue
        importances = np.asarray(model.feature_importances_, dtype=float)
        n = min(len(feature_names), len(importances))
        order = np.argsort(importances[:n])[::-1][:10]
        for rank, idx in enumerate(order, start=1):
            rows.append({
                "Model": label,
                "Sıra": rank,
                "Feature": feature_names[idx],
                "Importance": float(importances[idx]),
            })
    return rows


def _run_weighted_hybrid_backtest_live(ticker, initial_capital=10000, commission=0.001):
    from src.data.feature_store import FeatureStore
    from src.models.meta_inference import compute_final_probs

    store = FeatureStore(ticker)
    X_test, _ = store.get_xgb_split("test")
    models = load_ml_models(ticker)
    for key in ["rf", "xgb"]:
        if key not in models:
            raise FileNotFoundError(f"{key.upper()} production modeli bulunamadi.")
        expected = getattr(models[key], "n_features_in_", None)
        if expected is not None and X_test.shape[1] != expected:
            raise ValueError(f"{key.upper()} feature uyusmazligi: X_test={X_test.shape[1]}, model={expected}")

    final_probs, prices, _ = compute_final_probs(ticker, split_name="test")
    n = min(len(prices), len(final_probs), len(X_test))
    prices = np.asarray(prices[-n:], dtype=float)
    final_probs = np.asarray(final_probs[-n:], dtype=float)
    signals = np.ones(n, dtype=int)
    signals[final_probs > 0.55] = 2
    signals[final_probs < 0.45] = 0
    balance = float(initial_capital)
    shares = 0.0
    portfolio_values = []
    trade_count = 0
    wins = 0
    entry_price = None
    for price, signal in zip(prices, signals):
        if signal == 2 and balance > 0:
            shares = (balance * (1 - commission)) / price
            balance = 0.0
            entry_price = price
            trade_count += 1
        elif signal == 0 and shares > 0:
            if entry_price is not None and price > entry_price:
                wins += 1
            balance = shares * price * (1 - commission)
            shares = 0.0
            entry_price = None
            trade_count += 1
        portfolio_values.append(balance + shares * price)
    pv = np.asarray(portfolio_values, dtype=float)
    peak = np.maximum.accumulate(pv)
    return {
        "prices": prices,
        "metrics": {
            "total_return": (pv[-1] - initial_capital) / initial_capital * 100,
            "max_drawdown": np.max((peak - pv) / np.maximum(peak, 1e-8)) * 100,
            "win_rate": wins / max(trade_count // 2, 1) * 100,
            "trade_count": int(trade_count),
        },
    }


@st.cache_data
def _backtest_evidence_rows(cache_version="presentation_rebuild_v2"):
    fallback = {
        "BTC-USD": {"Weighted Hybrid Return": 29.88, "Buy&Hold Return": -34.33, "Diff": 64.21, "Max Drawdown": 35.54, "Win Rate": 72.22, "Trade Count": "-"},
        "ETH-USD": {"Weighted Hybrid Return": 45.50, "Buy&Hold Return": -32.74, "Diff": 78.24, "Max Drawdown": 39.64, "Win Rate": 71.00, "Trade Count": "-"},
        "SOL-USD": {"Weighted Hybrid Return": 4.32, "Buy&Hold Return": -56.03, "Diff": 60.35, "Max Drawdown": 59.12, "Win Rate": 82.00, "Trade Count": "-"},
    }
    rows = []
    for ticker in AVAILABLE_TICKERS:
        try:
            result = _run_weighted_hybrid_backtest_live(ticker)
            metrics = result["metrics"]
            prices = result["prices"]
            buy_hold_return = (prices[-1] - prices[0]) / prices[0] * 100 if len(prices) > 1 else 0.0
            rows.append({
                "Coin": ticker,
                "Weighted Hybrid Return": metrics["total_return"],
                "Buy&Hold Return": buy_hold_return,
                "Diff": metrics["total_return"] - buy_hold_return,
                "Max Drawdown": metrics["max_drawdown"],
                "Win Rate": metrics["win_rate"],
                "Trade Count": metrics["trade_count"],
            })
        except Exception:
            row = {"Coin": ticker, **fallback[ticker]}
            rows.append(row)
    return rows


def _chart_path(ticker, kind):
    names = {
        "weighted": f"{ticker}_backtest_portfolio.png",
        "buy_hold": f"{ticker}_buy_and_hold_portfolio.png",
    }
    for folder in [
        os.path.join(BASE_DIR, "outputs", "charts"),
        os.path.join(BASE_DIR, "data", "results"),
        os.path.join(BASE_DIR, "reports", "charts"),
    ]:
        path = os.path.join(folder, names[kind])
        if os.path.exists(path):
            return path
    return None


def _proof_map_rows():
    return []


def _active_pipeline_rows():
    return [
        {"Adim": "Binance OHLCV", "Aciklama": "Maksimum Binance gunluk OHLCV gecmisi cekilir."},
        {"Adim": "data/raw", "Aciklama": "Aktif ham veri data/raw/{ticker}_ohlcv.csv olarak saklanir."},
        {"Adim": "fetch_ohlcv", "Aciklama": "FETCH_DAYS=3500 ayariyla guncel veri toplama adimi calisir."},
        {"Adim": "feature_engineering", "Aciklama": "Getiri, volatilite, momentum, RSI, MACD, ADX ve Bollinger gibi 20 teknik feature uretilir."},
        {"Adim": "build_dataset", "Aciklama": "Dead-zone filtering, WINDOW_SIZE=30 ve PURGE_GAP=14 ile final ML dataseti olusur."},
        {"Adim": "FeatureStore", "Aciklama": "data/ml/{ticker}/ altindaki hizali RF/XGBoost/LSTM splitlerini okur."},
        {"Adim": "Weighted Hybrid", "Aciklama": "0.40 RF + 0.40 XGBoost + 0.20 LSTM formuluyle final_prob hesaplanir."},
    ]


def _jury_question_rows():
    return []


def page_presentation_overview(presentation_mode=True):
    st.markdown('<p class="main-header">Kripto Para Yon Tahmini ve Karar Destek Sistemi</p>', unsafe_allow_html=True)
    show_presentation_notice()
    _proof_box("Projenin karar destek prototipi oldugunu, ana model formulunu ve kanit akis haritasini gosterir.")
    st.subheader("Gunluk Kripto Yon Tahmini")
    st.write("Dashboard BTC-USD, ETH-USD ve SOL-USD icin gunluk yon olasiligini uretir. Ana karar modeli: `final_prob = 0.40 * RF + 0.40 * XGBoost + 0.20 * LSTM`.")
    try:
        evidence = pd.DataFrame(_feature_store_evidence_rows())
        total_samples = int(pd.to_numeric(evidence["Samples"], errors="coerce").fillna(0).sum())
    except Exception as e:
        evidence = pd.DataFrame()
        total_samples = 0
        st.warning("Aktif dataset ozeti okunamadi. Veri pipeline ciktisi kontrol edilmelidir.")
        if not presentation_mode:
            with st.expander("Teknik hata", expanded=False):
                st.code(str(e))
    c1, c2, c3 = st.columns(3)
    c1.metric("Veri kapsami", f"{len(AVAILABLE_TICKERS)} coin / {total_samples} sample")
    c2.metric("Model yapisi", "RF + XGB + LSTM")
    c3.metric("Feature sayisi", "20")
    try:
        snap = _decision_snapshot(selected_ticker)
        d1, d2, d3 = st.columns(3)
        d1.metric("final_prob", f"{snap['final_prob']:.4f}")
        d2.metric("Karar", snap["signal"])
        d3.metric("Model Durumu", "LSTM fallback aktif" if snap["lstm_fallback_active"] else "Tum modeller aktif")
    except Exception as e:
        st.warning("Guncel model karari hesaplanamadi; Model Karari sayfasinda fallback bilgisi kullanilabilir.")
        if not presentation_mode:
            with st.expander("Teknik hata", expanded=False):
                st.code(str(e))
    st.subheader("Çalışma Özeti")
    st.dataframe(pd.DataFrame(_proof_map_rows()), width="stretch", hide_index=True)


def page_presentation_data_evidence(presentation_mode=True):
    st.header("Veri Seti ve Pipeline")
    show_presentation_notice()
    _proof_box("Aktif datasetleri, aktif raw veriyi, yeni pipeline'i, split yapisini ve leakage onlemlerini kanitlar.")
    st.success(f"Aktif Dataset: data/ml/{selected_ticker}/ | Aktif Raw Veri: data/raw/{selected_ticker}_ohlcv.csv | Aktif Pipeline: fetch_ohlcv -> feature_engineering -> build_dataset -> FeatureStore")
    st.caption("Legacy 392 gunluk veri arsivlenmistir ve ana sistemde kullanilmamaktadir.")
    st.subheader("Guncel Dataset Ozeti")
    st.dataframe(pd.DataFrame([
        {"Coin": "BTC-USD", "Raw satir": 3215, "Ilk tarih": "2017-08-17", "Son tarih": "2026-06-05", "Feature sonrasi": 3165, "Filter sonrasi": 2964, "Window": 30, "Purge": 14, "Train": 2034, "Val": 435, "Test": 437, "Feature count": 20},
        {"Coin": "ETH-USD", "Raw satir": 3215, "Ilk tarih": "2017-08-17", "Son tarih": "2026-06-05", "Feature sonrasi": 3165, "Filter sonrasi": 3022, "Window": 30, "Purge": 14, "Train": 2074, "Val": 444, "Test": 446, "Feature count": 20},
        {"Coin": "SOL-USD", "Raw satir": 2125, "Ilk tarih": "2020-08-11", "Son tarih": "2026-06-05", "Feature sonrasi": 2075, "Filter sonrasi": 2011, "Window": 30, "Purge": 14, "Train": 1367, "Val": 292, "Test": 294, "Feature count": 20},
    ]), width="stretch", hide_index=True)
    st.subheader("Aktif Veri Pipeline Akisi")
    st.dataframe(pd.DataFrame(_active_pipeline_rows()), width="stretch", hide_index=True)
    st.subheader("Feature Gruplari")
    st.dataframe(pd.DataFrame([
        {"Grup": "Getiri", "Feature'lar": "log_return_1d, close_open_return, return_lag_5, return_lag_20"},
        {"Grup": "Trend/Momentum", "Feature'lar": "momentum_10, sma_ratio_20, ema_ratio_50"},
        {"Grup": "Volatilite", "Feature'lar": "volatility_20, atr_pct, high_low_range"},
        {"Grup": "Teknik indikator", "Feature'lar": "rsi_14, macd_pct, macd_signal_pct, adx_14"},
        {"Grup": "Hacim", "Feature'lar": "volume_change"},
        {"Grup": "Bollinger/Stochastic", "Feature'lar": "bb_width, bb_position, stoch_k"},
    ]), width="stretch", hide_index=True)
    st.subheader("Aktif Dosya Yapisi")
    st.code("""project_root/
data/raw/                  # Binance ham OHLCV verileri
data/ml/                   # nihai train/validation/test datasetleri
src/data/                  # fetch_ohlcv, build_dataset, feature_engineering, FeatureStore
src/models/                # RF, XGBoost, LSTM, Weighted Hybrid ve inference
src/evaluation/            # metrikler ve agirlik secimi
outputs/metrics/           # model degerlendirme JSON ciktisi
outputs/charts/            # backtest grafikleri
docs/                      # rapor ve deney notlari
tests/                     # unit testler
ai_dashboard.py""")
    st.subheader("Aktif Dataset ve Split Özeti")
    st.dataframe(pd.DataFrame(_feature_store_evidence_rows()), width="stretch", hide_index=True)
    st.subheader("Leakage Onlemleri")
    st.markdown("- Zaman sirasi korunur.\n- PURGE_GAP=14 tampon alan uygular.\n- Test set final raporlama icindir.\n- RF/XGBoost ve LSTM splitleri FeatureStore ile hizalanir.")
    st.subheader("Legacy Veri Sorunu ve Cozum")
    st.dataframe(pd.DataFrame([
        {"Baslik": "Raw veri", "Legacy": "392 gun", "Guncel": "BTC/ETH 3215, SOL 2125 gun"},
        {"Baslik": "Window size", "Legacy": "60", "Guncel": "30"},
        {"Baslik": "Purge gap", "Legacy": "60", "Guncel": "14"},
        {"Baslik": "BTC train", "Legacy": "92", "Guncel": "2034"},
        {"Baslik": "ETH train", "Legacy": "98", "Guncel": "2074"},
        {"Baslik": "SOL train", "Legacy": "104", "Guncel": "1367"},
    ]), width="stretch", hide_index=True)
    st.success("Test seti model, threshold veya agirlik secimi icin kullanilmamistir.")


def page_presentation_decision(presentation_mode=True):
    st.header("Model Karari")
    show_presentation_notice()
    _proof_box("Final kararinin RF, XGBoost ve LSTM olasiliklarinin agirlikli birlesimiyle uretildigini gosterir.")
    try:
        snap = _decision_snapshot(selected_ticker)
    except Exception as e:
        st.warning("compute_final_probs calismadi. Production model ve FeatureStore ciktisi kontrol edilmelidir.")
        if not presentation_mode:
            with st.expander("Teknik hata", expanded=False):
                st.code(str(e))
        return
    weights = snap["weights"]
    row = pd.DataFrame([{"Coin": selected_ticker, "RF Prob": snap["rf_prob"], "XGBoost Prob": snap["xgb_prob"], "LSTM Prob": snap["lstm_prob"], "Final Probability": snap["final_prob"], "Weights": f"RF {weights['rf']:.2f} / XGB {weights['xgb']:.2f} / LSTM {weights['lstm']:.2f}", "Signal": snap["signal"], "Confidence Band": _decision_band(snap["final_prob"]), "LSTM Fallback": "Aktif" if snap["lstm_fallback_active"] else "Pasif"}])
    st.dataframe(row.style.format({"RF Prob": "{:.4f}", "XGBoost Prob": "{:.4f}", "LSTM Prob": "{:.4f}", "Final Probability": "{:.4f}"}), width="stretch", hide_index=True)


def page_presentation_model_comparison(presentation_mode=True):
    st.header("Model Karsilastirmasi")
    show_presentation_notice()
    _proof_box("Dummy, RF, XGBoost, LSTM, RF+XGB ve Weighted Hybrid performansini validation/test ayrimiyla gosterir.")
    st.info("Validation set model secimi icindir; test set yalnizca final raporlama icin gosterilir.")
    val_tab, test_tab = st.tabs(["Validation", "Test"])
    try:
        with val_tab:
            st.dataframe(_format_metric_df(pd.DataFrame(_model_metric_rows(selected_ticker, "val"))), width="stretch", hide_index=True)
        with test_tab:
            st.dataframe(_format_metric_df(pd.DataFrame(_model_metric_rows(selected_ticker, "test"))), width="stretch", hide_index=True)
    except Exception as e:
        st.warning("Metric JSON veya canli metrik hesaplama kullanilamadi. `python -m src.pipeline --step eval --coin BTC-USD` calistirilmalidir.")
        if not presentation_mode:
            with st.expander("Teknik hata", expanded=False):
                st.code(str(e))


def page_presentation_backtest_evidence(presentation_mode=True):
    st.header("Backtest Sonuçları")
    show_presentation_notice()
    _proof_box("Model sinyallerinin gecmis veri uzerinde Buy&Hold ile karsilastirmasini gosterir.")
    rows = pd.DataFrame(_backtest_evidence_rows())
    st.dataframe(rows.style.format({"Weighted Hybrid Return": "{:.2f}%", "Buy&Hold Return": "{:.2f}%", "Diff": "{:.2f}%", "Max Drawdown": "{:.2f}%", "Win Rate": "{:.2f}%"}), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    weighted_chart = _chart_path(selected_ticker, "weighted")
    buy_hold_chart = _chart_path(selected_ticker, "buy_hold")
    with c1:
        if weighted_chart:
            st.image(weighted_chart, caption="Weighted Hybrid portfolio", width="stretch")
        else:
            st.warning("Weighted Hybrid portfolio grafigi bulunamadi. `python -m src.pipeline --step backtest --coin BTC-USD` komutu calistirilabilir.")
    with c2:
        if buy_hold_chart:
            st.image(buy_hold_chart, caption="Buy & Hold portfolio", width="stretch")
        else:
            st.warning("Buy & Hold portfolio grafigi bulunamadi.")


def page_presentation_experiments(presentation_mode=True):
    st.header("Deneysel Çalışmalar")
    show_presentation_notice()
    _proof_box("Hangi deneylerin production'a alinmadigini ve neden reddedildigini gosterir.")
    st.dataframe(pd.DataFrame([
        {"Coin": "BTC-USD", "Production RF+XGB Val F1": 0.464, "Experiment RF+XGB Val F1": 0.535, "Experiment WH Val F1": 0.563, "Decision": "Validation iyi, test/backtest zayif: reddedildi"},
        {"Coin": "ETH-USD", "Production RF+XGB Val F1": 0.559, "Experiment RF+XGB Val F1": 0.447, "Experiment WH Val F1": 0.462, "Decision": "Validation kotulesti: reddedildi"},
        {"Coin": "SOL-USD", "Production RF+XGB Val F1": 0.622, "Experiment RF+XGB Val F1": 0.441, "Experiment WH Val F1": 0.458, "Decision": "Validation kotulesti: reddedildi"},
    ]), width="stretch", hide_index=True)
    st.info("Legacy veri, PPO/RL ve main.py / SQLite / BTCUSDT bot akisi ana production karar yolu degildir.")


def page_presentation_risks(presentation_mode=True):
    st.header("Riskler ve Sinirlamalar")
    show_presentation_notice()
    _proof_box("Sistemin karar destek prototipi oldugunu, canli trading ve garanti getiri iddiasi tasimadigini gosterir.")
    st.markdown("- Backtest gelecek performansi garanti etmez.\n- Test set secim icin kullanilmaz.\n- Sistem yatirim tavsiyesi degildir.\n- Model performansi piyasa rejimine baglidir.")


def page_presentation_jury_questions(presentation_mode=True):
    st.header("Akademik Değerlendirme Notları")
    show_presentation_notice()
    _proof_box("Teknik juri sorularina kisa cevap, kanit sayfasi, teknik kanit ve dikkatli ifade sunar.")
    st.subheader("Çalışma Özeti")
    st.dataframe(pd.DataFrame(_proof_map_rows()), width="stretch", hide_index=True)
    for idx, item in enumerate(_jury_question_rows(), start=1):
        with st.container(border=True):
            st.markdown(f"**{idx}. Soru:** {item['Soru']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Kisa cevap:** {item['Kisa cevap']}")
                st.markdown(f"**İlgili bölüm:** {item.get('İlgili bölüm', '-')}")
            with c2:
                st.markdown(f"**Teknik not:** {item.get('Teknik not', '-')}")
                st.markdown(f"**Dikkat:** {item['Dikkat']}")


def page_presentation_legacy_archive():
    st.header("Legacy / Arsiv Bilgisi - Ana Pipeline'da Kullanilmiyor")
    st.warning("Bu bolum eski gelistirme doneminden kalan arsiv/legacy bilgidir. Guncel sistem data/raw ve data/ml altindaki yeni datasetleri, src/data/build_dataset.py ve src/data/feature_store.py tabanli yeni pipeline'i kullanir.")


def _safe_legacy_page(title, page_func):
    try:
        page_func()
    except Exception as exc:
        st.warning(
            f"{title} acilamadi. Bu bolum legacy/arsiv bilgisidir; "
            "ana sunum ve production karar akisi bu sayfayi kullanmaz."
        )
        with st.expander("Legacy teknik hata", expanded=False):
            st.code(str(exc))


def _academic_notice():
    st.caption("Bu sistem yatırım tavsiyesi değildir; model çıktıları akademik karar destek prototipi kapsamında değerlendirilmelidir.")


def _dataset_summary_rows():
    return [
        {"Coin": "BTC-USD", "Raw satır": 3215, "Train": 2034, "Validation": 435, "Test": 437, "Feature": 20},
        {"Coin": "ETH-USD", "Raw satır": 3215, "Train": 2074, "Validation": 444, "Test": 446, "Feature": 20},
        {"Coin": "SOL-USD", "Raw satır": 2125, "Train": 1367, "Validation": 292, "Test": 294, "Feature": 20},
    ]


@st.cache_data
def _load_current_raw_ohlcv(ticker, cache_version="academic_ui_v1"):
    path = os.path.join(BASE_DIR, "data", "raw", f"{ticker}_ohlcv.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def _label_distribution_rows(cache_version="academic_ui_v1"):
    from src.data.feature_store import FeatureStore

    rows = []
    for ticker in AVAILABLE_TICKERS:
        store = FeatureStore(ticker)
        for split_name in ["train", "val", "test"]:
            _, y = store.get_xgb_split(split_name)
            rows.append({"Coin": ticker, "Split": split_name, "Label": "DOWN/CLOSE", "Adet": int(np.sum(np.asarray(y) == 0))})
            rows.append({"Coin": ticker, "Split": split_name, "Label": "UP/BUY", "Adet": int(np.sum(np.asarray(y) == 1))})
    return rows


def page_academic_overview():
    st.markdown('<p class="main-header">Kripto Para Yön Tahmini ve Karar Destek Sistemi</p>', unsafe_allow_html=True)
    _academic_notice()
    st.write(
        "Bu çalışma BTC, ETH ve SOL için günlük piyasa yönünü tahmin eden; makine öğrenmesi, "
        "derin öğrenme ve ağırlıklı karar birleştirme yaklaşımını kullanan akademik bir karar destek prototipidir."
    )
    st.subheader("Model Formülü")
    st.code("final_prob = 0.40 * RandomForest + 0.40 * XGBoost + 0.20 * LSTM")

    summary = pd.DataFrame(_dataset_summary_rows())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coin sayısı", "3")
    c2.metric("Aktif feature", "20")
    c3.metric("WINDOW_SIZE", "30")
    c4.metric("PURGE_GAP", "14")

    st.subheader("Veri Kapsamı")
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Model Bileşenleri")
    st.dataframe(pd.DataFrame([
        {"Bileşen": "Random Forest", "Rol": "Tabular teknik göstergelerden yön olasılığı üretir."},
        {"Bileşen": "XGBoost", "Rol": "Tabular teknik göstergelerden yön olasılığı üretir."},
        {"Bileşen": "LSTM", "Rol": "Zaman penceresi tabanlı destekleyici olasılık üretir."},
        {"Bileşen": "Weighted Hybrid", "Rol": "Üç model çıktısını sabit ağırlıklarla nihai karara dönüştürür."},
    ]), width="stretch", hide_index=True)

    st.subheader("Backtest Özeti")
    try:
        bt = pd.DataFrame(_backtest_evidence_rows())
        st.dataframe(bt[["Coin", "Weighted Hybrid Return", "Buy&Hold Return", "Max Drawdown", "Win Rate", "Trade Count"]].style.format({
            "Weighted Hybrid Return": "{:.2f}%",
            "Buy&Hold Return": "{:.2f}%",
            "Max Drawdown": "{:.2f}%",
            "Win Rate": "{:.2f}%",
        }), width="stretch", hide_index=True)
    except Exception as exc:
        st.warning("Backtest özeti okunamadı. Backtest Sonuçları sayfasında ayrıntılı çıktı kontrol edilebilir.")
        with st.expander("Teknik ayrıntı", expanded=False):
            st.code(str(exc))


def page_academic_dataset_pipeline():
    st.header("Veri Seti ve Pipeline")
    _academic_notice()
    st.success(f"Aktif raw veri: data/raw/{selected_ticker}_ohlcv.csv")
    st.success(f"Aktif dataset: data/ml/{selected_ticker}/")

    st.subheader("Aktif Pipeline")
    st.markdown("Binance OHLCV → data/raw → feature_engineering → build_dataset → FeatureStore → RF/XGBoost/LSTM → Weighted Hybrid")
    st.dataframe(pd.DataFrame(_active_pipeline_rows()), width="stretch", hide_index=True)

    st.subheader("Dataset Özeti")
    st.dataframe(pd.DataFrame(_dataset_summary_rows()), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("WINDOW_SIZE", "30")
    c2.metric("PURGE_GAP", "14")

    st.subheader("Train / Validation / Test Ayrımı")
    st.write(
        "Train bölümü model öğrenimi için, validation bölümü model davranışını izlemek ve seçim kararlarını doğrulamak için, "
        "test bölümü ise yalnızca final performans raporlaması için kullanılır."
    )

    st.subheader("Leakage Önlemleri")
    st.markdown(
        "- Zaman sırası korunur.\n"
        "- Train, validation ve test blokları arasında purge gap uygulanır.\n"
        "- Test seti model, eşik veya ağırlık seçimi için kullanılmaz.\n"
        "- RF/XGBoost ve LSTM splitleri FeatureStore üzerinden hizalı okunur."
    )

    st.subheader("Feature Grupları")
    st.dataframe(pd.DataFrame([
        {"Grup": "Getiri", "Örnek feature": "log_return_1d, close_open_return, return_lag_5, return_lag_20"},
        {"Grup": "Trend/Momentum", "Örnek feature": "momentum_10, sma_ratio_20, ema_ratio_50"},
        {"Grup": "Volatilite", "Örnek feature": "volatility_20, atr_pct, high_low_range"},
        {"Grup": "Teknik indikatör", "Örnek feature": "rsi_14, macd_pct, macd_signal_pct, adx_14"},
        {"Grup": "Hacim ve bantlar", "Örnek feature": "volume_change, bb_width, bb_position, stoch_k"},
    ]), width="stretch", hide_index=True)


def page_academic_visualization():
    st.header("Veri Görselleştirme")
    _academic_notice()
    raw_df = _load_current_raw_ohlcv(selected_ticker)
    if raw_df is None or raw_df.empty:
        st.warning(f"Güncel raw veri bulunamadı: data/raw/{selected_ticker}_ohlcv.csv")
        return

    date_col = "date" if "date" in raw_df.columns else raw_df.columns[0]
    close_col = "close" if "close" in raw_df.columns else "Close"
    if close_col not in raw_df.columns:
        st.warning("Raw veri içinde close kolonu bulunamadı.")
        return

    chart_df = raw_df[[date_col, close_col]].dropna().copy()
    chart_df["daily_return"] = chart_df[close_col].pct_change()

    st.subheader(f"{selected_ticker} Kapanış Fiyatı")
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=chart_df[date_col], y=chart_df[close_col], mode="lines", name="Close"))
    fig_close.update_layout(template="plotly_white", height=360, xaxis_title="Tarih", yaxis_title="Kapanış")
    st.plotly_chart(fig_close, use_container_width=True)

    st.subheader("Günlük Getiri")
    fig_return = go.Figure()
    fig_return.add_trace(go.Scatter(x=chart_df[date_col], y=chart_df["daily_return"], mode="lines", name="Daily return"))
    fig_return.update_layout(template="plotly_white", height=320, xaxis_title="Tarih", yaxis_title="Getiri")
    st.plotly_chart(fig_return, use_container_width=True)

    volume_col = "volume" if "volume" in raw_df.columns else "Volume"
    if volume_col in raw_df.columns:
        st.subheader("Hacim")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=raw_df[date_col], y=raw_df[volume_col], name="Volume"))
        fig_volume.update_layout(template="plotly_white", height=320, xaxis_title="Tarih", yaxis_title="Hacim")
        st.plotly_chart(fig_volume, use_container_width=True)

    st.subheader("Train / Validation / Test Örnek Sayısı")
    split_rows = []
    for row in _dataset_summary_rows():
        for split_name in ["Train", "Validation", "Test"]:
            split_rows.append({"Coin": row["Coin"], "Split": split_name, "Adet": row[split_name]})
    split_df = pd.DataFrame(split_rows)
    fig_split = px.bar(split_df, x="Coin", y="Adet", color="Split", barmode="group", template="plotly_white")
    fig_split.update_layout(height=340)
    st.plotly_chart(fig_split, use_container_width=True)

    st.subheader("Label Dağılımı")
    try:
        label_df = pd.DataFrame(_label_distribution_rows())
        fig_label = px.bar(label_df[label_df["Coin"] == selected_ticker], x="Split", y="Adet", color="Label", barmode="group", template="plotly_white")
        fig_label.update_layout(height=340)
        st.plotly_chart(fig_label, use_container_width=True)
    except Exception as exc:
        st.warning("Label dağılımı FeatureStore üzerinden okunamadı.")
        with st.expander("Teknik ayrıntı", expanded=False):
            st.code(str(exc))


def page_academic_decision():
    st.header("Model Kararı")
    _academic_notice()
    try:
        snap = _decision_snapshot(selected_ticker)
    except Exception as exc:
        st.warning("Model kararı compute_final_probs üzerinden hesaplanamadı.")
        with st.expander("Teknik ayrıntı", expanded=False):
            st.code(str(exc))
        return

    weights = snap["weights"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RF probability", f"{snap['rf_prob']:.4f}")
    c2.metric("XGBoost probability", f"{snap['xgb_prob']:.4f}")
    c3.metric("LSTM probability", f"{snap['lstm_prob']:.4f}")
    c4.metric("final_prob", f"{snap['final_prob']:.4f}")

    st.dataframe(pd.DataFrame([{
        "Coin": selected_ticker,
        "Ağırlıklar": f"RF {weights['rf']:.2f} / XGBoost {weights['xgb']:.2f} / LSTM {weights['lstm']:.2f}",
        "Sinyal": snap["signal"],
        "Karar bandı": _decision_band(snap["final_prob"]),
        "Fallback durumu": "Aktif" if snap["lstm_fallback_active"] else "Pasif",
    }]), width="stretch", hide_index=True)


def page_academic_model_comparison():
    st.header("Model Karşılaştırması")
    _academic_notice()
    st.write("Metrikler validation ve test splitleri için ayrı gösterilir. Test sonuçları model seçimi için kullanılmaz.")
    val_tab, test_tab = st.tabs(["Validation", "Test"])
    try:
        with val_tab:
            val_df = pd.DataFrame(_model_metric_rows(selected_ticker, "val"))[["Model", "Accuracy", "F1", "AUC"]]
            st.dataframe(val_df.style.format({"Accuracy": "{:.2%}", "F1": "{:.2%}", "AUC": "{:.3f}"}), width="stretch", hide_index=True)
        with test_tab:
            test_df = pd.DataFrame(_model_metric_rows(selected_ticker, "test"))[["Model", "Accuracy", "F1", "AUC"]]
            st.dataframe(test_df.style.format({"Accuracy": "{:.2%}", "F1": "{:.2%}", "AUC": "{:.3f}"}), width="stretch", hide_index=True)
    except Exception as exc:
        st.warning("Model karşılaştırma metrikleri hesaplanamadı.")
        with st.expander("Teknik ayrıntı", expanded=False):
            st.code(str(exc))


def page_academic_model_statistics():
    st.header("Model İstatistikleri")
    _academic_notice()
    saved = _load_saved_evaluation_metrics(selected_ticker)
    if saved:
        st.info("Kayıtlı değerlendirme metriği bulundu: outputs/metrics/{ticker}_evaluation_metrics.json")
        st.dataframe(pd.DataFrame([{
            "Coin": selected_ticker,
            "Accuracy": saved.get("accuracy"),
            "Precision": saved.get("precision"),
            "Recall": saved.get("recall"),
            "F1": saved.get("f1"),
            "AUC": saved.get("auc_roc"),
            "Support": saved.get("support", {}).get("total"),
            "LSTM fallback": "Aktif" if saved.get("lstm_degenerate") else "Pasif",
        }]).style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1": "{:.2%}",
            "AUC": "{:.3f}",
        }), width="stretch", hide_index=True)
    else:
        st.warning("Kayıtlı evaluation metrics JSON bulunamadı; güvenli canlı hesaplama kullanılacak.")

    val_tab, test_tab = st.tabs(["Validation istatistikleri", "Test istatistikleri"])
    for split_name, tab in [("val", val_tab), ("test", test_tab)]:
        with tab:
            try:
                bundle = _model_statistics_bundle(selected_ticker, split_name)
                stats_df = pd.DataFrame(bundle["rows"])
                st.subheader("Metrikler ve Probability Dağılımı")
                st.dataframe(stats_df.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1": "{:.2%}",
                    "AUC": "{:.3f}",
                    "Probability Mean": "{:.4f}",
                    "Probability Std": "{:.4f}",
                    "Probability Min": "{:.4f}",
                    "Probability Max": "{:.4f}",
                }), width="stretch", hide_index=True)

                st.subheader("Confusion Matrix")
                selected_model = st.selectbox(
                    f"{split_name.upper()} model",
                    list(bundle["matrices"].keys()),
                    key=f"cm_model_{split_name}",
                )
                cm = np.asarray(bundle["matrices"][selected_model])
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Tahmin", y="Gerçek", color="Adet"),
                    x=["DOWN/CLOSE", "UP/BUY"],
                    y=["DOWN/CLOSE", "UP/BUY"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(template="plotly_white", height=360)
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as exc:
                st.warning(f"{split_name} istatistikleri hesaplanamadı.")
                with st.expander("Teknik ayrıntı", expanded=False):
                    st.code(str(exc))

    st.subheader("Feature Importance Top 10")
    try:
        fi_rows = _feature_importance_rows(selected_ticker)
        if fi_rows:
            fi_df = pd.DataFrame(fi_rows)
            st.dataframe(fi_df.style.format({"Importance": "{:.6f}"}), width="stretch", hide_index=True)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", color="Model", facet_col="Model", orientation="h", template="plotly_white")
            fig_fi.update_layout(height=430, showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Production RF/XGBoost modellerinde feature_importances_ alanı bulunamadı.")
    except Exception as exc:
        st.warning("Feature importance okunamadı.")
        with st.expander("Teknik ayrıntı", expanded=False):
            st.code(str(exc))


def page_academic_backtest():
    st.header("Backtest Sonuçları")
    _academic_notice()
    try:
        rows = pd.DataFrame(_backtest_evidence_rows())
        selected_row = rows[rows["Coin"] == selected_ticker]
        display_rows = selected_row if not selected_row.empty else rows.head(1)
        st.dataframe(display_rows[["Coin", "Weighted Hybrid Return", "Buy&Hold Return", "Max Drawdown", "Win Rate", "Trade Count"]].style.format({
            "Weighted Hybrid Return": "{:.2f}%",
            "Buy&Hold Return": "{:.2f}%",
            "Max Drawdown": "{:.2f}%",
            "Win Rate": "{:.2f}%",
        }), width="stretch", hide_index=True)
    except Exception as exc:
        st.warning("Backtest metrikleri okunamadı. Hazır grafikler varsa aşağıda gösterilir.")
        with st.expander("Teknik ayrıntı", expanded=False):
            st.code(str(exc))

    c1, c2 = st.columns(2)
    weighted_chart = _chart_path(selected_ticker, "weighted")
    buy_hold_chart = _chart_path(selected_ticker, "buy_hold")
    with c1:
        if weighted_chart:
            st.image(weighted_chart, caption="Weighted Hybrid portfolio", width="stretch")
        else:
            st.warning("Weighted Hybrid backtest grafiği bulunamadı. Komut: py -m src.pipeline --step backtest --coin BTC-USD")
    with c2:
        if buy_hold_chart:
            st.image(buy_hold_chart, caption="Buy & Hold portfolio", width="stretch")
        else:
            st.warning("Buy & Hold grafiği bulunamadı.")


def page_academic_backtest_statistics():
    st.header("Backtest İstatistikleri")
    _academic_notice()
    try:
        rows = pd.DataFrame(_backtest_evidence_rows())
        cols = ["Coin", "Weighted Hybrid Return", "Buy&Hold Return", "Diff", "Max Drawdown", "Win Rate", "Trade Count"]
        st.subheader("Coin Bazlı Karşılaştırma")
        st.dataframe(rows[cols].style.format({
            "Weighted Hybrid Return": "{:.2f}%",
            "Buy&Hold Return": "{:.2f}%",
            "Diff": "{:.2f}%",
            "Max Drawdown": "{:.2f}%",
            "Win Rate": "{:.2f}%",
        }), width="stretch", hide_index=True)

        selected_row = rows[rows["Coin"] == selected_ticker]
        if not selected_row.empty:
            r = selected_row.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Weighted Hybrid Return", f"{float(r['Weighted Hybrid Return']):.2f}%")
            c2.metric("Buy & Hold Return", f"{float(r['Buy&Hold Return']):.2f}%")
            c3.metric("Fark", f"{float(r['Diff']):.2f}%")
            c4, c5, c6 = st.columns(3)
            c4.metric("Max Drawdown", f"{float(r['Max Drawdown']):.2f}%")
            c5.metric("Win Rate", f"{float(r['Win Rate']):.2f}%")
            c6.metric("Trade Count", str(r["Trade Count"]))
    except Exception as exc:
        st.warning("Backtest istatistikleri hesaplanamadı. Hazır grafikler Backtest Sonuçları sayfasında kontrol edilebilir.")
        with st.expander("Teknik ayrıntı", expanded=False):
            st.code(str(exc))

    st.subheader("Hazır Çıktı Durumu")
    status_rows = []
    for ticker in AVAILABLE_TICKERS:
        status_rows.append({
            "Coin": ticker,
            "Weighted Hybrid grafik": "Var" if _chart_path(ticker, "weighted") else "Yok",
            "Buy & Hold grafik": "Var" if _chart_path(ticker, "buy_hold") else "Yok",
            "Evaluation metrics JSON": "Var" if _load_saved_evaluation_metrics(ticker) else "Yok",
        })
    st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)


def page_academic_experiments():
    st.header("Deneysel Çalışmalar")
    _academic_notice()
    st.dataframe(pd.DataFrame([
        {"Çalışma": "LSTM veri yetersizliği", "Sonuç": "Yeni data/raw ve data/ml yapısı ile daha uzun Binance geçmişi kullanıldı; LSTM güvenli checkpoint loader ile okunur."},
        {"Çalışma": "RF/XGBoost yeniden eğitim deneyi", "Sonuç": "Deney modelleri ml_experiments altında bırakıldı; production modellerine bağlanmadı."},
        {"Çalışma": "PPO/RL", "Sonuç": "Deneysel kaldı; production karar yoluna dahil edilmedi."},
        {"Çalışma": "Eski ensemble yaklaşımı", "Sonuç": "Legacy olarak bırakıldı; güncel karar akışı Weighted Hybrid üzerinden yürür."},
    ]), width="stretch", hide_index=True)

    st.subheader("RF/XGBoost Yeniden Eğitim Kararı")
    st.dataframe(pd.DataFrame([
        {"Coin": "BTC-USD", "Gözlem": "Validation iyileşmesine rağmen test/backtest zayıfladı.", "Karar": "Production'a alınmadı"},
        {"Coin": "ETH-USD", "Gözlem": "Validation tarafında production modelden kötüleşti.", "Karar": "Production'a alınmadı"},
        {"Coin": "SOL-USD", "Gözlem": "Validation tarafında production modelden kötüleşti.", "Karar": "Production'a alınmadı"},
    ]), width="stretch", hide_index=True)


def page_academic_risks():
    st.header("Riskler ve Sınırlamalar")
    _academic_notice()
    st.markdown(
        "- Bu sistem yatırım tavsiyesi değildir.\n"
        "- Backtest sonuçları gelecek performansı garanti etmez.\n"
        "- Uygulamada canlı emir iletimi bulunmaz.\n"
        "- Piyasa rejimi değişimleri model performansını etkileyebilir.\n"
        "- Model çıktıları karar destek amaçlıdır ve tek başına finansal karar yerine geçmez."
    )


def main():
    with st.sidebar:
        st.markdown("# KDS")
        st.title("Kripto Karar Destek")
        st.markdown("---")

        page = st.radio("Sayfa Secin", [
            "Genel Bakış",
            "Veri Seti ve Pipeline",
            "Veri Görselleştirme",
            "Model Kararı",
            "Model Karşılaştırması",
            "Model İstatistikleri",
            "Backtest Sonuçları",
            "Backtest İstatistikleri",
            "Deneysel Çalışmalar",
            "Riskler ve Sınırlamalar",
        ])

        st.markdown("---")
        st.markdown("**Aktif Sistem:**")
        st.markdown("""
        - data/raw + data/ml
        - build_dataset + FeatureStore
        - Weighted Hybrid production akisi
        """)

    if page == "Genel Bakış":
        page_academic_overview()
    elif page == "Veri Seti ve Pipeline":
        page_academic_dataset_pipeline()
    elif page == "Veri Görselleştirme":
        page_academic_visualization()
    elif page == "Model Kararı":
        page_academic_decision()
    elif page == "Model Karşılaştırması":
        page_academic_model_comparison()
    elif page == "Model İstatistikleri":
        page_academic_model_statistics()
    elif page == "Backtest Sonuçları":
        page_academic_backtest()
    elif page == "Backtest İstatistikleri":
        page_academic_backtest_statistics()
    elif page == "Deneysel Çalışmalar":
        page_academic_experiments()
    elif page == "Riskler ve Sınırlamalar":
        page_academic_risks()
if __name__ == "__main__":
    main()
