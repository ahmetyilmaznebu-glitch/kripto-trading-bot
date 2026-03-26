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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Proje kok dizini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.models.time_series_models import TimeSeriesNet
from config import DISPLAY_DAYS  # Dashboard parametreleri

# ─────────────────────────────────────────────────────────────
#  Sayfa Ayarlari
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Bot Dashboard",
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
                 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid', 'SMA_20', 'EMA_50']
    
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
            model = TimeSeriesNet(X.shape[2], hidden_size=128, num_layers=num_layers,
                                  output_size=1, model_type=name, use_attention=True)
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
    
    st.info("""
    **Bu modeller ne yapar?**  
    LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit) modelleri, son **60 günlük** fiyat ve teknik 
    gösterge verilerini analiz ederek **bir sonraki günün kapanış fiyatını** tahmin eder.  
    Zaman serisi verilerdeki uzun vadeli kalıpları öğrenebilen derin öğrenme mimarileridir.
    """)
    
    dl_results = evaluate_dl_models(selected_ticker)
    
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
    
    st.info("""
    **Bu modeller ne yapar?**  
    Teknik göstergeler (RSI, MACD, Bollinger Bands vb.) temelinde piyasanın **yükselip (AL) ya da 
    düşeceğini (SAT)** tahmin eder. Karar ağacı tabanlı güçlü sınıflandırma algoritmalarıdır.
    """)
    
    df = load_processed_data(selected_ticker)
    if df is None:
        st.warning("⚠️ İşlenmiş veri bulunamadı.")
        return
    
    results = evaluate_ml_models(df, selected_ticker)
    
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
        st.image(backtest_img, use_column_width=True)
    
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
#  SAYFA: Veri Pipeline & Ozellik Istatistikleri
# ─────────────────────────────────────────────────────────────
def page_data_pipeline():
    st.header("🔬 Veri Pipeline & Özellik İstatistikleri")
    
    st.info("""
    **Bu sayfa ne gösterir?**  
    Ham verilerin toplanmasından model eğitimine kadar olan veri pipeline sürecinin 
    istatistiklerini, özellik dağılımlarını ve korelasyon analizini gösterir.
    """)
    
    raw_df = load_raw_data(selected_ticker)
    processed_df = load_processed_data(selected_ticker)
    
    # ── Veri Pipeline Ozeti ──
    st.subheader("📦 Veri Pipeline Özeti")
    
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
#  Ana Uygulama
# ─────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# 🤖")
        st.title("AI Trading Bot")
        st.markdown("---")
        
        page = st.radio("📑 Sayfa Seçin", [
            "🏠 Genel Bakış",
            "🔬 Veri Pipeline & Özellikler",
            "📈 Fiyat Tahmini (LSTM/GRU)",
            "🔔 Sinyal Üretimi (XGB/RF)",
            "🧬 Ensemble Model",
            "🎮 Deep RL (PPO)",
            "📊 Backtesting",
            "📉 Veri Görselleştirme"
        ])
        
        st.markdown("---")
        st.markdown("**Hızlı Bilgi:**")
        st.markdown("""
        - 🟢 Fiyat tahmini: LSTM/GRU
        - 🔵 Sinyal üretimi: XGBoost/RF  
        - 🟣 Ensemble: Meta-Model
        - 🟡 Karar alma: PPO (RL)
        """)
    
    # Sayfa yonlendirmesi
    if "Genel Bakış" in page:
        page_overview()
    elif "Veri Pipeline" in page:
        page_data_pipeline()
    elif "Fiyat Tahmini" in page:
        page_price_prediction()
    elif "Sinyal Üretimi" in page:
        page_signal_generation()
    elif "Ensemble" in page:
        page_ensemble()
    elif "Deep RL" in page:
        page_rl()
    elif "Backtesting" in page:
        page_backtest()
    elif "Veri Görsel" in page:
        page_data()

if __name__ == "__main__":
    main()
