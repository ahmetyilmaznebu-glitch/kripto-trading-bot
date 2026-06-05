"""
train_agent.py — Deneysel PPO/RL Ajan Egitimi
================================================================
⚠️  DENEYSEL (EXPERIMENTAL) — Ana pipeline'in parcasi DEGILDIR.

Bu modul pekistirmeli ogrenme (Reinforcement Learning) ile trading
ajan egitimi icin tasarlanmistir. Ana karar mekanizmasi olarak
Weighted Hybrid modeli (src/models/weighted_hybrid.py) kullanilir.

RL modulu, arastirma / deney amaclidir ve ana akis (src/pipeline.py)
tarafindan cagrilmaz.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import deque

# Proje kokunu sys.path'e ekle (subprocess olarak calistiginda src.* importlari icin)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.meta_inference import compute_final_probs

# Geriye uyumluluk: eski WINDOW_SIZE sabiti
WINDOW_SIZE = 60

class TradingEnv(gym.Env):
    """
    Kripto/Hisse Senedi Alim Satimi Icin Ozel RL Ortami
    
    Iyilestirmeler:
    - Genisletilmis observation space (4 -> 10 ozellik)
    - Risk-adjusted reward fonksiyonu
    - Drawdown cezasi ve asiri trading cezasi
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, meta_signal_series, initial_balance=10000,
                 transaction_fee=0.001, window_size=60):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.meta_signal_series = meta_signal_series
        # Eylem Uzayı: 0=(Sat/Nakit), 1=(Tut/Bekle), 2=(Al/Varlik)
        self.action_space = spaces.Discrete(3)
        
        # Genisletilmis Gozlem Uzayi (10 ozellik):
        # [bakiye, net_worth, meta_prob, fiyat, rsi, macd_signal, 
        #  price_change_5d, volatility_20d, volume_ratio, position_ratio]
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        
        self.initial_balance = initial_balance
        self.fee = transaction_fee
        self.window_size = window_size
        
        # DataFrame verilerini hazirla
        self.closes = df['Close'].values

        # Teknik gostergeler icin gerekli veriler
        self.rsi_values = df['RSI'].values if 'RSI' in df.columns else np.full(len(df), 50.0)
        self.macd_signal_values = df['MACD_Signal'].values if 'MACD_Signal' in df.columns else np.full(len(df), 0.0)
        self.volume_values = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))

        # LOOK-AHEAD BIAS DUZELTME: Normalizasyon parametreleri sadece ilk
        # 'warmup_size' gozlemden hesaplanir (gelecek veri kullanilmaz).
        # warmup_size = 1 yil (252 islem gunu) veya verinin %20'si, hangisi kucukse.
        warmup_size = min(252, max(window_size + 1, len(self.closes) // 5))
        warmup_closes = self.closes[:warmup_size]
        warmup_macd = self.macd_signal_values[:warmup_size]
        warmup_volume = self.volume_values[:warmup_size]

        # Fiyat normalizasyonu: sadece warmup penceresi min/max
        self.price_min = float(np.min(warmup_closes))
        self.price_max = float(np.max(warmup_closes))
        if self.price_max == self.price_min:
            self.price_max = self.price_min + 1.0

        # RSI: tanim geregi 0-100 araliginda, veri gerektirmez
        self.rsi_min = 0.0
        self.rsi_max = 100.0

        # MACD normalizasyonu: sadece warmup penceresi
        self.macd_min = float(np.nanmin(warmup_macd)) if not np.all(np.isnan(warmup_macd)) else -1.0
        self.macd_max = float(np.nanmax(warmup_macd)) if not np.all(np.isnan(warmup_macd)) else 1.0
        if self.macd_max == self.macd_min:
            self.macd_max = self.macd_min + 1.0

        # Volume normalizasyonu: sadece warmup penceresi ortalamasi
        warmup_vol_mean = float(np.nanmean(warmup_volume))
        self.vol_mean = warmup_vol_mean if warmup_vol_mean > 0 else 1.0
        
        self.current_step = window_size
        self.max_steps = len(df) - 1
        
        # Baslangic Degerleri
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        # Reward hesaplamasi icin gecmis getiriler
        self.recent_returns = deque(maxlen=20)
        self.prev_action = 1  # Baslangicta "tut"
        
        # Aksiyon cesitliligi takibi
        self.action_counts = {0: 0, 1: 0, 2: 0}  # SAT, TUT, AL
        self.position_duration = 0  # Ayni pozisyonda kalma suresi
        self.total_steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = self.window_size
        self.recent_returns = deque(maxlen=20)
        self.prev_action = 1
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.position_duration = 0
        self.total_steps_taken = 0
        
        return self._next_observation(), {}

    def _normalize(self, value, vmin, vmax):
        """Degeri 0-1 arasina normalize et."""
        if vmax == vmin:
            return 0.5
        result = (value - vmin) / (vmax - vmin)
        return float(np.clip(result, 0.0, 1.0))

    def _next_observation(self):
        # Guvenli indeks kontrolu
        idx = min(self.current_step, len(self.closes) - 1)
        
        # 1. Meta-model olasiligi
        if idx < len(self.meta_signal_series):
            raw_prob = float(self.meta_signal_series[idx])
        else:
            raw_prob = 0.5
        if not np.isfinite(raw_prob):
            raw_prob = 0.5
        meta_signal_prob = float(np.clip(raw_prob, 0.0, 1.0))
        
        # 2. Fiyat normalizasyonu
        current_price = self.closes[idx]
        normalized_price = self._normalize(current_price, self.price_min, self.price_max)
        
        # DUZELTME (Hata #8): RSI normalizasyonu verinin gercek min/max'ini kullanir
        # Orijinal fiyat verisi kullanildiginda RSI 0-100 arasi olur
        # Olceklenmis veri kullanilsaydi 0-1 arasi olurdu — her iki durumda da dogru calisir
        rsi_val = self.rsi_values[idx] if idx < len(self.rsi_values) else 50.0
        rsi_norm = self._normalize(rsi_val, self.rsi_min, self.rsi_max)
        
        # 4. MACD Signal normalizasyonu
        macd_val = self.macd_signal_values[idx] if idx < len(self.macd_signal_values) else 0.0
        macd_norm = self._normalize(macd_val, self.macd_min, self.macd_max)
        
        # 5. Son 5 gunluk fiyat degisimi
        if idx >= 5:
            price_change_5d = (self.closes[idx] - self.closes[idx - 5]) / (self.closes[idx - 5] + 1e-8)
            price_change_5d = float(np.clip(price_change_5d, -0.5, 0.5)) + 0.5  # -0.5..0.5 -> 0..1
        else:
            price_change_5d = 0.5
        
        # 6. 20 gunluk volatilite
        if idx >= 20:
            returns_20d = np.diff(self.closes[idx-20:idx+1]) / (self.closes[idx-20:idx] + 1e-8)
            volatility = float(np.std(returns_20d))
            volatility = min(volatility / 0.1, 1.0)  # %10 volatilite = 1.0
        else:
            volatility = 0.5
        
        # 7. Hacim orani (guncel / ortalama)
        vol_now = self.volume_values[idx] if idx < len(self.volume_values) else self.vol_mean
        volume_ratio = float(np.clip(vol_now / (self.vol_mean + 1e-8), 0, 3.0)) / 3.0
        
        # 8. Pozisyon orani (portfoydeki varlik orani)
        total_value = self.balance + self.shares_held * current_price
        position_ratio = (self.shares_held * current_price) / (total_value + 1e-8) if total_value > 0 else 0.0
        
        obs = np.array([
            self.balance / (self.initial_balance * 2),     # Normalize bakiye
            self.net_worth / (self.initial_balance * 2),   # Normalize net worth
            meta_signal_prob,                               # Meta-model olasiligi
            normalized_price,                               # Normalize fiyat
            rsi_norm,                                       # RSI (0-1)
            macd_norm,                                      # MACD Signal (0-1)
            price_change_5d,                                # 5 gunluk fiyat degisimi (0-1)
            volatility,                                     # 20 gunluk volatilite (0-1)
            volume_ratio,                                   # Hacim orani (0-1)
            position_ratio,                                 # Pozisyon orani (0-1)
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.5, posinf=1.0, neginf=0.0)
        return np.clip(obs, 0, 1)

    def _compute_reward(self, action, prev_net_worth):
        """
        Risk-adjusted reward fonksiyonu (v2 — log-return bazli).
        
        Degisiklikler:
        - PnL yerine log-return kullanilir (bull/bear bias azaltir)
        - Entropy-bazli cesitlilik cezasi (buy-only davranisi onler)
        """
        # 1. Log-return bazli reward (bull/bear fark etmez)
        if prev_net_worth > 0 and self.net_worth > 0:
            log_ret = np.log(self.net_worth / prev_net_worth)
            pnl_reward = log_ret * 100  # olcekleme
        else:
            pnl_reward = 0.0
        
        # 2. Drawdown cezasi
        if self.max_net_worth > 0:
            drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            dd_penalty = -drawdown * 2.0
        else:
            dd_penalty = 0
        
        # 3. Asiri trading cezasi (hold disindaki her islem kucuk ceza)
        trade_penalty = -0.001 if action != 1 else 0.0
        
        # 4. Sharpe-benzeri normalizasyon
        if len(self.recent_returns) > 5:
            returns_std = np.std(list(self.recent_returns)) + 1e-8
            sharpe_bonus = pnl_reward / returns_std * 0.1
        else:
            sharpe_bonus = 0
        
        total_reward = pnl_reward + dd_penalty + trade_penalty + sharpe_bonus
        
        # 5. Entropy-bazli aksiyon cesitliligi cezasi
        # Monoton davranis (sadece AL veya sadece TUT) cezalandirilir
        if self.total_steps_taken > 20:
            recent_n = min(self.total_steps_taken, 50)
            counts = np.array([self.action_counts.get(i, 0) for i in range(3)], dtype=float)
            counts = counts / (counts.sum() + 1e-8)
            # Entropy hesapla: yuksek entropy = cesitli aksiyonlar
            entropy = -sum(p * np.log(p + 1e-9) for p in counts)
            max_entropy = np.log(3)
            # Dusuk entropy = monoton davranis = buyuk ceza
            diversity_penalty = -0.3 * (1.0 - entropy / max_entropy)
            total_reward += diversity_penalty
        
        # 6. Cok uzun suredir ayni pozisyonda kalma cezasi
        if self.position_duration > 50:
            total_reward -= 0.02 * (self.position_duration - 50) / 50
        
        return float(np.clip(total_reward, -10, 10))

    def step(self, action):
        current_price = self.closes[self.current_step]
        prev_net_worth = self.net_worth
        
        # Fiyat validity kontrolu
        if current_price <= 0 or not np.isfinite(current_price):
            self.current_step += 1
            done = self.net_worth <= 0 or self.current_step >= self.max_steps
            reward = -1.0
            return self._next_observation(), reward, done, False, {
                'step': self.current_step,
                'net_worth': self.net_worth,
                'action': action,
                'price': current_price
            }
        
        # Aksiyonlari Uygula
        if action == 2 and self.balance > 0:  # Alim Yap
            shares_bought = (self.balance * (1 - self.fee)) / current_price
            if np.isfinite(shares_bought) and shares_bought > 0:
                self.shares_held += shares_bought
                self.balance = 0
            
        elif action == 0 and self.shares_held > 0:  # Satis Yap
            sale_amount = (self.shares_held * current_price) * (1 - self.fee)
            if np.isfinite(sale_amount) and sale_amount > 0:
                self.balance += sale_amount
                self.shares_held = 0
            
        elif action == 1:  # Tut veya Bekle
            pass
            
        # Portfoy Degerini Guncelle
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        if not np.isfinite(self.net_worth):
            self.net_worth = prev_net_worth
        
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        
        # Getiri kaydet
        ret = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-8)
        self.recent_returns.append(ret)
        
        # Risk-adjusted reward
        reward = self._compute_reward(action, prev_net_worth)

        # Pozisyon suresi takibi (prev_action guncellenmeden once karsilastir)
        if action == self.prev_action:
            self.position_duration += 1
        else:
            self.position_duration = 0

        self.prev_action = action
        self.current_step += 1
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.total_steps_taken += 1
        
        done = self.net_worth <= 0 or self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'step': self.current_step,
            'net_worth': self.net_worth,
            'action': action,
            'price': current_price,
            'drawdown': (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-8)
        }
        
        return self._next_observation(), reward, done, truncated, info

def train_rl_agent(ticker="BTC-USD"):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(f"\n--- {ticker} Weighted Hybrid olasiliklari hesaplaniyor (RL icin) ---")
    # compute_final_probs: Weighted Hybrid formulu ile final olasilik
    meta_probs, _closes, _component_info = compute_final_probs(ticker, split_name="test")

    # Orijinal fiyat verilerini yukle (RL ortami icin)
    import pandas as pd
    raw_path = os.path.join(base_dir, "data", "raw", f"{ticker}_ohlcv.csv")
    alt_path = os.path.join(base_dir, "data", "raw", f"{ticker}_raw.csv")
    if os.path.exists(raw_path):
        df_original = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    elif os.path.exists(alt_path):
        df_original = pd.read_csv(alt_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(f"Ham veri bulunamadi: {raw_path}")

    # Orijinal (olceklenmemis) fiyatlari kullan — gercekci simulasyon icin
    df_for_env = df_original
    closes = df_for_env['Close'].values
    num_windows = len(meta_probs)

    # DATA LEAKAGE ONLEMI: RL ajani sadece egitim bolumunde (%80) egitilmeli.
    # Test bolumu (%20) backtester tarafindan degerlendirme icin ayrilmistir.
    train_size = int(num_windows * 0.8)
    train_end_idx = train_size + (WINDOW_SIZE - 1)  # pencere ofsetini hesaba kat
    df_train = df_for_env.iloc[:train_end_idx].reset_index(drop=True)

    # Pencerelerin bitis gunlerine eslenen olasilik serisi (sadece egitim kismi)
    meta_signal_series = np.full(len(df_train), 0.5, dtype=float)
    for i in range(train_size):
        idx = i + (WINDOW_SIZE - 1)
        if idx < len(df_train):
            meta_signal_series[idx] = meta_probs[i]

    print(f"\n--- {ticker} Ortam(Environment) Olusturuluyor (Egitim verisi: {len(df_train)} gun, Test ayrildi) ---")
    env = DummyVecEnv([lambda: TradingEnv(df_train, meta_signal_series, window_size=WINDOW_SIZE)])
    
    print(f"\n--- {ticker} PPO Ajan Egitimi Basliyor (500K timestep) ---")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, 
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.15,  # Kesfetme tesvik bonusu: 0.15 ile cesitli aksiyonlari zorlar (0.10->0.15)
    )
    model.learn(total_timesteps=500000)
    
    rl_models_dir = os.path.join(base_dir, 'src', 'rl', 'saved_agents')
    os.makedirs(rl_models_dir, exist_ok=True)
    
    model.save(os.path.join(rl_models_dir, f'{ticker}_ppo_trading_agent'))
    print(f"✅ {ticker} RL Ajan başarıyla kaydedildi.")

if __name__ == "__main__":
    train_rl_agent()

