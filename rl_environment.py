# ============================================================
#  rl_environment.py  —  Pekiştirmeli Öğrenme Ortamı
#  Gymnasium tabanlı özel Kripto Para Alım/Satım Ortamı
#
#  [DEPRECATED] Bu dosya 1-5. hafta prototip mimarisine aittir.
#  Action space: {0:Hold, 1:Buy, 2:Sell} | Basit PnL reward
#
#  Aktif mimari için: src/rl/train_agent.py
#  (Action space: {0:Sat, 1:Tut, 2:Al} | Meta-signal entegrasyonu |
#   Risk-adjusted reward | PPO eğitimi)
# ============================================================

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

logger = logging.getLogger(__name__)

class CryptoTradingEnv(gym.Env if gym else object):
    """
    Kripto para ticareti için Gymnasium çevresi.
    
    Aksiyon Uzayı (Ayrık - Discrete):
        0: Hold (Bekle)
        1: Buy (Al)
        2: Sell (Sat)
        
    Gözlem Uzayı (Sürekli - Box):
        Fiyatlar, hacim, teknik göstergeler, cüzdan durumu ve güncel tahminler.
    """
    
    metadata = {'render_modes': ['human', 'system']}
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, transaction_fee_percent: float = 0.001):
        super(CryptoTradingEnv, self).__init__()
        
        if gym is None:
            logger.error("⚠️ gymnasium kurulu değil! Lütfen 'pip install gymnasium' çalıştırın.")
            return

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = transaction_fee_percent
        
        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: Cüzdan bilgisi (2 özellik) + df'teki numerik sütunlar
        # Sürekli değerler için -inf, +inf
        self.n_features = len(self.df.select_dtypes(include=[np.number]).columns) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )
        
        # Ortam Değişkenleri
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Cüzdan Değişkenleri
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        self.history = []

    def _get_observation(self) -> np.ndarray:
        """Geçerli adımdaki durumu (state) döndürür."""
        obs = self.df.select_dtypes(include=[np.number]).iloc[self.current_step].values
        
        # Cüzdan özellikleri de gözleme eklenir
        wallet_features = np.array([
            self.balance / self.initial_balance,
            self.crypto_held
        ])
        
        return np.concatenate((obs, wallet_features)).astype(np.float32)

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Ortamı başlangıç durumuna sıfırlar."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        self.history = []
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_info(self) -> Dict[str, Any]:
        """Ek bilgi döndürür."""
        return {
            'step': self.current_step,
            'net_worth': self.net_worth,
            'balance': self.balance,
            'crypto_held': self.crypto_held
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Belirtilen aksiyonu uygular ve yeni durumu döndürür.
        """
        current_price = self.df['close'].iloc[self.current_step]
        
        prev_net_worth = self.net_worth
        
        # İşlem mantığı
        if action == 1: # BUY
            # Tüm parayla alım yapalım (Basit strateji)
            if self.balance > 0:
                coins_bought = (self.balance * (1 - self.fee)) / current_price
                self.crypto_held += coins_bought
                self.balance = 0.0
                logger.debug(f"Step {self.current_step}: BUY at {current_price}")
                
        elif action == 2: # SELL
            # Tüm coinleri satalım
            if self.crypto_held > 0:
                self.balance += (self.crypto_held * current_price) * (1 - self.fee)
                self.crypto_held = 0.0
                logger.debug(f"Step {self.current_step}: SELL at {current_price}")
        
        # HOLD (action == 0) durumu
        
        # Net değer güncellemesi
        self.net_worth = self.balance + (self.crypto_held * current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Ödül (Reward) Mekanizması
        # Yüzdesel değişim: farklı fiyat seviyelerinde karşılaştırılabilir reward üretir
        reward = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-8)
        
        # Bitiş kontrolü
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False # İsteğe bağlı zaman sınırı konabilir
        
        if terminated:
            logger.info(f"Simülasyon Bitti. İlk Bakiye: {self.initial_balance:.2f}, Son Net Değer: {self.net_worth:.2f}")

        # Geçmiş kaydı (Render / Analiz için)
        self.history.append({
            'step': self.current_step,
            'net_worth': self.net_worth,
            'reward': reward,
            'action': action,
            'price': current_price
        })

        obs = self._get_observation()
        info = self._get_info()
        
        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Çıktı grafiği veya anlık log."""
        mode = self.metadata['render_modes'][1]
        if mode == 'system':
            print(f"Adım: {self.current_step}, Fiyat: {self.df['close'].iloc[self.current_step]}, Net Değer: {self.net_worth:.2f}")

