import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')  # GUI olmayan ortamlarda grafik kaydetmek icin
import matplotlib.pyplot as plt

# Proje kokunu sys.path'e ekle (subprocess olarak calistiginda src.* importlari icin)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.meta_inference import compute_meta_probs, WINDOW_SIZE

class SimpleBacktester:
    """
    Modellerin urettigi sinyallerle (AL/SAT/TUT) gecmis veride 
    simule edilmis ticaret yapar ve performans metriklerini hesaplar.
    """
    
    def __init__(self, prices, signals, initial_capital=10000, commission=0.001):
        """
        Args:
            prices (np.array): Kapanis fiyatlarinin zaman serisi
            signals (np.array): Her adim icin sinyal (0=SAT, 1=TUT, 2=AL)
            initial_capital (float): Baslangic sermayesi
            commission (float): Islem basina komisyon orani
        """
        self.prices = prices
        self.signals = signals
        self.initial_capital = initial_capital
        self.commission = commission
        
        # Sonuclar
        self.portfolio_values = []
        self.trades = []
        
    def run(self):
        # DUZELTME (Hata #10): Tekrar cagrildiginda birikmesini onle
        self.portfolio_values = []
        self.trades = []
        
        balance = self.initial_capital
        shares = 0
        
        for i in range(len(self.prices)):
            price = self.prices[i]
            signal = self.signals[i]
            
            if signal == 2 and balance > 0:  # AL
                shares_bought = (balance * (1 - self.commission)) / price
                shares += shares_bought
                self.trades.append({'step': i, 'action': 'BUY', 'price': price, 'shares': shares_bought})
                balance = 0
                
            elif signal == 0 and shares > 0:  # SAT
                revenue = shares * price * (1 - self.commission)
                self.trades.append({'step': i, 'action': 'SELL', 'price': price, 'shares': shares})
                balance += revenue
                shares = 0
                
            # TUT ise islem yok
            
            portfolio_value = balance + (shares * price)
            self.portfolio_values.append(portfolio_value)
            
        self.portfolio_values = np.array(self.portfolio_values)
        return self.portfolio_values
    
    def calculate_metrics(self):
        """Temel performans metriklerini hesaplar."""
        if len(self.portfolio_values) == 0:
            self.run()
            
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Gunluk getiriler
        daily_returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        # Sharpe Orani (Yillik, 252 islem gunu varsayimi)
        if np.std(daily_returns) != 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        # Max Drawdown (En buyuk zirveden duse)
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (peak - self.portfolio_values) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # DUZELTME (Hata #6): FIFO eslestirme ile Win Rate hesaplama
        # Her SELL islemini en son eslesmemis BUY ile eslestirir
        buy_queue = []  # Henuz SELL ile eslesmemis BUY'lar
        wins = 0
        total_closed = 0
        for t in self.trades:
            if t['action'] == 'BUY':
                buy_queue.append(t)
            elif t['action'] == 'SELL' and buy_queue:
                matched_buy = buy_queue.pop(0)  # FIFO: Ilk giren ilk cikar
                total_closed += 1
                if t['price'] > matched_buy['price']:
                    wins += 1
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        
        metrics = {
            'Baslangic Sermayesi': f"${self.initial_capital:,.2f}",
            'Son Portfoy Degeri': f"${final_value:,.2f}",
            'Toplam Getiri (%)': f"{total_return:.2f}%",
            'Sharpe Orani': f"{sharpe_ratio:.4f}",
            'Max Drawdown (%)': f"{max_drawdown:.2f}%",
            'Toplam Islem Sayisi': len(self.trades),
            'Win Rate (%)': f"{win_rate:.2f}%"
        }
        
        return metrics
    
    def plot_results(self, save_path=None):
        """Portfoy deger grafigi ve islem noktalarini cizer."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        # Ust Grafik: Portfoy Degeri
        ax1.plot(self.portfolio_values, label='Portfoy Degeri', color='#2196F3', linewidth=1.5)
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Baslangic Sermayesi')
        ax1.set_title('Portfoy Performansi (Backtest)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfoy Degeri ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # AL/SAT noktalarini isaretleyelim
        for trade in self.trades:
            color = 'green' if trade['action'] == 'BUY' else 'red'
            marker = '^' if trade['action'] == 'BUY' else 'v'
            ax1.scatter(trade['step'], self.portfolio_values[trade['step']], 
                       color=color, marker=marker, s=30, zorder=5)
        
        # Alt Grafik: Drawdown
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (peak - self.portfolio_values) / peak * 100
        ax2.fill_between(range(len(drawdown)), drawdown, alpha=0.4, color='red')
        ax2.set_title('Drawdown (%)', fontsize=12)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Zaman Adimi')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Grafik kaydedildi: {save_path}")
        plt.close()

def main(ticker="BTC-USD"):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print("=" * 60)
    print(f"       {ticker} ENSEMBLE META-MODEL TAHMINLERI YUKLENIYOR")
    print("=" * 60)
    # DUZELTME (Hata #5): Orijinal (olceklenmemis) fiyatlari kullan
    meta_probs, df_scaled, df_original = compute_meta_probs(base_dir, ticker=ticker)

    closes = df_original['Close'].values

    # Zaman serisi/ML tarafinda kullanilan pencere boyutu ile uyumlu sekilde
    # sadece test bolumundeki (son %20) pencerelerin fiyat ve sinyallerini alalim
    num_windows = len(meta_probs)
    train_size = int(num_windows * 0.8)

    window_indices_test = np.arange(train_size, num_windows)
    price_indices = window_indices_test + (WINDOW_SIZE - 1)
    price_indices = price_indices[price_indices < len(closes)]

    prices = closes[price_indices]

    # Meta-modelin UP (1) sinifina verdigi olasiligi AL/SAT/TUT sinyallerine cevir
    meta_probs_test = meta_probs[train_size : train_size + len(price_indices)]
    signals = np.zeros_like(meta_probs_test, dtype=int)
    signals[meta_probs_test > 0.55] = 2     # AL
    signals[meta_probs_test < 0.45] = 0     # SAT
    signals[(meta_probs_test >= 0.45) & (meta_probs_test <= 0.55)] = 1  # TUT
    
    print("=" * 60)
    print(f"       {ticker} BACKTESTING MOTORU CALISTIRILIYOR (ENSEMBLE SINYAL)")
    print("=" * 60)
    
    bt = SimpleBacktester(prices, signals, initial_capital=10000, commission=0.001)
    bt.run()
    metrics = bt.calculate_metrics()

    # ── Buy-and-Hold Benchmark ──────────────────────────────────
    # Baslangicta al, sonuna kadar tut. En basit strateji budur.
    # Modelimiz bunu yenemiyorsa, gercek bir deger katmiyor demektir.
    bah_signals = np.full(len(prices), 2, dtype=int)   # hep AL
    bah_signals[0] = 2                                  # ilk adimda al
    bah_signals[1:] = 1                                 # geri kalaninda tut
    bt_bah = SimpleBacktester(prices, bah_signals, initial_capital=10000, commission=0.001)
    bt_bah.run()
    metrics_bah = bt_bah.calculate_metrics()

    # ── Always-Hold Baseline (hic islem yapma) ──────────────────
    # Nakit tutmak — %0 getiri referansi
    hold_signals = np.ones(len(prices), dtype=int)      # hep TUT (nakit)
    bt_hold = SimpleBacktester(prices, hold_signals, initial_capital=10000, commission=0.001)
    bt_hold.run()
    metrics_hold = bt_hold.calculate_metrics()

    # ── Karsilastirma Tablosu ────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {ticker} STRATEJI KARSILASTIRMA TABLOSU")
    print(f"{'='*65}")
    print(f"  {'Metrik':<28} {'Ensemble':>12} {'Buy&Hold':>12} {'Nakit Tut':>10}")
    print(f"  {'-'*62}")
    comparison_keys = ['Toplam Getiri (%)', 'Sharpe Orani', 'Max Drawdown (%)', 'Win Rate (%)']
    for key in comparison_keys:
        v_model = metrics.get(key, 'N/A')
        v_bah   = metrics_bah.get(key, 'N/A')
        v_hold  = metrics_hold.get(key, 'N/A')
        print(f"  {key:<28} {str(v_model):>12} {str(v_bah):>12} {str(v_hold):>10}")
    print(f"  {'Toplam Islem Sayisi':<28} {metrics.get('Toplam Islem Sayisi', 0):>12} "
          f"{metrics_bah.get('Toplam Islem Sayisi', 0):>12} {metrics_hold.get('Toplam Islem Sayisi', 0):>10}")
    print(f"{'='*65}")

    # Modelin Buy-and-Hold'u yenip yenmedigini belirt
    def _parse_pct(val_str):
        try:
            return float(str(val_str).replace('%', ''))
        except Exception:
            return 0.0
    model_ret = _parse_pct(metrics.get('Toplam Getiri (%)', '0%'))
    bah_ret   = _parse_pct(metrics_bah.get('Toplam Getiri (%)', '0%'))
    if model_ret > bah_ret:
        print(f"  SONUC: Ensemble, Buy-and-Hold'u {model_ret - bah_ret:.2f}% farkla ATTI.")
    else:
        print(f"  SONUC: Ensemble, Buy-and-Hold'un {bah_ret - model_ret:.2f}% GERISINDE kaldi.")
        print(f"  NOT: Bu, modelin aktif strateji olarak deger katmadigini gosterir.")

    # Grafik Kaydet
    results_dir = os.path.join(base_dir, 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    bt.plot_results(save_path=os.path.join(results_dir, f'{ticker}_backtest_portfolio.png'))

    # Buy-and-hold grafigi de kaydet
    bt_bah.plot_results(save_path=os.path.join(results_dir, f'{ticker}_buy_and_hold_portfolio.png'))

    print(f"\n{ticker} Backtest tamamlandi!")

if __name__ == "__main__":
    main()

