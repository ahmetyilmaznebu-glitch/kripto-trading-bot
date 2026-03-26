import os
import subprocess
import sys

def run_script(script_path, step_name):
    print("=" * 60)
    print(f"🚀 BASLIYOR: {step_name}")
    print(f"Çalıştırılan dosya: {script_path}")
    print("=" * 60)
    
    if not os.path.exists(script_path):
        print(f"❌ HATA: {script_path} bulunamadı!")
        sys.exit(1)

    # İşletim sistemine göre python komutunu ayarla
    python_cmd = sys.executable

    try:
        # Scripti çalıştır ve çıktısını anlık olarak terminale yansıt
        process = subprocess.Popen([python_cmd, script_path], 
                                   stdout=sys.stdout, 
                                   stderr=sys.stderr)
        process.communicate()
        
        if process.returncode != 0:
            print(f"\n❌ HATA: {step_name} çalışırken bir hata oluştu (Return code: {process.returncode}).")
            sys.exit(process.returncode)
            
        print(f"\n✅ BASARILI: {step_name} tamamlandı.\n")
        
    except Exception as e:
        print(f"\n❌ BEKLENMEYEN HATA: {step_name} çalıştırılamadı.\nDetay: {e}")
        sys.exit(1)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Çalıştırılacak scriptlerin sırası ve yolları
    pipeline = [
        {
            "name": "Aşama 1: Veri Toplama ve Ön İşleme",
            "path": os.path.join(base_dir, "src", "data", "data_pipeline.py")
        },
        {
            "name": "Aşama 2a: LSTM ve GRU Modellerinin Eğitimi",
            "path": os.path.join(base_dir, "src", "models", "time_series_models.py")
        },
        {
            "name": "Aşama 2b: Random Forest ve XGBoost Modellerinin Eğitimi",
            "path": os.path.join(base_dir, "src", "models", "ml_classification_models.py")
        },
        {
            "name": "Aşama 3: Ensemble (Stacking Meta-Model) Eğitimi",
            "path": os.path.join(base_dir, "src", "models", "ensemble_model.py")
        },
        {
            "name": "Aşama 4: Deep RL (PPO) Ajan Eğitimi",
            "path": os.path.join(base_dir, "src", "rl", "train_agent.py")
        },
        {
            "name": "Aşama 5: Backtesting ve Raporlama",
            "path": os.path.join(base_dir, "src", "utils", "backtester.py")
        }
    ]

    print("Yapay Zeka Destekli Hibrit Trading Sistemi - Tüm Süreç Başlatılıyor...\n")
    
    for step in pipeline:
        run_script(step["path"], step["name"])
        
    print("🎉 TEBRİKLER! Tüm aşamalar sırasıyla ve başarıyla tamamlandı.")
    print("Sonuçlara 'data/results/' dizininden ulaşabilirsiniz.")

if __name__ == "__main__":
    main()
