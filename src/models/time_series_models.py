import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier


class FocalLoss(nn.Module):
    """Focal Loss: kolay orneklerin agirligini azaltir, zor orneklere odaklanir.
    
    alpha: pozitif sinif agirligi (sinif dengesizligi icin)
    gamma: odaklanma parametresi (gamma=0 ise normal BCE'ye esit)
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        # p_t: dogru sinifin olasiligi
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # Focal weight: kolay orneklerin agirligini dusur
        focal_weight = (1 - p_t) ** self.gamma
        # Alpha weight: sinif dengesizligi
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class Attention(nn.Module):
    """Temporal Attention: hangi zaman adımlarının önemli olduğunu öğrenir."""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, rnn_output):
        # rnn_output: (batch, seq_len, hidden_size)
        attn_weights = self.attention(rnn_output)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        # Ağırlıklı toplam
        context = torch.sum(rnn_output * attn_weights, dim=1)  # (batch, hidden_size)
        return context, attn_weights


class TimeSeriesNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 model_type="LSTM", bidirectional=False, use_attention=True):
        super(TimeSeriesNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Dropout orani (katman sayisina gore)
        dropout_rate = 0.3 if num_layers > 1 else 0.0
        
        if model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                               bidirectional=bidirectional, dropout=dropout_rate)
        elif model_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                              bidirectional=bidirectional, dropout=dropout_rate)
        else:
            raise ValueError("Desteklenmeyen model turu: Sadece 'LSTM' veya 'GRU' desteklenir.")
        
        D = 2 if bidirectional else 1
        rnn_output_size = hidden_size * D
        
        # Attention layer
        if use_attention:
            self.attention = Attention(rnn_output_size)
        
        # Fully connected head
        # DUZELTME: Dropout oranları düşürüldü (0.3->0.2, 0.2->0.1).
        # Ardışık iki yüksek dropout, bilgi akışını kesiyor ve dejenere
        # çözümlere yol açıyordu.
        self.head = nn.Sequential(
            nn.LayerNorm(rnn_output_size),
            nn.Dropout(0.2),
            nn.Linear(rnn_output_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.rnn(x)
        
        if self.use_attention:
            # Attention: tüm zaman adımlarına bakarak önemli olanları seç
            out, _ = self.attention(out)  # (batch, hidden_size*D)
        else:
            # Eski yöntem: sadece son zaman adımı
            out = out[:, -1, :]
        
        out = self.head(out)
        return out


def train_model(model, X_train, y_train, X_val, y_val, num_epochs=300, batch_size=32,
                learning_rate=0.001, patience=30, label_smoothing=0.05):
    """
    Model egitimi: Focal Loss + Attention + Gradient Clipping + Label Smoothing + Cosine LR.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Sinif dengesizligi icin pos_weight hesapla
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight_val = n_neg / (n_pos + 1e-8)
    print(f"  Sinif dagilimi: UP={int(n_pos)}, DOWN={int(n_neg)}, pos_weight={pos_weight_val:.3f}")
    
    # Focal Loss: dejenere davranisi engellemek icin.
    # DUZELTME: alpha = pozitif sinifin (UP) veri icerisindeki orani, [0.25, 0.75]'e kirpilir.
    # Eski kod: alpha = min(pos_weight_val, 2.0) → alpha=1.0 oldugunda DOWN sinifinin
    # agirligi (1-alpha)=0 oluyordu; model hic DOWN ornegi goremeden egitiliyordu.
    alpha = float(np.clip(n_pos / (n_pos + n_neg + 1e-8), 0.25, 0.75))
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine Annealing: LR'yi düzgünce azaltır
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )
    
    # Label smoothing uygula
    y_train_smooth = y_train.copy()
    y_train_smooth = np.where(y_train_smooth == 1, 1.0 - label_smoothing, label_smoothing)
    
    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train_smooth))
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            if len(batch_y.shape) == 1:
               batch_y = batch_y.view(-1, 1)
               
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping: gradient patlamasını engeller
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.view(-1, 1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # Accuracy hesapla
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 10 == 0 or epoch < 3:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\n⚠️  Early Stopping: {patience} epoch boyunca iyilesme yok. "
                  f"En iyi val_loss: {best_loss:.6f}")
            break
            
    model.load_state_dict(best_model_state)
    print(f"✅ En iyi model yuklendi (val_loss: {best_loss:.6f})")
    
    # Optimal threshold arama (F1 maximize)
    model.eval()
    with torch.no_grad():
        val_tensor = torch.FloatTensor(X_val).to(device)
        val_logits = model(val_tensor)
        val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
    
    best_threshold = 0.5
    best_f1 = 0.0
    for thr in np.arange(0.3, 0.7, 0.01):
        preds = (val_probs > thr).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr
    
    print(f"🎯 Optimal threshold: {best_threshold:.2f} (Val F1: {best_f1:.4f})")
    return model, best_threshold

def main(ticker="BTC-USD"):
    # On islenmis veriyi yukle
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    x_path = os.path.join(base_dir, 'data', 'processed', f'{ticker}_X_windows.npy')
    y_path = os.path.join(base_dir, 'data', 'processed', f'{ticker}_y_targets.npy')
    
    if not os.path.exists(x_path):
        print(f"Hata: {ticker}_X_windows.npy bulunamadi. Once data_pipeline.py calistirin.")
        return
        
    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Egitim / Test Verisi Ayirma (%80/%20, kronolojik)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"\n--- {ticker} LSTM/GRU Egitimi (Focal Loss + Attention + Optimal Threshold) ---")
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
    
    input_size = X_train.shape[2]  # feature sayisi
    hidden_size = 128
    # DUZELTME: num_layers 3->2. 3 katmanli modelde dropout 2 kez uygulaniyordu;
    # bu gradient akisini kesiyor ve dejenere minimuma erken yakinsamamaya yol aciyordu.
    num_layers = 2
    output_size = 1

    thresholds = {}

    print(f"\n--- {ticker} LSTM + Attention Egitiliyor (150 epoch, Focal Loss) ---")
    lstm_model = TimeSeriesNet(input_size, hidden_size, num_layers, output_size,
                                model_type="LSTM", bidirectional=False, use_attention=True)
    # DUZELTME: num_epochs 300->150, patience 30->20. Daha kisa egitim dejenere
    # davranisi erken tespit etmeyi kolaylastirir.
    lstm_model, lstm_thr = train_model(lstm_model, X_train, y_train, X_val, y_val,
                                        num_epochs=150, batch_size=32, patience=20)
    thresholds['LSTM'] = float(lstm_thr)

    print(f"\n--- {ticker} GRU + Attention Egitiliyor (150 epoch, Focal Loss) ---")
    gru_model = TimeSeriesNet(input_size, hidden_size, num_layers, output_size,
                               model_type="GRU", bidirectional=False, use_attention=True)
    gru_model, gru_thr = train_model(gru_model, X_train, y_train, X_val, y_val,
                                      num_epochs=150, batch_size=32, patience=20)
    thresholds['GRU'] = float(gru_thr)
    
    # ── Confusion Matrix + Dummy Baseline Degerlendirmesi ────────────────────
    print("\n--- DL Modelleri Degerlendirme (Confusion Matrix + Dummy Baseline) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_val_int = y_val.astype(int)

    def _eval_dl_model(model, threshold, name):
        model.eval()
        with torch.no_grad():
            val_tensor = torch.FloatTensor(X_val).to(device)
            logits = model(val_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > threshold).astype(int)
        acc = np.mean(preds == y_val_int)
        cm = confusion_matrix(y_val_int, preds)
        tp = np.sum((preds == 1) & (y_val_int == 1))
        fp = np.sum((preds == 1) & (y_val_int == 0))
        fn = np.sum((preds == 0) & (y_val_int == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        print(f"\n  {name} (threshold={threshold:.2f}):")
        print(f"  {'':12s} Tahmin DOWN  Tahmin UP")
        print(f"  {'Gercek DOWN':12s} {cm[0][0]:>11}  {cm[0][1]:>9}")
        print(f"  {'Gercek UP  ':12s} {cm[1][0]:>11}  {cm[1][1]:>9}")
        if cm[0][1] == 0 or cm[1][0] == 0:
            print(f"  UYARI: {name} yalnizca tek sinifi tahmin ediyor — model ogrenemiyor olabilir!")
        return acc, f1

    lstm_acc, lstm_f1 = _eval_dl_model(lstm_model, lstm_thr, "LSTM")
    gru_acc,  gru_f1  = _eval_dl_model(gru_model,  gru_thr,  "GRU")

    # ── Dummy Classifier Baseline ─────────────────────────────────────────────
    print("\n--- Dummy Classifier Baseline (Referans) ---")
    dummy_majority = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_majority.fit(X_val.reshape(len(X_val), -1), y_val_int)
    dummy_maj_preds = dummy_majority.predict(X_val.reshape(len(X_val), -1))
    dummy_maj_acc = np.mean(dummy_maj_preds == y_val_int)
    tp = np.sum((dummy_maj_preds == 1) & (y_val_int == 1))
    fp = np.sum((dummy_maj_preds == 1) & (y_val_int == 0))
    fn = np.sum((dummy_maj_preds == 0) & (y_val_int == 1))
    dummy_maj_f1 = 2*(tp/(tp+fp+1e-8))*(tp/(tp+fn+1e-8)) / ((tp/(tp+fp+1e-8))+(tp/(tp+fn+1e-8))+1e-8)

    dummy_strat = DummyClassifier(strategy='stratified', random_state=42)
    dummy_strat.fit(X_val.reshape(len(X_val), -1), y_val_int)
    dummy_strat_preds = dummy_strat.predict(X_val.reshape(len(X_val), -1))
    dummy_strat_acc = np.mean(dummy_strat_preds == y_val_int)
    tp = np.sum((dummy_strat_preds == 1) & (y_val_int == 1))
    fp = np.sum((dummy_strat_preds == 1) & (y_val_int == 0))
    fn = np.sum((dummy_strat_preds == 0) & (y_val_int == 1))
    dummy_strat_f1 = 2*(tp/(tp+fp+1e-8))*(tp/(tp+fn+1e-8)) / ((tp/(tp+fp+1e-8))+(tp/(tp+fn+1e-8))+1e-8)

    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'-'*43}")
    print(f"  {'Dummy (Cogunluk Sinifi)':<25} {dummy_maj_acc:>10.4f} {dummy_maj_f1:>8.4f}")
    print(f"  {'Dummy (Orantili Rastgele)':<25} {dummy_strat_acc:>10.4f} {dummy_strat_f1:>8.4f}")
    print(f"  {'LSTM':<25} {lstm_acc:>10.4f} {lstm_f1:>8.4f}")
    print(f"  {'GRU':<25} {gru_acc:>10.4f} {gru_f1:>8.4f}")
    print(f"  {'-'*43}")
    best_dummy_acc = max(dummy_maj_acc, dummy_strat_acc)
    for mname, macc in [("LSTM", lstm_acc), ("GRU", gru_acc)]:
        if macc > best_dummy_acc:
            print(f"  SONUC: {mname} dummy baseline'i GECTI ({macc:.4f} > {best_dummy_acc:.4f}).")
        else:
            print(f"  UYARI: {mname} dummy baseline'i GECEMIYOR ({macc:.4f} <= {best_dummy_acc:.4f}).")

    # Modelleri kaydet
    models_dir = os.path.join(base_dir, 'src', 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)

    torch.save(lstm_model.state_dict(), os.path.join(models_dir, f'{ticker}_lstm_best.pth'))
    torch.save(gru_model.state_dict(), os.path.join(models_dir, f'{ticker}_gru_best.pth'))
    
    # Optimal thresholdlari kaydet
    thr_path = os.path.join(models_dir, f'{ticker}_dl_thresholds.json')
    with open(thr_path, 'w') as f:
        json.dump(thresholds, f)
    print(f"\n{ticker} modelleri ve thresholdlar '{models_dir}' dizinine kaydedildi.")
    print(f"  Thresholds: {thresholds}")

if __name__ == "__main__":
    main()
