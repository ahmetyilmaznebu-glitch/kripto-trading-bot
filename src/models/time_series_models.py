"""
time_series_models.py — LSTM/GRU Zaman Serisi Modelleri 
=========================================================================
Kripto Para Trading Botu | Ahmet Yılmaz

DEĞİŞİKLİK GÜNLÜĞÜ (ML4T Kitap Referanslı Düzeltmeler):
─────────────────────────────────────────────────────────
1. [KRİTİK] Purging gap eklendi (Bölüm 6, s.210-220)
   - Train/val arasında WINDOW_SIZE kadar boşluk bırakılıyor
   - Pencere örtüşmesinden kaynaklanan bilgi sızıntısı engellendi

2. [KRİTİK] 3-yollu split: Train / Val / Test (Bölüm 6, s.200-215)
   - Val: early stopping + threshold tuning
   - Test: SADECE final performans raporu
   - Double-dipping (çift kullanım) hatası giderildi

3. [YÜKSEK] Dejenere davranış tespiti ve müdahale (Bölüm 17)
   - Her epoch sonunda tahmin dağılımı izleniyor
   - Tek sınıf tahmini → LR reset + ağırlık pertürbasyonu
   - Dejenere sayacı ile erken uyarı sistemi

4. [YÜKSEK] Gradient akış izleme (Bölüm 17, s.560)
   - Gradient norm'ları loglanıyor
   - Sıfır gradient → alarm

5. [ORTA] F1-aware early stopping (Bölüm 6)
   - Sadece val_loss yerine val_f1 de izleniyor
   - F1 = 0 olan model "en iyi" olarak seçilemez

6. [ORTA] Focal Loss alpha düzeltmesi
   - Eski: alpha = clip(n_pos/(n_pos+n_neg)) → DOWN sınıfı görmezden gelinebiliyordu
   - Yeni: alpha = clip(n_neg/(n_pos+n_neg)) → azınlık sınıfı ağırlıklandırılır
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, f1_score as sklearn_f1
from sklearn.dummy import DummyClassifier


# ══════════════════════════════════════════════════════════════
#  SABITLER
# ══════════════════════════════════════════════════════════════

# DEĞİŞTİRİLDİ (2026-06-05): ml_config.py ile senkronize
# Eski: WINDOW_SIZE=60, PURGE_GAP=60
WINDOW_SIZE = 30          # Sliding window boyutu (ml_config.py ile uyumlu)
PURGE_GAP = 14            # Train/Val ve Val/Test arasındaki boşluk
DEGENERATE_PATIENCE = 5   # Art arda kaç epoch dejenere davranış toleransı
DEGENERATE_THRESHOLD = 0.95  # Tahminlerin %95'i tek sınıfsa → dejenere


# ══════════════════════════════════════════════════════════════
#  FOCAL LOSS
# ══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss: kolay örneklerin ağırlığını azaltır, zor örneklere odaklanır.

    DÜZELTİLDİ: alpha artık AZINLIK sınıfının oranını temsil eder.
    Eski kodda alpha = n_pos/(n_pos+n_neg) kullanılıyordu; bu durumda
    eğer UP çoğunluksa alpha yüksek olur ve DOWN sınıfının ağırlığı
    (1-alpha) çok düşer → model DOWN'ı hiç öğrenemez.

    Yeni: alpha = n_neg/(n_pos+n_neg) → dengesiz sınıflarda azınlık
    sınıfı daha fazla ağırlık alır.
    """
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


# ══════════════════════════════════════════════════════════════
#  TEMPORAL ATTENTION
# ══════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """Temporal Attention: hangi zaman adımlarının önemli olduğunu öğrenir."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, rnn_output):
        attn_weights = self.attention(rnn_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(rnn_output * attn_weights, dim=1)
        return context, attn_weights


# ══════════════════════════════════════════════════════════════
#  ANA MODEL
# ══════════════════════════════════════════════════════════════

class TimeSeriesNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 model_type="LSTM", bidirectional=False, use_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        dropout_rate = 0.3 if num_layers > 1 else 0.0

        if model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=bidirectional,
                               dropout=dropout_rate)
        elif model_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, bidirectional=bidirectional,
                              dropout=dropout_rate)
        else:
            raise ValueError(f"Desteklenmeyen model türü: {model_type}")

        D = 2 if bidirectional else 1
        rnn_output_size = hidden_size * D

        if use_attention:
            self.attention = Attention(rnn_output_size)

        self.head = nn.Sequential(
            nn.LayerNorm(rnn_output_size),
            nn.Dropout(0.2),
            nn.Linear(rnn_output_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.use_attention:
            out, _ = self.attention(out)
        else:
            out = out[:, -1, :]
        out = self.head(out)
        return out


# ══════════════════════════════════════════════════════════════
#  YARDIMCI: Dejenere Tespit
# ══════════════════════════════════════════════════════════════

def _check_degenerate(model, X_tensor, device, threshold=0.5):
    """
    Modelin tahmin dağılımını kontrol eder.
    Eğer tahminlerin büyük çoğunluğu tek sınıfsa → dejenere.

    Returns:
        is_degenerate (bool), up_ratio (float), predictions (np.ndarray)
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs > threshold).astype(int)
    up_ratio = preds.mean()
    is_deg = up_ratio > DEGENERATE_THRESHOLD or up_ratio < (1 - DEGENERATE_THRESHOLD)
    return is_deg, up_ratio, preds


def _perturb_weights(model, scale=0.01):
    """
    Model ağırlıklarına küçük gürültü ekler.
    Dejenere minimumdan çıkmaya yardımcı olur.
    """
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * scale
            param.add_(noise)


# ══════════════════════════════════════════════════════════════
#  YARDIMCI: Gradient İzleme
# ══════════════════════════════════════════════════════════════

def _get_grad_norm(model):
    """Model parametrelerinin gradient norm'unu hesaplar."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ══════════════════════════════════════════════════════════════
#  YARDIMCI: F1 Hesaplama
# ══════════════════════════════════════════════════════════════

def _compute_f1(model, X_tensor, y_true, device, threshold=0.5):
    """Model tahminlerinden F1 skoru hesaplar."""
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs > threshold).astype(int)
    y_int = y_true.astype(int) if isinstance(y_true, np.ndarray) else y_true
    return sklearn_f1(y_int, preds, zero_division=0)


# ══════════════════════════════════════════════════════════════
#  EĞİTİM FONKSİYONU (DÜZELTİLMİŞ)
# ══════════════════════════════════════════════════════════════

def train_model(model, X_train, y_train, X_val, y_val,
                num_epochs=150, batch_size=32, learning_rate=0.001,
                patience=20, label_smoothing=0.05):
    """
    Model eğitimi — Dejenere Tespitli, F1-Aware Early Stopping.

    DEĞİŞİKLİKLER:
    - Dejenere davranış tespiti: her 5 epoch'ta tahmin dağılımı kontrol edilir.
      Art arda DEGENERATE_PATIENCE kez dejenere → LR reset + weight perturbation.
    - F1-aware early stopping: val_loss düşse bile F1=0 ise "en iyi model" sayılmaz.
    - Gradient norm izleme: sıfır gradient alarm.
    - Val seti SADECE early stopping için kullanılır (threshold tuning ayrı).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ── Sınıf dengesizliği ──
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    print(f"  Sınıf dağılımı: UP={int(n_pos)}, DOWN={int(n_neg)}")

    # DÜZELTİLDİ: alpha = azınlık sınıfının oranı
    alpha = float(np.clip(n_neg / (n_pos + n_neg + 1e-8), 0.3, 0.7))

    # 2 FAZLI EĞİTİM:
    # Faz 1 (ilk 30 epoch): Düz BCE (gamma=0) → modeli karar sınırından uzaklaştır
    # Faz 2 (kalan epochlar): Focal Loss (gamma=1.0) → zor örneklere odaklan
    # NOT: Eski gamma=2.0 çok agresifti — sigmoid≈0.5'te gradient 4x azalıyordu
    criterion_bce = FocalLoss(alpha=alpha, gamma=0.0)   # = weighted BCE
    criterion_focal = FocalLoss(alpha=alpha, gamma=1.0)  # hafif focal
    warmup_epochs = 30
    print(f"  2-Fazlı Eğitim: BCE warmup({warmup_epochs} epoch) → Focal Loss(gamma=1.0)")
    print(f"  Alpha={alpha:.3f}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    # Label smoothing
    y_train_smooth = np.where(y_train == 1, 1.0 - label_smoothing, label_smoothing)

    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train_smooth))
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val.astype(np.float32)))

    # Balanced mini-batch: azinlik sinifini oversampling ile dengele
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    sample_weights = np.where(y_train >= 0.5, 1.0 / (class_counts[1] + 1), 1.0 / (class_counts[0] + 1))
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # Val tensörü (F1 ve dejenere kontrolü için)
    val_tensor = torch.FloatTensor(X_val)

    # ── Eğitim değişkenleri ──
    best_loss = float('inf')
    best_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0
    degenerate_count = 0
    degenerate_resets = 0
    max_resets = 3  # En fazla 3 kez reset

    print(f"\n  {'Epoch':>5} | {'T.Loss':>8} | {'V.Loss':>8} | {'V.Acc':>6} | "
          f"{'V.F1':>6} | {'UP%':>5} | {'GradNorm':>9} | {'LR':>10} | Durum")
    print(f"  {'─' * 85}")

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        train_loss = 0
        last_grad_norm = 0

        # Faz seçimi
        criterion = criterion_bce if epoch < warmup_epochs else criterion_focal

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            if len(batch_y.shape) == 1:
                batch_y = batch_y.view(-1, 1)
            loss = criterion(outputs, batch_y)

            # Degenerate penalty: tahminler tek sinifa yakinsa ceza
            probs_batch = torch.sigmoid(outputs)
            mean_prob = probs_batch.mean()
            degen_penalty = 0.5 * (mean_prob - 0.5).pow(2)
            loss = loss + degen_penalty

            optimizer.zero_grad()
            loss.backward()

            # Gradient izleme
            last_grad_norm = _get_grad_norm(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validation ──
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
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0
        val_f1 = _compute_f1(model, val_tensor, y_val, device)
        current_lr = optimizer.param_groups[0]['lr']

        # ── Dejenere kontrolü (her 5 epoch'ta) ──
        status = ""
        if (epoch + 1) % 5 == 0 or epoch < 3:
            is_deg, up_ratio, _ = _check_degenerate(model, val_tensor, device)
            if is_deg:
                degenerate_count += 1
                status = f"⚠️ DEJ({degenerate_count}/{DEGENERATE_PATIENCE})"

                # Art arda çok fazla dejenere → müdahale
                if degenerate_count >= DEGENERATE_PATIENCE and degenerate_resets < max_resets:
                    degenerate_resets += 1
                    degenerate_count = 0

                    # LR'yi yükselt ve ağırlıkları pertürbe et
                    new_lr = learning_rate * (0.5 ** degenerate_resets)
                    for pg in optimizer.param_groups:
                        pg['lr'] = new_lr
                    _perturb_weights(model, scale=0.02)
                    status = f"🔄 RESET #{degenerate_resets} (LR→{new_lr:.5f})"
            else:
                degenerate_count = max(0, degenerate_count - 1)
                status = "✅"
        else:
            _, up_ratio, _ = _check_degenerate(model, val_tensor, device)

        # ── Loglama ──
        if (epoch + 1) % 10 == 0 or epoch < 5 or status.startswith("🔄"):
            print(f"  {epoch+1:>5} | {train_loss:>8.5f} | {val_loss:>8.5f} | "
                  f"{val_acc:>5.3f} | {val_f1:>5.3f} | {up_ratio:>4.1%} | "
                  f"{last_grad_norm:>9.4f} | {current_lr:>10.7f} | {status}")

        # ── Sıfır gradient alarmı ──
        if last_grad_norm < 1e-7 and epoch > 5:
            print(f"  ⚠️  Epoch {epoch+1}: Gradient norm ≈ 0! Model öğrenmiyor olabilir.")

        # ── F1-aware early stopping ──
        # DÜZELTİLDİ: Sadece val_loss değil, val_f1 de kontrol edilir.
        # F1 = 0 olan model "en iyi" kabul edilmez (dejenere modeli kurtarır).
        improved = False
        if val_loss < best_loss and val_f1 > 0.01:
            best_loss = val_loss
            best_f1 = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            improved = True
        elif val_f1 > best_f1 + 0.02:
            # F1 önemli ölçüde arttıysa, loss biraz yüksek olsa da kabul et
            best_f1 = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            improved = True
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n  ⏹️  Early Stopping: {patience} epoch iyileşme yok. "
                  f"En iyi val_loss: {best_loss:.6f}, F1: {best_f1:.4f}")
            break

    # ── En iyi modeli yükle ──
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  ✅ En iyi model yüklendi (val_loss: {best_loss:.6f}, F1: {best_f1:.4f})")
    else:
        print(f"  ⚠️ F1 > 0 olan model bulunamadı! Model dejenere olmuş olabilir.")

    return model


# ══════════════════════════════════════════════════════════════
#  OPTIMAL THRESHOLD ARAMA (AYRI FONKSİYON)
# ══════════════════════════════════════════════════════════════

def find_optimal_threshold(model, X_data, y_data, device):
    """
    F1 skoru maximize eden threshold'u bulur.

    DÜZELTİLDİ: Bu işlem artık train_model içinde değil, ayrı bir
    fonksiyon. Böylece hangi veri seti üzerinde yapıldığı açıkça kontrol edilir.
    Val seti üzerinde çalıştırılır, test seti üzerinde ASLA.
    """
    model.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(X_data).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    y_int = y_data.astype(int)
    best_threshold = 0.5
    best_f1 = 0.0

    for thr in np.arange(0.30, 0.70, 0.01):
        preds = (probs > thr).astype(int)
        # Tek sınıf tahmini yapan threshold'u atla
        if preds.mean() > 0.95 or preds.mean() < 0.05:
            continue
        f1 = sklearn_f1(y_int, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    # Dejenere kontrol: tüm threshold'lar tek sınıf üretiyorsa → 0.50 kullan
    final_preds = (probs > best_threshold).astype(int)
    if final_preds.mean() > 0.95 or final_preds.mean() < 0.05:
        print(f"  ⚠️  Tüm threshold'lar tek sınıf üretiyor. Varsayılan 0.50 kullanılıyor.")
        best_threshold = 0.50
        best_f1 = sklearn_f1(y_int, (probs > 0.50).astype(int), zero_division=0)

    print(f"  🎯 Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold


# ══════════════════════════════════════════════════════════════
#  DEĞERLENDİRME (AYRI TEST SETİ ÜZERİNDE)
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, threshold, model_name, device):
    """
    DÜZELTİLDİ: Final değerlendirme AYRI test seti üzerinde yapılır.
    Bu veri ne eğitimde ne early stopping'de ne threshold tuning'de kullanılmıştır.
    Böylece gerçek out-of-sample performans raporlanır.
    """
    model.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(X_test).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    preds = (probs > threshold).astype(int)
    y_int = y_test.astype(int)

    acc = np.mean(preds == y_int)
    f1 = sklearn_f1(y_int, preds, zero_division=0)
    cm = confusion_matrix(y_int, preds)

    # Prediction dağılımı
    up_ratio = preds.mean()

    print(f"\n  {model_name} (threshold={threshold:.2f}) — TEST SETİ:")
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | UP oranı: {up_ratio:.1%}")
    print(f"  {'':12s} Tahmin DOWN  Tahmin UP")
    if cm.shape == (2, 2):
        print(f"  {'Gerçek DOWN':12s} {cm[0][0]:>11}  {cm[0][1]:>9}")
        print(f"  {'Gerçek UP  ':12s} {cm[1][0]:>11}  {cm[1][1]:>9}")
        if cm[0][1] == 0 and cm[1][0] == 0:
            print(f"  ⚠️  {model_name} yalnızca tek sınıfı tahmin ediyor!")
    else:
        print(f"  ⚠️  Confusion matrix beklenmeyen boyutta: {cm.shape}")
        print(f"  {cm}")

    return acc, f1


# ══════════════════════════════════════════════════════════════
#  3-YOLLU SPLİT (PURGING İLE)
# ══════════════════════════════════════════════════════════════

def split_with_purging(X, y, train_ratio=0.70, val_ratio=0.15):
    """
    Zaman serisi verisini Train / Val / Test olarak 3'e ayırır.
    Her set arasında PURGE_GAP kadar boşluk bırakır.

    Kitap Referansı (Bölüm 6, s.210-220):
    "Purging removes training observations whose labels overlap
     with the test period, preventing information leakage."

    Düzen:
    [───── TRAIN ─────][gap][──── VAL ────][gap][──── TEST ────]
                       ↑                   ↑
                    PURGE_GAP          PURGE_GAP
    """
    n = len(X)

    # Her gap'te PURGE_GAP örnek kaybediyoruz
    usable = n - 2 * PURGE_GAP
    train_end = int(usable * train_ratio)
    val_end = train_end + int(usable * val_ratio)

    # İndeksler
    train_idx_end = train_end
    val_idx_start = train_end + PURGE_GAP
    val_idx_end = val_end + PURGE_GAP
    test_idx_start = val_end + 2 * PURGE_GAP

    X_train = X[:train_idx_end]
    y_train = y[:train_idx_end]

    X_val = X[val_idx_start:val_idx_end]
    y_val = y[val_idx_start:val_idx_end]

    X_test = X[test_idx_start:]
    y_test = y[test_idx_start:]

    print(f"\n  📊 3-Yollu Split (Purging gap={PURGE_GAP}):")
    print(f"     Train : {len(X_train):>5} örnek  [0:{train_idx_end}]")
    print(f"     Val   : {len(X_val):>5} örnek  [{val_idx_start}:{val_idx_end}]")
    print(f"     Test  : {len(X_test):>5} örnek  [{test_idx_start}:{n}]")
    print(f"     Purge : {PURGE_GAP * 2:>5} örnek kaybı (güvenilirlik için)")

    # Sınıf dağılımlarını kontrol et
    for name, y_part in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        n_up = np.sum(y_part == 1)
        n_dn = np.sum(y_part == 0)
        total = len(y_part)
        print(f"     {name:5s}: UP={n_up} ({n_up/total:.1%}), DOWN={n_dn} ({n_dn/total:.1%})")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ══════════════════════════════════════════════════════════════
#  ANA FONKSİYON
# ══════════════════════════════════════════════════════════════

def main(ticker="BTC-USD"):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    from src.data.feature_store import FeatureStore
    try:
        store = FeatureStore(ticker)
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        return

    print(f"\n{'=' * 70}")
    print(f"  {ticker} LSTM/GRU Egitimi (Unified FeatureStore + Anti-Degenerate)")
    print(f"{'=' * 70}")
    print(f"  {store.summary()}")

    # ── Unified split (FeatureStore — tum modeller ayni bolumu kullanir) ──
    X_train, y_train = store.get_lstm_split("train")
    X_val, y_val = store.get_lstm_split("val")
    X_test, y_test = store.get_lstm_split("test")
    print(f"\n  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    input_size = X_train.shape[2]
    # DÜZELTİLDİ: Daha küçük model — 1200 eğitim örneği için 128/2 çok büyüktü
    # Küçük model overfitting'i azaltır ve daha hızlı yakınsar
    hidden_size = 64
    num_layers = 1
    output_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    thresholds = {}

    # ────────── LSTM ──────────
    print(f"\n{'─' * 70}")
    print(f"  {ticker} LSTM + Attention Eğitiliyor")
    print(f"{'─' * 70}")

    lstm_model = TimeSeriesNet(
        input_size, hidden_size, num_layers, output_size,
        model_type="LSTM", bidirectional=False, use_attention=True
    )
    lstm_model = train_model(
        lstm_model, X_train, y_train, X_val, y_val,
        num_epochs=150, batch_size=32, patience=40
    )
    # Threshold: Val seti üzerinde (test seti DOKUNULMAZ)
    lstm_thr = find_optimal_threshold(lstm_model, X_val, y_val, device)
    thresholds['LSTM'] = float(lstm_thr)

    # ────────── GRU ──────────
    print(f"\n{'─' * 70}")
    print(f"  {ticker} GRU + Attention Eğitiliyor")
    print(f"{'─' * 70}")

    gru_model = TimeSeriesNet(
        input_size, hidden_size, num_layers, output_size,
        model_type="GRU", bidirectional=False, use_attention=True
    )
    gru_model = train_model(
        gru_model, X_train, y_train, X_val, y_val,
        num_epochs=150, batch_size=32, patience=40
    )
    gru_thr = find_optimal_threshold(gru_model, X_val, y_val, device)
    thresholds['GRU'] = float(gru_thr)

    # ────────── FINAL DEĞERLENDİRME (TEST SETİ) ──────────
    print(f"\n{'═' * 70}")
    print(f"  FINAL DEĞERLENDİRME — TEST SETİ (hiç görülmemiş veri)")
    print(f"{'═' * 70}")

    lstm_acc, lstm_f1 = evaluate_model(
        lstm_model, X_test, y_test, lstm_thr, "LSTM", device)
    gru_acc, gru_f1 = evaluate_model(
        gru_model, X_test, y_test, gru_thr, "GRU", device)

    # ── Dummy Baseline (Test seti üzerinde) ──
    print(f"\n  {'─' * 50}")
    print(f"  Dummy Classifier Baseline (Referans)")
    X_test_flat = X_test.reshape(len(X_test), -1)
    y_test_int = y_test.astype(int)

    dummy_maj = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_maj.fit(X_test_flat, y_test_int)
    dummy_maj_acc = dummy_maj.score(X_test_flat, y_test_int)
    dummy_maj_f1 = sklearn_f1(y_test_int, dummy_maj.predict(X_test_flat), zero_division=0)

    dummy_strat = DummyClassifier(strategy='stratified', random_state=42)
    dummy_strat.fit(X_test_flat, y_test_int)
    dummy_strat_acc = dummy_strat.score(X_test_flat, y_test_int)
    dummy_strat_f1 = sklearn_f1(y_test_int, dummy_strat.predict(X_test_flat), zero_division=0)

    print(f"\n  {'Model':<28} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'─' * 48}")
    print(f"  {'Dummy (Çoğunluk Sınıfı)':<28} {dummy_maj_acc:>10.4f} {dummy_maj_f1:>8.4f}")
    print(f"  {'Dummy (Orantılı Rastgele)':<28} {dummy_strat_acc:>10.4f} {dummy_strat_f1:>8.4f}")
    print(f"  {'LSTM':<28} {lstm_acc:>10.4f} {lstm_f1:>8.4f}")
    print(f"  {'GRU':<28} {gru_acc:>10.4f} {gru_f1:>8.4f}")
    print(f"  {'─' * 48}")

    best_dummy = max(dummy_maj_acc, dummy_strat_acc)
    for mname, macc, mf1 in [("LSTM", lstm_acc, lstm_f1), ("GRU", gru_acc, gru_f1)]:
        if macc > best_dummy and mf1 > 0:
            print(f"  ✅ {mname}: Dummy baseline'ı GEÇTİ ({macc:.4f} > {best_dummy:.4f})")
        else:
            print(f"  ⚠️  {mname}: Dummy baseline'ı GEÇEMİYOR ({macc:.4f} ≤ {best_dummy:.4f})")
            if mf1 == 0:
                print(f"      → F1=0: Model tek sınıf tahmin ediyor. "
                      f"Feature kalitesini kontrol edin!")

    # ── Modelleri kaydet ──
    models_dir = os.path.join(base_dir, 'src', 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)

    torch.save(lstm_model.state_dict(),
               os.path.join(models_dir, f'{ticker}_lstm_best.pth'))
    torch.save(gru_model.state_dict(),
               os.path.join(models_dir, f'{ticker}_gru_best.pth'))

    thr_path = os.path.join(models_dir, f'{ticker}_dl_thresholds.json')
    with open(thr_path, 'w') as f:
        json.dump(thresholds, f)

    print(f"\n  💾 Modeller kaydedildi: {models_dir}")
    print(f"     Thresholds: {thresholds}")

    # ── ÖNEMLİ NOT ──
    print(f"\n{'═' * 70}")
    print(f"  ℹ️  NOT: Bu düzeltme modelin DAVRANIŞını iyileştirir ama")
    print(f"  tahmin KALİTESİ feature engineering'e bağlıdır.")
    print(f"  Eğer F1 hâlâ düşükse → data_pipeline.py'de:")
    print(f"    1. Log-return features ekleyin (ham fiyat yerine)")
    print(f"    2. Lag features ekleyin (return_lag_1, 5, 10, 20)")
    print(f"    3. Cyclical time encoding (sin/cos)")
    print(f"    4. Daha fazla indikatör (ADX, Stochastic, MFI...)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()