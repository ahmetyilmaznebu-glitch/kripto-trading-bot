import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier


def create_classification_target(df, threshold=0.0):
    """
    Siniflandirma hedef degiskenini olusturur.
    Target_Close > Close ise 1 (UP / ALIS), degilse 0 (DOWN / SATIS).
    """
    df = df.copy()
    df['Signal'] = (df['Target_Close'] > df['Close'] * (1 + threshold)).astype(int)
    return df


def train_classification_models(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Random Forest ve XGBoost siniflandirici modellerini egitir.
    
    Iyilestirmeler:
    - Hyperparameter tuning (RandomizedSearchCV + TimeSeriesSplit)
    - Sinif dengesizligi kontrolu (class_weight / scale_pos_weight)
    
    Args:
        feature_names: Ozellik adlari listesi. None ise otomatik uretilir.
    """
    results = {}
    
    # ────────── Random Forest ──────────
    print("\n--- Random Forest Egitiliyor (Hyperparameter Tuning ile) ---")
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500, 700],
        'max_depth': [5, 8, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'class_weight': ['balanced', 'balanced_subsample'],
        'max_features': ['sqrt', 'log2', 0.5, 0.8],
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf_search = RandomizedSearchCV(
        rf_base, rf_param_grid, n_iter=40, cv=tscv,
        scoring='f1', random_state=42, n_jobs=-1, verbose=1
    )
    rf_search.fit(X_train, y_train)
    rf_model = rf_search.best_estimator_
    
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    
    print(f"  RF En Iyi Parametreler: {rf_search.best_params_}")
    print(f"  RF Test Accuracy: {rf_acc:.4f}")
    print(f"  RF Test F1 Score: {rf_f1:.4f}")
    results['rf'] = {'model': rf_model, 'accuracy': rf_acc, 'f1': rf_f1}
    
    # ────────── XGBoost ──────────
    print("\n--- XGBoost Egitiliyor (Hyperparameter Tuning ile) ---")
    
    # Sinif dengesizligi icin scale_pos_weight hesapla
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0
    
    xgb_param_grid = {
        'n_estimators': [100, 200, 300, 500, 700],
        'max_depth': [3, 5, 6, 8, 10],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    }
    
    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        random_state=42
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_param_grid, n_iter=40, cv=tscv,
        scoring='f1', random_state=42, n_jobs=-1, verbose=1
    )
    xgb_search.fit(X_train, y_train)
    xgb_model = xgb_search.best_estimator_
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds)
    
    print(f"  XGB En Iyi Parametreler: {xgb_search.best_params_}")
    print(f"  XGB Test Accuracy: {xgb_acc:.4f}")
    print(f"  XGB Test F1 Score: {xgb_f1:.4f}")
    results['xgb'] = {'model': xgb_model, 'accuracy': xgb_acc, 'f1': xgb_f1}
    
    print("\n--- Feature Importance Raporu ---")
    # DUZELTME: Hardcoded liste yerine parametre olarak gelen isimleri kullan
    # Eksik sutun filtrelenmisken hardcoded liste etiket kaymasina neden oluyordu
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    rf_imp = rf_model.feature_importances_
    xgb_imp = xgb_model.feature_importances_
    for i, name in enumerate(feature_names[:len(rf_imp)]):
        print(f"  {name:18s} | RF: {rf_imp[i]:.4f} | XGB: {xgb_imp[i]:.4f}")

    # ── Confusion Matrix ─────────────────────────────────────────
    # Modelin gercekten ogrenip ogrenmedigini gosterir.
    # Sadece tek sinifi tahmin eden bir model confusion matrix'te hemen gorulur.
    print("\n--- Confusion Matrix ---")
    for name, preds in [("Random Forest", rf_preds), ("XGBoost", xgb_preds)]:
        cm = confusion_matrix(y_test, preds)
        print(f"\n  {name}:")
        print(f"  {'':12s} Tahmin DOWN  Tahmin UP")
        print(f"  {'Gercek DOWN':12s} {cm[0][0]:>11}  {cm[0][1]:>9}")
        print(f"  {'Gercek UP  ':12s} {cm[1][0]:>11}  {cm[1][1]:>9}")
        # Uyari: tek sinif tahmin ediliyorsa belirt
        if cm[0][1] == 0 or cm[1][0] == 0:
            print(f"  UYARI: {name} yalnizca tek sinifi tahmin ediyor — model ogrenemiyor olabilir!")

    # ── Dummy Classifier Baseline ────────────────────────────────
    # Hic ogrenmeyen bir model ne kadar basari gosterir?
    # Bunu gecemeyen model gercek bir deger katmiyor demektir.
    print("\n--- Dummy Classifier Baseline (Referans) ---")
    dummy_majority = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_majority.fit(X_train, y_train)
    dummy_maj_preds = dummy_majority.predict(X_test)
    dummy_maj_acc = accuracy_score(y_test, dummy_maj_preds)
    dummy_maj_f1  = f1_score(y_test, dummy_maj_preds, zero_division=0)

    dummy_strat = DummyClassifier(strategy='stratified', random_state=42)
    dummy_strat.fit(X_train, y_train)
    dummy_strat_preds = dummy_strat.predict(X_test)
    dummy_strat_acc = accuracy_score(y_test, dummy_strat_preds)
    dummy_strat_f1  = f1_score(y_test, dummy_strat_preds, zero_division=0)

    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'-'*43}")
    print(f"  {'Dummy (Cogunluk Sinifi)':<25} {dummy_maj_acc:>10.4f} {dummy_maj_f1:>8.4f}")
    print(f"  {'Dummy (Orantili Rastgele)':<25} {dummy_strat_acc:>10.4f} {dummy_strat_f1:>8.4f}")
    print(f"  {'Random Forest':<25} {rf_acc:>10.4f} {rf_f1:>8.4f}")
    print(f"  {'XGBoost':<25} {xgb_acc:>10.4f} {xgb_f1:>8.4f}")
    print(f"  {'-'*43}")
    best_dummy_acc = max(dummy_maj_acc, dummy_strat_acc)
    if rf_acc > best_dummy_acc and xgb_acc > best_dummy_acc:
        print(f"  SONUC: Her iki model de dummy baseline'i GECTI.")
    else:
        for mname, macc in [("Random Forest", rf_acc), ("XGBoost", xgb_acc)]:
            if macc <= best_dummy_acc:
                print(f"  UYARI: {mname} dummy baseline'i GECEMIYOR ({macc:.4f} <= {best_dummy_acc:.4f})")

    return results


def main(ticker="BTC-USD"):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    from src.data.feature_store import FeatureStore
    try:
        store = FeatureStore(ticker)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"\n{'=' * 60}")
    print(f"  {ticker} ML Siniflandirma (Unified FeatureStore)")
    print(f"{'=' * 60}")
    print(f"  {store.summary()}")

    # Unified split — tum modeller ayni bolumu kullanir
    X_train, y_train = store.get_xgb_split("train")
    X_val, y_val = store.get_xgb_split("val")
    X_test, y_test = store.get_xgb_split("test")
    feature_cols = store.feature_columns

    print(f"\n  Train: {X_train.shape[0]} ornek, Val: {X_val.shape[0]} ornek, Test: {X_test.shape[0]} ornek")
    print(f"  Sinif dagilimi (Train) - UP: {sum(y_train==1)}, DOWN: {sum(y_train==0)}")

    # Egitim: train + val birlesik (RandomizedSearchCV icinde CV yapiliyor)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    results = train_classification_models(X_trainval, y_trainval, X_test, y_test, feature_names=feature_cols)

    # Modelleri kaydet
    models_dir = os.path.join(base_dir, 'src', 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(results['rf']['model'], os.path.join(models_dir, f'{ticker}_rf_classifier.pkl'))
    joblib.dump(results['xgb']['model'], os.path.join(models_dir, f'{ticker}_xgb_classifier.pkl'))
    print(f"\n{ticker} modelleri '{models_dir}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()

