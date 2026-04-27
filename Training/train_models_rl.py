import pandas as pd
import numpy as np
import glob
import os
import joblib
import gc
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# --- CONFIG ---
DATA_DIR = './data/' 
SAMPLE_RATE = 0.25 
CHUNK_SIZE = 150000 

# Verified Real-Time Metadata Features (No Future Bias)
best_features = [
    'Init Fwd Win Byts', 
    'Fwd Seg Size Min', 
    'Protocol', 
    'Fwd Header Len', 
    'Fwd Pkt Len Max', 
    'ACK Flag Cnt'
]

def run_pipeline():
    start_time = time.time()
    print("\n" + "="*60)
    print("🚀 METAGUARD: REAL-TIME OPTIMIZED TRAINING PIPELINE")
    print("="*60)
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluations', exist_ok=True)

    # --- PHASE 1: LOADING WITH PROGRESS BARS ---
    all_files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv")) if "live" not in f]
    df_list = []
    
    print(f"📂 Scanning {len(all_files)} files...")
    for f in tqdm(all_files, desc="Reading CSVs", unit="file"):
        try:
            # Load only the required columns and strip whitespace immediately
            for chunk in pd.read_csv(f, usecols=lambda x: x.strip() in best_features or x.strip() == 'Label', 
                                    chunksize=CHUNK_SIZE, low_memory=False):
                chunk.columns = chunk.columns.str.strip()
                df_list.append(chunk.sample(frac=SAMPLE_RATE, random_state=42))
        except Exception as e: 
            print(f"\n❌ Error loading {f}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()
    
    print(f"\n✅ Data Loaded. Total Rows: {len(df):,}")
    print("🧹 Cleaning & Casting Types...")
    
    df['Label'] = df['Label'].astype(str).str.strip()
    for col in best_features: 
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=best_features + ['Label'], inplace=True)
    df = df[df['Label'].str.lower() != 'label']
    
    # --- PHASE 2: PREPROCESSING ---
    le = LabelEncoder()
    y = le.fit_transform(df['Label'])
    X = df[best_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- PHASE 3: BALANCING (Oversampling Stealthy Attacks) ---
    print("\n⚖️ Balancing Classes (SMOTE-alternative)...")
    stealth_candidates = ['SQL Injection', 'Brute Force -Web', 'Brute Force -XSS', 'Infilteration']
    sampling_targets = {list(le.classes_).index(l): max(np.sum(y_train == list(le.classes_).index(l)), 15000) 
                        for l in stealth_candidates if l in le.classes_}
    
    ros = RandomOverSampler(sampling_strategy=sampling_targets, random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_scaled, y_train)

    # --- PHASE 4: HIGH-VISIBILITY TRAINING ---
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    model_configs = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced', verbose=1),
            "params": {'n_estimators': [100], 'max_depth': [15]}
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=1),
            "params": {'n_estimators': [100], 'learning_rate': [0.1]}
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, force_row_wise=True, verbose=-1),
            "params": {'n_estimators': [100], 'learning_rate': [0.1]}
        }
    }

    # Thresholds to favor Benign unless attack is certain
    threshold_map = {'Benign': 0.80, 'Infiltration': 0.20, 'SQL Injection': 0.20}

    for name, config in model_configs.items():
        print(f"\n" + "-"*60)
        print(f"⏳ TUNING & TRAINING: {name}")
        print("-"*60)
        m_start = time.time()
        
        # verbose=3 shows detailed cross-validation progress
        grid = GridSearchCV(config['model'], config['params'], cv=skf, scoring='f1_macro', n_jobs=-1, verbose=3)
        grid.fit(X_train_res, y_train_res)
        best_model = grid.best_estimator_

        # Save model
        joblib.dump(best_model, f'models/realtime2_{name.lower()}.pkl')
        
        # Evaluation with logic
        probs = best_model.predict_proba(X_test_scaled)
        y_pred = []
        for p in probs:
            idx = np.argmax(p)
            lbl = le.classes_[idx]
            if p[idx] >= threshold_map.get(lbl, 0.70):
                y_pred.append(idx)
            else:
                y_pred.append(list(le.classes_).index('Benign'))

        m_duration = (time.time() - m_start) / 60
        print(f"\n✅ {name} Finished in {m_duration:.2f}m")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
        
        # Save Confusion Matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='viridis', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'{name} Real-Time Performance')
        plt.tight_layout()
        plt.savefig(f'evaluations/cm_realtime2_{name.lower()}.png')
        plt.close()

    # Final Assets
    joblib.dump(scaler, 'models/realtime2_scaler.pkl')
    joblib.dump(le, 'models/realtime2_label_encoder.pkl')
    
    total_duration = (time.time() - start_time) / 60
    print("\n" + "="*60)
    print(f"✨ ALL DONE! Total Execution Time: {total_duration:.2f} minutes")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()