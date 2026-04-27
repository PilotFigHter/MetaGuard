import pandas as pd
import numpy as np
import glob
import os
import joblib
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler

# --- CONFIG ---
DATA_DIR = './data/' 
SAMPLE_RATE = 0.25 
CHUNK_SIZE = 150000 

# Pruned features based on correlation analysis
# best_features = [
#     'Fwd Seg Size Min', 'Bwd Pkts/s', 'Flow Pkts/s', 
#     'Init Fwd Win Byts', 'Flow IAT Max', 'Flow IAT Mean'
# ]

best_features = [
    'Fwd Seg Size Min',  # Highest unique correlation (0.49)
    'Init Fwd Win Byts', # Handshake signature
]

def run_pipeline():
    start_time = time.time()
    print("\n🚀 STARTING COMPLETE TRIPLE-MODEL PIPELINE")
    
    # --- PHASE 1: DATA LOADING & INF FILTERING ---
    all_files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv")) if "live" not in f]
    df_list = []
    for f in all_files:
        try:
            for chunk in pd.read_csv(f, usecols=lambda x: x.strip() in best_features or x.strip() == 'Label', 
                                    chunksize=CHUNK_SIZE, low_memory=False):
                chunk.columns = chunk.columns.str.strip()
                df_list.append(chunk.sample(frac=SAMPLE_RATE, random_state=42))
        except Exception as e: print(f"❌ Error loading {f}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    del df_list
    
    print("🧹 Cleaning data (Fixing Infinity & NaN)...")
    df['Label'] = df['Label'].astype(str).str.strip()
    for col in best_features: 
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Explicitly catch infinity to prevent StandardScaler ValueError
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

    # --- PHASE 3: OVERSAMPLING ---
    stealth_candidates = ['SQL Injection', 'Brute Force -Web', 'Brute Force -XSS', 'Infilteration']
    sampling_targets = {list(le.classes_).index(l): max(np.sum(y_train == list(le.classes_).index(l)), 15000) 
                        for l in stealth_candidates if l in le.classes_}
    ros = RandomOverSampler(sampling_strategy=sampling_targets, random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_scaled, y_train)

    # --- PHASE 4: GRID SEARCH & PERSISTENCE ---
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model_configs = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "params": {'n_estimators': [50, 100], 'max_depth': [10, 15]}
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric='mlogloss'),
            "params": {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]}
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, verbose=-1),
            "params": {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2], 'num_leaves': [31, 62]}
        }
    }

    # Thresholds to help catch stealthy attacks
    threshold_map = {'Benign': 0.85, 'Infiltration': 0.15, 'DoS attacks-SlowHTTPTest': 0.10, 'SQL Injection': 0.15}

    for name, config in model_configs.items():
        print(f"\n" + "-"*40 + f"\n⏳ TRAINING & EVALUATING: {name}")
        m_start = time.time()
        
        grid = GridSearchCV(config['model'], config['params'], cv=skf, scoring='f1_macro', n_jobs=-1, verbose=1)
        grid.fit(X_train_res, y_train_res)
        best_model = grid.best_estimator_

        # 💾 SAVE MODEL
        save_name = f'models/best_{name.lower()}_model.pkl'
        joblib.dump(best_model, save_name)
        
        # 📊 PREDICTION WITH TUNED THRESHOLDS
        probs = best_model.predict_proba(X_test_scaled)
        y_pred = []
        for p in probs:
            idx = np.argmax(p)
            label = le.classes_[idx]
            if p[idx] >= threshold_map.get(label, 0.70):
                y_pred.append(idx)
            else:
                y_pred.append(list(le.classes_).index('Benign'))

        # 📝 CLASSIFICATION REPORT
        print(f"\n📝 CLASSIFICATION REPORT FOR {name}:")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
        
        # 🖼️ CONFUSION MATRIX VISUALIZATION
        plt.figure(figsize=(14, 10))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='viridis', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'{name} Final Confusion Matrix')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.tight_layout()
        
        img_name = f'evaluations/confusion_matrix_{name.lower()}.png'
        plt.savefig(img_name)
        plt.close()
        
        print(f"✅ Finished {name} in {(time.time() - m_start)/60:.2f}m")
        print(f"💾 Saved Model: {save_name} | Saved Plot: {img_name}")

    # Final Shared Assets
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print(f"\n✨ COMPLETE! Total Time: {(time.time() - start_time)/60:.2f}m")

if __name__ == "__main__":
    run_pipeline()