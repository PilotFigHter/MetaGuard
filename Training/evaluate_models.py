import pandas as pd
import numpy as np
import glob
import os
import joblib
import gc
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Ensure terminal can handle emojis/UTF-8 output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- CONFIG ---
DATA_DIR = './data/'
MODELS_DIR = './models/'
EVAL_DIR = './evaluations/'
CHUNK_SIZE = 250000 
PRED_BATCH_SIZE = 100000
REPORT_FILE = os.path.join(EVAL_DIR, "master_combined_report_realtime.txt")
PLOT_FILE = os.path.join(EVAL_DIR, "model_comparison_f1_realtime.png")

os.makedirs(EVAL_DIR, exist_ok=True)

FEATURES = [
    'Init Fwd Win Byts', 
    'Fwd Seg Size Min', 
    'Protocol', 
    'Fwd Header Len', 
    'Fwd Pkt Len Max', 
    'ACK Flag Cnt'

]

def load_all_data_to_memory(scaler, le):
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    x_collector = []
    y_collector = []

    print(f"🚀 Starting Master Combination of {len(all_files)} files...")
    
    for f_path in tqdm(all_files, desc="Processing CSV Files", unit="file"):
        try:
            reader = pd.read_csv(f_path, usecols=lambda x: x.strip() in FEATURES or x.strip() == 'Label', 
                                 chunksize=CHUNK_SIZE, low_memory=False)
            
            for chunk in reader:
                chunk.columns = chunk.columns.str.strip()
                chunk['Label'] = chunk['Label'].astype(str).str.strip()
                
                for col in FEATURES:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                
                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                chunk.dropna(subset=FEATURES + ['Label'], inplace=True)
                chunk = chunk[chunk['Label'].str.lower() != 'label']
                
                if not chunk.empty:
                    x_collector.append(scaler.transform(chunk[FEATURES]).astype(np.float32))
                    y_collector.append(le.transform(chunk['Label']).astype(np.int8))
            gc.collect()
        except Exception as e:
            print(f"⚠️ Error skipping {os.path.basename(f_path)}: {e}")

    print("\n🔗 Concatenating Master Arrays...")
    X_master = np.vstack(x_collector)
    y_master = np.concatenate(y_collector)
    
    del x_collector, y_collector
    gc.collect()
    
    print(f"✅ Master Dataset Ready: {X_master.shape[0]:,} rows.")
    return X_master, y_master

def run_master_evaluation():
    # Load Assets
    scaler = joblib.load(os.path.join(MODELS_DIR, 'realtime_scaler.pkl'))
    le = joblib.load(os.path.join(MODELS_DIR, 'realtime_label_encoder.pkl'))
    
    X_master, y_master = load_all_data_to_memory(scaler, le)

    print("🤖 Loading sub-models for Ensemble...")
    rf_sub = joblib.load(os.path.join(MODELS_DIR, 'realtime_randomforest.pkl'))
    xgb_sub = joblib.load(os.path.join(MODELS_DIR, 'realtime_xgboost.pkl'))
    lgbm_sub = joblib.load(os.path.join(MODELS_DIR, 'realtime_lightgbm.pkl'))

    model_files = {
        "RandomForest": 'realtime_randomforest.pkl',
        "XGBoost": 'realtime_xgboost.pkl',
        "LightGBM": 'realtime_lightgbm.pkl',
        "Ensemble": 'realtime_ensemble.pkl'
    }

    f1_results = {}

    # Open with utf-8 encoding to fix the UnicodeEncodeError
    with open(REPORT_FILE, "w", encoding="utf-8") as f_out:
        header = "🛡️ MASTER DATASET EVALUATION (Aggregated)\n" + "="*60 + "\n"
        f_out.write(header)
        print("\n" + header)

        for name, filename in model_files.items():
            m_path = os.path.join(MODELS_DIR, filename)
            if not os.path.exists(m_path):
                print(f"Skipping {name}, file not found.")
                continue
            
            print(f"\n🧠 Evaluating {name}...")
            model = joblib.load(m_path)
            
            # Ensemble Fit Fix
            if name == "Ensemble":
                model.estimators_ = [rf_sub, xgb_sub, lgbm_sub]
                model.classes_ = rf_sub.classes_
                model.le_ = le
            
            y_pred_list = []
            num_batches = int(np.ceil(X_master.shape[0] / PRED_BATCH_SIZE))
            
            for i in tqdm(range(0, X_master.shape[0], PRED_BATCH_SIZE), 
                          total=num_batches, desc=f"Predicting {name}", leave=False):
                batch_pred = model.predict(X_master[i:i+PRED_BATCH_SIZE])
                
                # Convert strings back to numeric if necessary
                if isinstance(batch_pred[0], (str, np.str_)):
                    batch_pred = le.transform(batch_pred)
                y_pred_list.append(batch_pred.astype(np.int8))
            
            y_pred_all = np.concatenate(y_pred_list)
            
            # --- Scoring ---
            present_classes = np.unique(np.concatenate([y_master, y_pred_all]))
            target_names_filtered = [le.classes_[i] for i in present_classes]

            report_dict = classification_report(y_master, y_pred_all, labels=present_classes, 
                                               target_names=target_names_filtered, output_dict=True, zero_division=0)
            f1_results[name] = report_dict['macro avg']['f1-score']

            report_text = classification_report(y_master, y_pred_all, labels=present_classes, 
                                               target_names=target_names_filtered, zero_division=0)
            
            # False Alarm Rate
            benign_idx = np.where(le.classes_ == 'Benign')[0][0]
            total_benign = np.sum(y_master == benign_idx)
            false_alarms = np.sum((y_master == benign_idx) & (y_pred_all != benign_idx))
            fa_rate = (false_alarms/total_benign)*100 if total_benign > 0 else 0

            output = f"\n📊 MODEL: {name}\n" + "-"*30 + f"\n{report_text}\n"
            output += f"🚨 Global False Alarm Rate: {fa_rate:.2f}% ({false_alarms:,} cases)\n"
            
            print(output)
            f_out.write(output)
            
            del model, y_pred_list, y_pred_all
            gc.collect()

    # --- PLOTTING ---
    print("\n📊 Generating comparison plot...")
    plt.figure(figsize=(10, 6))
    names = list(f1_results.keys())
    scores = list(f1_results.values())
    
    sns.barplot(x=names, y=scores, palette='magma')
    plt.title('Final Project Comparison: Macro F1-Score (All Files)', fontsize=14)
    plt.ylabel('Macro F1-Score', fontsize=12)
    plt.ylim(0, 1.1)
    
    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"🖼️ Plot saved: {PLOT_FILE}")
    print(f"💾 Report saved: {REPORT_FILE}")

if __name__ == "__main__":
    run_master_evaluation()