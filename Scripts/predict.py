import pandas as pd
import joblib
import numpy as np
import os

# --- CONFIG ---
# Update this to the CSV you want to analyze
INPUT_CSV = 'data/02-14-2018.csv' 
OUTPUT_CSV = 'traffic_predictions_results.csv'

# Must match the features used in training
FEATURES = [
    'Fwd Seg Size Min', 'Bwd Pkts/s', 'Flow Pkts/s', 
    'Init Fwd Win Byts', 'Flow IAT Max', 'Flow IAT Mean'
]

def predict_traffic():
    print(f"📂 Loading model assets...")
    try:
        # Load the best performing model (change name if you picked XGBoost)
        model = joblib.load('models/best_randomforest_model.pkl') 
        scaler = joblib.load('models/scaler.pkl')
        le = joblib.load('models/label_encoder.pkl')
    except FileNotFoundError:
        print("❌ Error: Missing .pkl files. Run training script first.")
        return

    print(f"📖 Reading {INPUT_CSV}...")
    # Read only required columns to save memory
    df = pd.read_csv(INPUT_CSV)
    df = df[FEATURES]
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Check if all features exist in the CSV
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"❌ Error: CSV is missing required features: {missing}")
        return

    # --- DATA SANITIZATION ---
    # Convert to numeric and fix infinity/NaN issues
    X = df[FEATURES].copy()
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True) # Fill missing with 0 for prediction safety

    # --- INFERENCE ---
    print("🧠 Running predictions...")
    X_scaled = scaler.transform(X)
    
    # Get probabilities and final class
    probs = model.predict_proba(X_scaled)
    preds = np.argmax(probs, axis=1)
    
    # Decode numeric labels back to names (e.g., 0 -> Benign)
    df['Predicted_Label'] = le.inverse_transform(preds)
    df['Confidence_Score'] = np.max(probs, axis=1)

    # --- FILTERING THREATS ---
    threats = df[df['Predicted_Label'] != 'Benign']
    
    print("\n" + "="*30)
    print(f"✅ PREDICTION COMPLETE")
    print(f"Total Rows Processed: {len(df)}")
    print(f"Threats Detected: {len(threats)}")
    print("="*30)

    # Save results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Results saved to {OUTPUT_CSV}")

    if len(threats) > 0:
        print("\n🚨 TOP DETECTED THREATS:")
        print(threats['Predicted_Label'].value_counts())

if __name__ == "__main__":
    predict_traffic()