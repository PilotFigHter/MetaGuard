import joblib
import os
from sklearn.ensemble import VotingClassifier

# --- CONFIG ---
MODELS_DIR = './models/'

def create_and_save_ensemble():
    print("🔄 Loading pre-trained base models...")
    
    # 1. Load your existing fitted models
    try:
        rf_model = joblib.load(os.path.join(MODELS_DIR, 'realtime_randomforest.pkl'))
        xgb_model = joblib.load(os.path.join(MODELS_DIR, 'realtime_xgboost.pkl'))
        lgbm_model = joblib.load(os.path.join(MODELS_DIR, 'realtime_lightgbm.pkl'))
    except FileNotFoundError as e:
        print(f"❌ Error: Ensure all three base models exist in {MODELS_DIR}\n{e}")
        return

    # 2. Define the Ensemble (Voting Classifier)
    # We use 'soft' voting so it averages the probability scores
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        voting='soft',
        weights=[1, 2, 1]  # Giving more weight to XGBoost as it had the lowest False Alarm Rate
    )

    print("🛠️ Building the Ensemble structure...")
    
    # 3. "Force-Fit" the Ensemble
    # Since the sub-models are already trained, we manually set the internal 
    # attributes so we don't have to call .fit() on 16 million rows again.
    ensemble_model.estimators_ = [rf_model, xgb_model, lgbm_model]
    ensemble_model.le_ = joblib.load(os.path.join(MODELS_DIR, 'realtime_label_encoder.pkl'))
    ensemble_model.classes_ = rf_model.classes_

    # 4. Save the final Ensemble
    ensemble_path = os.path.join(MODELS_DIR, 'realtime_ensemble.pkl')
    joblib.dump(ensemble_model, ensemble_path)
    
    print(f"✅ Ensemble model successfully created and saved to: {ensemble_path}")

if __name__ == "__main__":
    create_and_save_ensemble()