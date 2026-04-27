# MetaGuard - Network Intrusion Detection System

## Project Structure

```
MetaGuard_New/
├── Data/                    # CICIDS2017 dataset CSV files
├── Models/                  # Trained model files (.pkl)
├── Training/                # Model training scripts
├── DeepLearning/            # Neural Network training & results
├── EDA/                     # Exploratory Data Analysis files & graphs
├── Evaluations/             # Model evaluation results & graphs
├── MetaWeb/                 # Flask web application
├── Scripts/                 # Utility scripts
├── Documentation/           # Project documentation
└── run.bat                  # Quick launch script
```

## Folder Descriptions

### Data/
- Contains CICIDS2017 dataset CSV files (download from Kaggle)
- Used for training and evaluation
- **Note:** Dataset not included in repo due to size. Download from:
  https://www.kaggle.com/datasets/cicdataset/cicids2017

### Models/
- `realtime_ensemble.pkl` - Main ensemble model (RF + XGBoost + LightGBM)
- `realtime_scaler.pkl` - StandardScaler for normalization
- `realtime_label_encoder.pkl` - Label encoder
- `nn_label_encoder.pkl` - NN label encoder
- `nn_scaler.pkl` - NN scaler

### Training/
- `train_models.py` - Train individual models (RF, XGB, LGBM)
- `create_ensemble.py` - Create ensemble model
- `evaluate_models.py` - Evaluate model performance
- `train_models_rl.py` - Training with reinforcement learning

### DeepLearning/
- `nn_intrusion_detection.py` - Neural network training script
- `results/` - NN training results and graphs

### EDA/
- `generate_eda.py` - Generate EDA graphs
- `generate_extra_eda.py` - Generate additional EDA graphs
- `*.png` - EDA visualization graphs

### Evaluations/
- Confusion matrix graphs
- Feature importance graphs
- Model comparison graphs
- Reality test results

### MetaWeb/
- `app.py` - Flask web application
- `templates/index.html` - Web interface
- UML diagrams

### Scripts/
- `generate_error_graphs.py` - Error rate per class graphs
- `generate_uml.py` - UML diagram generator
- `run_eda.py` - EDA runner
- `predict.py` - Standalone prediction script

## Quick Start

1. Install dependencies:
```bash
pip install flask scikit-learn xgboost lightgbm pandas numpy matplotlib torch scapy joblib
```

2. Run the web application:
```bash
run.bat
```

3. Open browser: http://127.0.0.1:4000

## Models Used

1. **Ensemble** - RF + XGBoost + LightGBM with Soft Voting
2. **Neural Network** - 6 -> 64 -> 32 -> 15 neurons (ReLU activation)

## Features Used (6 real-time features)

1. Init Fwd Win Byts
2. Fwd Seg Size Min
3. Protocol
4. Fwd Header Len
5. Fwd Pkt Len Max
6. ACK Flag Cnt