"""
Generate error graphs for NN and Ensemble models per class
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os, joblib
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = ['Init Fwd Win Byts', 'Fwd Seg Size Min', 'Protocol', 'Fwd Header Len', 'Fwd Pkt Len Max', 'ACK Flag Cnt']
os.makedirs('evaluations', exist_ok=True)

# Load data
print("Loading data...")
files = glob.glob('data/*.csv')
dfs = []
for f in files[:3]:
    try:
        df = pd.read_csv(f, usecols=lambda x: x.strip() in FEATURES + ['Label'], nrows=30000)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    except: pass

df = pd.concat(dfs, ignore_index=True)
df['Label'] = df['Label'].astype(str).str.strip()
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

# Load models
ensemble = joblib.load('models/realtime_ensemble.pkl')
scaler = joblib.load('models/realtime_scaler.pkl')
X = df[FEATURES].values
X_scaled = scaler.transform(X)

# NN predictions
import torch
class Net(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Load NN model from pth file
nn_pth = torch.load('DeepLearning/results/nn_model.pth', map_location='cpu', weights_only=False)

nn_model = Net(len(FEATURES), len(nn_pth['le'].classes_))
nn_model.load_state_dict(nn_pth['model'])
nn_model.eval()
nn_scaler = nn_pth['scaler']
nn_le = nn_pth['le']

X_nn = nn_scaler.transform(X)
with torch.no_grad():
    nn_preds = nn_model(torch.FloatTensor(X_nn)).argmax(1).numpy()
nn_labels = nn_le.inverse_transform(nn_preds)

# Ensemble predictions
y_ensemble = ensemble.predict(X_scaled)
y_ensemble_labels = ensemble.predict(X_scaled)
# Convert numeric to string
mapping = {i: c for i, c in enumerate(ensemble.classes_)}
y_ensemble_labels = np.array([mapping.get(p, str(p)) for p in y_ensemble])

# True labels
y_true = df['Label'].values

# Get top 5 classes
top_classes = pd.Series(y_true).value_counts().head(5).index.tolist()
print(f"Top 5 classes: {top_classes}")

# Filter for top classes
mask = np.isin(y_true, top_classes)
y_true_filtered = y_true[mask]
nn_preds_filtered = nn_labels[mask]
ensemble_preds_filtered = y_ensemble_labels[mask]

# Calculate errors per class
nn_errors = {}
ensemble_errors = {}

for cls in top_classes:
    cls_mask = y_true_filtered == cls
    nn_errors[cls] = np.sum(nn_preds_filtered[cls_mask] != cls) / np.sum(cls_mask) * 100
    ensemble_errors[cls] = np.sum(ensemble_preds_filtered[cls_mask] != cls) / np.sum(cls_mask) * 100

# ============ GRAPH 1: NN Error per Class ============
fig, ax = plt.subplots(figsize=(12, 6))
classes_short = [c[:20] for c in top_classes]
x = np.arange(len(classes_short))
width = 0.6

bars = ax.bar(x, [nn_errors[c] for c in top_classes], width, color='#e74c3c', alpha=0.8, edgecolor='white', linewidth=2)

ax.set_ylabel('Error Rate (%)', fontsize=12)
ax.set_xlabel('Class', fontsize=12)
ax.set_title('Neural Network - Error Rate per Class (Top 5)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes_short, rotation=30, ha='right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max([nn_errors[c] for c in top_classes]) * 1.2 + 5)

for bar, cls in zip(bars, top_classes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{nn_errors[cls]:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('evaluations/nn_error_per_class.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: evaluations/nn_error_per_class.png")

# ============ GRAPH 2: NN vs Ensemble Error Comparison ============
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(top_classes))
width = 0.35

bars1 = ax.bar(x - width/2, [nn_errors[c] for c in top_classes], width, label='Neural Network', color='#e74c3c', alpha=0.8, edgecolor='white')
bars2 = ax.bar(x + width/2, [ensemble_errors[c] for c in top_classes], width, label='Ensemble (RF+XGB+LGBM)', color='#27ae60', alpha=0.8, edgecolor='white')

ax.set_ylabel('Error Rate (%)', fontsize=12)
ax.set_xlabel('Class', fontsize=12)
ax.set_title('Error Rate Comparison: Neural Network vs Ensemble (Top 5 Classes)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes_short, rotation=30, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

max_error = max(max([nn_errors[c] for c in top_classes]), max([ensemble_errors[c] for c in top_classes]))
ax.set_ylim(0, max_error * 1.2 + 5)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%',
            ha='center', va='bottom', fontsize=8, color='#e74c3c')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%',
            ha='center', va='bottom', fontsize=8, color='#27ae60')

plt.tight_layout()
plt.savefig('evaluations/nn_vs_ensemble_error_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: evaluations/nn_vs_ensemble_error_comparison.png")

print("\n=== Summary ===")
print(f"{'Class':<25} {'NN Error (%)':<15} {'Ensemble Error (%)':<15}")
print("-" * 55)
for cls in top_classes:
    print(f"{cls[:24]:<25} {nn_errors[cls]:<15.2f} {ensemble_errors[cls]:<15.2f}")

print("\nDone! Graphs saved to evaluations/")