import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

os.makedirs("DeepLearning/results", exist_ok=True)

FEATURES = ['Init Fwd Win Byts', 'Fwd Seg Size Min', 'Protocol', 'Fwd Header Len', 'Fwd Pkt Len Max', 'ACK Flag Cnt']

print("="*60)
print("DEEP LEARNING - NEURAL NETWORK FOR INTRUSION DETECTION")
print("="*60)

# Load multiple data files
print("\n[1/8] Loading data...")
files = glob.glob('data/*.csv')
dfs = []
for f in files[:3]:  # Load first 3 files
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
print(f"Rows: {len(df):,}")

# Prepare data
le = LabelEncoder()
y = le.fit_transform(df['Label'])
X = df[FEATURES].values
num_classes = len(le.classes_)
print(f"Classes ({num_classes}): {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Model Definition (exactly as per instructions)
class MyNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

model = MyNet(input_size=len(FEATURES), num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"\nModel architecture:\n{model}")

# Training
print("\n[2/8] Training...")
epochs = 50
train_losses, test_accs, test_f1s = [], [], []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).argmax(1).numpy()
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        test_accs.append(acc)
        test_f1s.append(f1)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f} - Acc: {acc:.4f} - F1: {f1:.4f}")

# Final evaluation
print("\n[3/8] Final evaluation...")
model.eval()
with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test)).argmax(1).numpy()

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n{'='*60}")
print("NEURAL NETWORK RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Save classification report
with open("DeepLearning/results/classification_report.txt", "w") as f:
    f.write("NEURAL NETWORK CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score (macro): {f1:.4f}\n\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

print("\n[4/8] Generating graphs...")

# Graph 1: Training Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'b-', linewidth=2)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('DeepLearning/results/01_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# Graph 2: Accuracy & F1 over epochs
plt.figure(figsize=(10, 5))
plt.plot(test_accs, 'g-', label='Accuracy', linewidth=2)
plt.plot(test_f1s, 'r-', label='F1 Score', linewidth=2)
plt.title('Test Metrics Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig('DeepLearning/results/02_metrics_over_epochs.png', dpi=150, bbox_inches='tight')
plt.close()

# Graph 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(max(10, num_classes), max(8, num_classes)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
plt.title('Neural Network - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('DeepLearning/results/03_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Graph 4: Per-class Performance
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
classes = [c for c in le.classes_ if c in report]
f1_scores_list = [report[c]['f1-score'] for c in classes]
precisions = [report[c]['precision'] for c in classes]
recalls = [report[c]['recall'] for c in classes]

if len(classes) > 1:
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(max(12, len(classes)*1.5), 6))
    plt.bar(x - width, precisions, width, label='Precision', color='#3498db')
    plt.bar(x, f1_scores_list, width, label='F1 Score', color='#e74c3c')
    plt.bar(x + width, recalls, width, label='Recall', color='#2ecc71')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('DeepLearning/results/04_per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

# Graph 5: Class Distribution Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y_test, bins=num_classes, color='#3498db', edgecolor='white', alpha=0.7)
axes[0].set_title('True Labels Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Class Index')
axes[0].set_ylabel('Count')

axes[1].hist(y_pred, bins=num_classes, color='#e74c3c', edgecolor='white', alpha=0.7)
axes[1].set_title('Predicted Labels Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Class Index')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('DeepLearning/results/05_label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Graph 6: Prediction Confidence
model.eval()
with torch.no_grad():
    probs = torch.softmax(model(torch.FloatTensor(X_test)), dim=1)
    confidences = probs.max(dim=1).values.numpy()

# Handle case where all confidences are similar
unique_confs = len(np.unique(confidences))
n_bins = min(20, unique_confs)

plt.figure(figsize=(10, 5))
plt.hist(confidences, bins=n_bins, color='#9b59b6', edgecolor='white', alpha=0.7)
plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('DeepLearning/results/06_confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Graph 7: Summary Comparison
models_names = ['Neural Network', 'Ensemble']
accuracies_vals = [acc, 0.98]
f1_vals = [f1, 0.98]

x = np.arange(len(models_names))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, accuracies_vals, width, label='Accuracy', color='#3498db')
bars2 = plt.bar(x + width/2, f1_vals, width, label='F1 Score', color='#e74c3c')

plt.xticks(x, models_names)
plt.ylabel('Score')
plt.title('Model Comparison: Neural Network vs Ensemble', fontsize=14, fontweight='bold')
plt.ylim(0, 1.1)
plt.legend()

for bar in bars1:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center')
for bar in bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('DeepLearning/results/07_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Graph 8: Class-wise F1 Score Bar
plt.figure(figsize=(12, 6))
f1_vals_bar = [report[c]['f1-score'] for c in classes]
colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
plt.bar(range(len(classes)), f1_vals_bar, color=colors)
plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
plt.title('F1 Score per Class', fontsize=14, fontweight='bold')
plt.ylabel('F1 Score')
plt.xlabel('Class')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('DeepLearning/results/08_f1_per_class.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[5/8] Saving model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'label_encoder': le,
    'features': FEATURES
}, 'DeepLearning/results/nn_model.pth')

print("\n[6/8] Summary Statistics...")

summary = f"""
============================================================
DEEP LEARNING SUMMARY
============================================================

MODEL ARCHITECTURE:
- Type: Feed-Forward Neural Network
- Input Layer: {len(FEATURES)} features
- Hidden Layer 1: 64 neurons + ReLU
- Hidden Layer 2: 32 neurons + ReLU
- Output Layer: {num_classes} classes
- Loss Function: CrossEntropyLoss

FEATURES USED:
{chr(10).join(['  - ' + f for f in FEATURES])}

TRAINING CONFIG:
- Epochs: {epochs}
- Batch Size: 512
- Learning Rate: 0.001
- Optimizer: Adam
- Train/Test Split: 80/20

RESULTS:
- Accuracy: {acc:.4f} ({acc*100:.2f}%)
- F1 Score (macro): {f1:.4f}

COMPARISON WITH ENSEMBLE:
- Neural Network Accuracy: {acc*100:.1f}%
- Ensemble Accuracy: ~98%
- Neural Network F1: {f1*100:.1f}%
- Ensemble F1: ~98%

FILES GENERATED:
- nn_model.pth (trained model)
- classification_report.txt
- 01_training_loss.png
- 02_metrics_over_epochs.png
- 03_confusion_matrix.png
- 04_per_class_metrics.png
- 05_label_distribution.png
- 06_confidence_distribution.png
- 07_model_comparison.png
- 08_f1_per_class.png
- summary.txt
"""

with open("DeepLearning/results/summary.txt", "w") as f:
    f.write(summary)

print(summary)

print("\n" + "="*60)
print("COMPLETE! All files saved to: DeepLearning/results/")
print("="*60)