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
import joblib

os.makedirs("DeepLearning/results", exist_ok=True)

FEATURES = ['Init Fwd Win Byts', 'Fwd Seg Size Min', 'Protocol', 'Fwd Header Len', 'Fwd Pkt Len Max', 'ACK Flag Cnt']

print("="*60)
print("DEEP LEARNING - NEURAL NETWORK FOR INTRUSION DETECTION")
print("="*60)

# Load data - sample from each file for memory efficiency
print("\n[1/8] Loading data...")
files = glob.glob('data/*.csv')
print(f"Files found: {len(files)}")

dfs = []
ROWS_PER_FILE = 40000  # Balanced for memory + class coverage
for f in files:
    try:
        df = pd.read_csv(f, usecols=lambda x: x.strip() in FEATURES + ['Label'], 
                         low_memory=False, nrows=ROWS_PER_FILE)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    except:
        pass

print(f"Loaded {len(dfs)} files ({ROWS_PER_FILE} rows each)")

df = pd.concat(dfs, ignore_index=True)
df['Label'] = df['Label'].astype(str).str.strip()

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df = df[df['Label'].str.lower() != 'label']
print(f"Rows: {len(df):,}")

# Use existing label encoder with all 15 classes
le = joblib.load('models/realtime_label_encoder.pkl')
known_classes = le.classes_
print(f"Using {len(known_classes)} classes from realtime model: {list(known_classes)}")

# Only keep rows with known labels
df = df[df['Label'].isin(known_classes)]
print(f"After filtering to known classes: {len(df):,} rows")

y = le.transform(df['Label'])
X = df[FEATURES].values
num_classes = len(le.classes_)
print(f"Classes: {num_classes}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Model
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

print(f"\nModel: {len(FEATURES)} inputs -> 64 -> 32 -> {num_classes} outputs")

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
    
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).argmax(1).numpy()
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        test_accs.append(acc)
        test_f1s.append(f1)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f} - Acc: {acc:.4f} - F1: {f1:.4f}")

# Results
print("\n[3/8] Final evaluation...")
model.eval()
with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test)).argmax(1).numpy()

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

present_labels = np.unique(np.concatenate([y_test, y_pred]))
present_names = [le.classes_[i] for i in present_labels]

print(f"\n{'='*60}")
print("NEURAL NETWORK RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred, labels=present_labels, target_names=present_names, zero_division=0)}")

with open("DeepLearning/results/classification_report.txt", "w") as f:
    f.write(f"NEURAL NETWORK RESULTS\nAccuracy: {acc:.4f}\nF1: {f1:.4f}\n\n")
    f.write(classification_report(y_test, y_pred, labels=present_labels, target_names=present_names, zero_division=0))

# Graphs
print("\n[4/8] Generating graphs...")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'b-', linewidth=2)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('DeepLearning/results/01_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()

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

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(le.classes_), yticklabels=list(le.classes_))
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('DeepLearning/results/03_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Model comparison
plt.figure(figsize=(10, 6))
models_names = ['Neural Network', 'Ensemble']
accuracies_vals = [acc, 0.98]
f1_vals = [f1, 0.98]
x = np.arange(len(models_names))
plt.bar(x - 0.175, accuracies_vals, 0.35, label='Accuracy', color='#3498db')
plt.bar(x + 0.175, f1_vals, 0.35, label='F1 Score', color='#e74c3c')
plt.xticks(x, models_names)
plt.ylim(0, 1.1)
plt.title('Model Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
for i, (a, f) in enumerate(zip(accuracies_vals, f1_vals)):
    plt.text(i-0.175, a+0.02, f'{a:.3f}', ha='center')
    plt.text(i+0.175, f+0.02, f'{f:.3f}', ha='center')
plt.savefig('DeepLearning/results/04_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Save model
print("\n[5/8] Saving model...")
torch.save({'model': model.state_dict(), 'scaler': scaler, 'le': le}, 'DeepLearning/results/nn_model.pth')

# Summary
summary = f"""
{'='*60}
DEEP LEARNING SUMMARY
{'='*60}

MODEL: MyNet (6 -> 64 -> 32 -> {num_classes})
Features: {FEATURES}
Epochs: {epochs}
Accuracy: {acc:.4f}
F1 Score: {f1:.4f}

Comparison:
- NN: Acc={acc:.3f}, F1={f1:.3f}
- Ensemble: Acc=0.980, F1=0.980
"""
print(summary)
with open("DeepLearning/results/summary.txt", "w") as f:
    f.write(summary)

print(f"\n{'='*60}")
print("COMPLETE! Files in DeepLearning/results/")
print("="*60)