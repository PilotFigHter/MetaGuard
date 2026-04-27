"""
Generate 5 key EDA graphs for portfolio project
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

FEATURES = ['Init Fwd Win Byts', 'Fwd Seg Size Min', 'Protocol', 'Fwd Header Len', 'Fwd Pkt Len Max', 'ACK Flag Cnt']

print("Loading data...")
files = [
    'data/02-14-2018.csv', 'data/02-15-2018.csv', 'data/02-16-2018.csv',
    'data/02-20-2018.csv', 'data/02-21-2018.csv', 'data/02-22-2018.csv'
]
dfs = []
for f in files:
    try:
        df = pd.read_csv(f, usecols=lambda x: x.strip() in FEATURES + ['Label'], low_memory=False, nrows=20000)
        dfs.append(df)
    except:
        pass

df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()
df['Label'] = df['Label'].astype(str).str.strip()

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[df['Label'].str.lower() != 'label']
print(f"Total rows: {len(df):,}")

OUTPUT_DIR = 'EDA_Graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================
plt.figure(figsize=(14, 7))
class_counts = df['Label'].value_counts()
colors = plt.cm.tab20(np.linspace(0, 1, len(class_counts)))
plt.bar(range(len(class_counts)), class_counts.values, color=colors)
plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right', fontsize=10)
plt.ylabel('Number of Samples', fontsize=12)
plt.xlabel('Attack Type', fontsize=12)
plt.title('1. Class Distribution - Network Intrusion Detection Dataset', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 300, f'{v:,}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("1. Class distribution saved")
print(f"   Classes: {len(class_counts)}")
print(f"   " + ", ".join([f"{k}:{v:,}" for k,v in class_counts.head(5).items()]))

# ============================================================
# 2. FEATURE CORRELATION HEATMAP
# ============================================================
plt.figure(figsize=(10, 8))
corr = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('2. Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("2. Feature correlation saved")

# ============================================================
# 3. FEATURE DISTRIBUTIONS (HISTOGRAMS)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(FEATURES):
    ax = axes[idx]
    data = df[col].dropna()
    data = data[(data > data.quantile(0.01)) & (data < data.quantile(0.99))]
    ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('3. Feature Distributions (Histogram - 1st-99th Percentile)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_feature_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("3. Feature histograms saved")

# ============================================================
# 4. FEATURE BY ATTACK TYPE (BAR CHART)
# ============================================================
top_classes = df['Label'].value_counts().head(6).index
feat_avg = df[df['Label'].isin(top_classes)].groupby('Label')[FEATURES].mean()

x = np.arange(len(FEATURES))
width = 0.12
fig, ax = plt.subplots(figsize=(14, 7))

for i, cls in enumerate(top_classes):
    offset = (i - 2.5) * width
    bars = ax.bar(x + offset, feat_avg.loc[cls].values, width, label=cls[:20], alpha=0.85)

ax.set_ylabel('Mean Value', fontsize=12)
ax.set_title('4. Average Feature Values by Attack Type', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=30, ha='right', fontsize=10)
ax.legend(title='Attack Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_feature_by_attack.png', dpi=150, bbox_inches='tight')
plt.close()
print("4. Feature by attack type saved")

# ============================================================
# 5. ATTACK PATTERN HEATMAP
# ============================================================
from sklearn.preprocessing import StandardScaler

class_means = df.groupby('Label')[FEATURES].mean()
scaler = StandardScaler()
class_means_scaled = pd.DataFrame(
    scaler.fit_transform(class_means),
    index=class_means.index,
    columns=FEATURES
)

top_classes = df['Label'].value_counts().head(12).index
class_means_scaled = class_means_scaled.loc[top_classes]

plt.figure(figsize=(12, 9))
sns.heatmap(class_means_scaled, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('5. Attack Pattern Signatures (Normalized Feature Means)', fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=11)
plt.ylabel('Attack Type', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_attack_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("5. Attack patterns saved")

print(f"\n{'='*50}")
print(f"All 5 graphs saved to {OUTPUT_DIR}/")
print(f"{'='*50}")