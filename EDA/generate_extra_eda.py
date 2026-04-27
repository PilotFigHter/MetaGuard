"""
Generate additional EDA graphs including:
1. Class distribution AFTER RandomOverSampler
2. More interesting visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

FEATURES = ['Init Fwd Win Byts', 'Fwd Seg Size Min', 'Protocol', 'Fwd Header Len', 'Fwd Pkt Len Max', 'ACK Flag Cnt']

OUTPUT_DIR = 'EDA_Graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
df = df.dropna(subset=FEATURES + ['Label'])
print(f"Total rows: {len(df):,}")

class_counts_before = df['Label'].value_counts()
print("\n=== BEFORE OVERSAMPLING ===")
for label, count in class_counts_before.items():
    print(f"  {label}: {count:,}")

# ============================================================
# 1. CLASS DISTRIBUTION AFTER RANDOM OVERSAMPLER
# ============================================================
print("\nApplying RandomOverSampler...")
X = df[FEATURES].values
y = df['Label'].values

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

df_resampled = pd.DataFrame(X_resampled, columns=FEATURES)
df_resampled['Label'] = y_resampled

class_counts_after = pd.Series(y_resampled).value_counts()
print("\n=== AFTER OVERSAMPLING ===")
for label, count in class_counts_after.items():
    print(f"  {label}: {count:,}")

# Plot AFTER
plt.figure(figsize=(14, 7))
colors = plt.cm.tab20(np.linspace(0, 1, len(class_counts_after)))
bars = plt.bar(range(len(class_counts_after)), class_counts_after.values, color=colors)
plt.xticks(range(len(class_counts_after)), class_counts_after.index, rotation=45, ha='right', fontsize=10)
plt.ylabel('Number of Samples', fontsize=12)
plt.xlabel('Attack Type', fontsize=12)
plt.title('Class Distribution AFTER RandomOverSampler\n(Balanced Dataset for Training)', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(class_counts_after.values):
    plt.text(i, v + 200, f'{v:,}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_class_after_oversampling.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n1. Class distribution AFTER oversampling saved")

# ============================================================
# 2. BEFORE vs AFTER COMPARISON
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before
ax1 = axes[0]
colors1 = plt.cm.Reds(np.linspace(0.3, 0.9, len(class_counts_before)))
ax1.barh(range(len(class_counts_before)), class_counts_before.values, color=colors1)
ax1.set_yticks(range(len(class_counts_before)))
ax1.set_yticklabels(class_counts_before.index, fontsize=9)
ax1.set_xlabel('Number of Samples')
ax1.set_title('BEFORE: Imbalanced Classes', fontsize=12, fontweight='bold')
ax1.set_xlim(0, max(class_counts_before.max() * 1.1, class_counts_after.max() * 1.1))
for i, v in enumerate(class_counts_before.values):
    ax1.text(v + 100, i, f'{v:,}', va='center', fontsize=8)

# After
ax2 = axes[1]
colors2 = plt.cm.Greens(np.linspace(0.3, 0.9, len(class_counts_after)))
ax2.barh(range(len(class_counts_after)), class_counts_after.values, color=colors2)
ax2.set_yticks(range(len(class_counts_after)))
ax2.set_yticklabels(class_counts_after.index, fontsize=9)
ax2.set_xlabel('Number of Samples')
ax2.set_title('AFTER: Balanced with RandomOverSampler', fontsize=12, fontweight='bold')
ax2.set_xlim(0, max(class_counts_before.max() * 1.1, class_counts_after.max() * 1.1))
for i, v in enumerate(class_counts_after.values):
    ax2.text(v + 100, i, f'{v:,}', va='center', fontsize=8)

plt.suptitle('Class Imbalance: Before vs After Oversampling', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_before_after_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("2. Before/After comparison saved")

# ============================================================
# 3. PROTOCOL DISTRIBUTION PIE CHART
# ============================================================
protocol_counts = df['Protocol'].value_counts()
plt.figure(figsize=(10, 8))
colors = plt.cm.Set2(np.linspace(0, 1, len(protocol_counts)))
wedges, texts, autotexts = plt.pie(protocol_counts.values, labels=protocol_counts.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90,
                                   explode=[0.02] * len(protocol_counts))
plt.setp(autotexts, size=11, weight='bold')
plt.setp(texts, size=12)
plt.title('Protocol Distribution\n(TCP vs UDP vs ICMP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_protocol_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("3. Protocol distribution saved")

# ============================================================
# 4. FEATURE BOXPLOTS BY ATTACK TYPE
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

top_classes = df['Label'].value_counts().head(6).index
df_top = df[df['Label'].isin(top_classes)]

for idx, col in enumerate(FEATURES):
    ax = axes[idx]
    top5 = df['Label'].value_counts().head(5).index
    df_top5 = df[df['Label'].isin(top5)]
    
    data_to_plot = []
    labels = []
    for cls in top_classes:
        data = df_top[df_top['Label'] == cls][col].dropna()
        if len(data) > 0:
            q1, q99 = np.percentile(data, [1, 99])
            data_filtered = data[(data >= q1) & (data <= q99)]
            if len(data_filtered) > 0:
                data_to_plot.append(data_filtered.values)
                labels.append(cls[:12])
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Feature Distributions by Attack Type (Boxplots)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_feature_boxplots_attack.png', dpi=150, bbox_inches='tight')
plt.close()
print("4. Feature boxplots saved")

# ============================================================
# 5. ATTACK SEVERITY Pie (grouped by category)
# ============================================================
attack_mapping = {
    'Benign': 'Benign',
    'Bot': 'Bot',
    'BruteForce': 'BruteForce', 
    'DoS': 'DoS',
    'Dwarf': 'DoS',
    'GuessPassword': 'R2L',
    'Probe': 'Probe',
    'Scan': 'Probe',
    'Shellcode': 'U2R',
    'XSS': 'Web Attack',
    'SqlInjection': 'Web Attack'
}

df['Attack_Category'] = df['Label'].apply(lambda x: attack_mapping.get(x, 'Other'))
category_counts = df['Attack_Category'].value_counts()

plt.figure(figsize=(10, 8))
colors = plt.cm.Paired(np.linspace(0, 1, len(category_counts)))
wedges, texts, autotexts = plt.pie(category_counts.values, labels=category_counts.index,
                                   autopct='%1.1f%%', colors=colors, startangle=90,
                                   explode=[0.03] * len(category_counts))
plt.setp(autotexts, size=11, weight='bold')
plt.setp(texts, size=12)
plt.title('Attack Category Distribution\n(Grouped by Attack Type)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_attack_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print("5. Attack categories saved")

# ============================================================
# 6. FEATURE VIOLIN PLOTS (showing distribution shape)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

top5 = df['Label'].value_counts().head(5).index
df_top5 = df[df['Label'].isin(top5)]

for idx, col in enumerate(FEATURES):
    ax = axes[idx]
    sns.violinplot(data=df_top5, x='Label', y=col, ax=ax, palette='Set2', inner='box')
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Feature Distribution Shapes by Attack Type (Violin Plots)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_feature_violin.png', dpi=150, bbox_inches='tight')
plt.close()
print("6. Violin plots saved")

# ============================================================
# 7. PACKET SIZE DISTRIBUTION (Fwd Pkt Len Max)
# ============================================================
plt.figure(figsize=(12, 6))
benign = df[df['Label'] == 'Benign']['Fwd Pkt Len Max'].dropna()
attacks = df[df['Label'] != 'Benign']['Fwd Pkt Len Max'].dropna()

benign = benign[(benign > benign.quantile(0.01)) & (benign < benign.quantile(0.99))]
attacks = attacks[(attacks > attacks.quantile(0.01)) & (attacks < attacks.quantile(0.99))]

plt.hist(benign, bins=50, alpha=0.6, label='Benign', color='green', density=True)
plt.hist(attacks, bins=50, alpha=0.6, label='Attack', color='red', density=True)
plt.xlabel('Forward Packet Length Max (bytes)')
plt.ylabel('Density')
plt.title('Packet Size Distribution: Benign vs Attacks', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_pkt_size_benign_vs_attack.png', dpi=150, bbox_inches='tight')
plt.close()
print("7. Packet size distribution saved")

print(f"\n{'='*50}")
print(f"All 7 new graphs saved to {OUTPUT_DIR}/")
print(f"{'='*50}")