import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from tqdm import tqdm

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = "EDA_Graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("METAWATCH - EDA ANALYSIS & VISUALIZATION")
print("="*60)

# Load all data files
DATA_DIR = "data"
files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"\nLoading {len(files)} files...")

dfs = []
for f in tqdm(files, desc="Reading CSVs"):
    try:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {f}: {e}")

df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df):,}")

# Clean data
df['Label'] = df['Label'].astype(str).str.strip()
df = df[df['Label'].str.lower() != 'label']

# Our 6 features
FEATURES = ['Init Fwd Win Byts', 'Fwd Seg Size Min', 'Protocol', 
             'Fwd Header Len', 'Fwd Pkt Len Max', 'ACK Flag Cnt']

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\n" + "="*60)
print("GENERATING GRAPHS...")
print("="*60)

# =============================================================================
# 1. CLASS DISTRIBUTION
# =============================================================================
print("\n[1/12] Class Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pie chart
label_counts = df['Label'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
axes[0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90, pctdistance=0.75)
axes[0].set_title('Class Distribution (Pie)', fontsize=14, fontweight='bold')

# Bar chart (top 10)
top_10 = label_counts.head(10)
bars = axes[1].barh(range(len(top_10)), top_10.values, color=sns.color_palette("husl", len(top_10)))
axes[1].set_yticks(range(len(top_10)))
axes[1].set_yticklabels(top_10.index)
axes[1].set_xlabel('Count')
axes[1].set_title('Top 10 Classes (Bar)', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()

for i, v in enumerate(top_10.values):
    axes[1].text(v + v*0.01, i, f'{v:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_class_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 2. FEATURE DISTRIBUTION (Boxplots)
# =============================================================================
print("[2/12] Feature Distributions...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(FEATURES):
    data = df[feat].dropna()
    # Sample for visualization
    sample = data.sample(min(10000, len(data)), random_state=42)
    
    bp = axes[i].boxplot(sample, patch_artist=True)
    bp['boxes'][0].set_facecolor(sns.color_palette("husl", 6)[i])
    axes[i].set_title(feat, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Value')

plt.suptitle('Feature Distributions (Boxplots)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_feature_boxplots.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 3. FEATURE DISTRIBUTION (Histograms)
# =============================================================================
print("[3/12] Feature Histograms...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(FEATURES):
    data = df[feat].dropna()
    sample = data.sample(min(50000, len(data)), random_state=42)
    axes[i].hist(sample, bins=50, color=sns.color_palette("husl", 6)[i], edgecolor='white', alpha=0.8)
    axes[i].set_title(feat, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

plt.suptitle('Feature Distributions (Histograms)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_feature_histograms.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. CORRELATION MATRIX
# =============================================================================
print("[4/12] Feature Correlation...")

corr = df[FEATURES].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_correlation_matrix.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 5. MISSING VALUES HEATMAP
# =============================================================================
print("[5/12] Missing Values...")

# Sample for visualization
sample_df = df.sample(min(50000, len(df)), random_state=42)

# Calculate missing per feature
missing_pct = (sample_df.isnull().sum() / len(sample_df) * 100).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
top_missing = missing_pct[missing_pct > 0].head(15)
axes[0].barh(range(len(top_missing)), top_missing.values, color='#e74c3c')
axes[0].set_yticks(range(len(top_missing)))
axes[0].set_yticklabels(top_missing.index)
axes[0].set_xlabel('Missing Percentage (%)')
axes[0].set_title('Top 15 Features with Missing Values', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()

# Summary text
axes[1].text(0.5, 0.7, f'Total Rows: {len(df):,}', fontsize=14, ha='center', transform=axes[1].transAxes)
axes[1].text(0.5, 0.5, f'Total Missing: {df.isnull().sum().sum():,}', fontsize=14, ha='center', transform=axes[1].transAxes)
axes[1].text(0.5, 0.3, f'Missing %: {df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.4f}%', 
             fontsize=14, ha='center', transform=axes[1].transAxes)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].axis('off')
axes[1].set_title('Missing Data Summary', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_missing_values.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. CLASS DISTRIBUTION BY PROTOCOL
# =============================================================================
print("[6/12] Protocol Analysis...")

if 'Protocol' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    proto_label = df.groupby(['Protocol', 'Label']).size().unstack(fill_value=0)
    proto_label_pct = proto_label.div(proto_label.sum(axis=1), axis=0) * 100
    
    proto_label_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
    ax.set_xlabel('Protocol')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Class Distribution by Protocol', fontsize=14, fontweight='bold')
    ax.legend(title='Attack Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_protocol_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
# 7. ATTACK TYPES OVERVIEW
# =============================================================================
print("[7/12] Attack Types Overview...")

# Categorize attacks
def categorize_attack(label):
    if label == 'Benign':
        return 'Benign'
    elif any(x in label for x in ['DDoS', 'DDOS']):
        return 'DDoS'
    elif any(x in label for x in ['DoS', 'Hulk', 'GoldenEye', 'Slow']):
        return 'DoS'
    elif any(x in label for x in ['Brute', 'Force']):
        return 'Brute Force'
    elif label in ['Bot', 'Infilteration', 'SQL Injection', 'Web Attack']:
        return 'Other'
    return 'Other'

df['Attack_Category'] = df['Label'].apply(categorize_attack)
category_counts = df['Attack_Category'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors_cat = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
axes[0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
           colors=colors_cat[:len(category_counts)], startangle=90)
axes[0].set_title('Attack Categories Distribution', fontsize=14, fontweight='bold')

bars = axes[1].bar(category_counts.index, category_counts.values, color=colors_cat[:len(category_counts)])
axes[1].set_ylabel('Count')
axes[1].set_title('Attack Categories (Bar)', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)

for bar, count in zip(bars, category_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                 f'{count:,}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_attack_categories.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 8. FEATURE VS ATTACK (Violin Plots)
# =============================================================================
print("[8/12] Feature vs Attack Type...")

# Sample for visualization
sample = df.sample(min(20000, len(df)), random_state=42)
sample_melted = sample.melt(id_vars=['Attack_Category'], value_vars=FEATURES[:3],
                             var_name='Feature', value_name='Value')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feat in enumerate(FEATURES[:3]):
    feat_data = sample[[feat, 'Attack_Category']].dropna()
    categories = feat_data['Attack_Category'].unique()[:5]  # Top 5
    filtered = feat_data[feat_data['Attack_Category'].isin(categories)]
    
    vp = sns.violinplot(data=filtered, x='Attack_Category', y=feat, ax=axes[i], 
                       palette='Set2', inner='box')
    axes[i].set_title(feat, fontsize=12, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=20)

plt.suptitle('Feature Distribution by Attack Type (Top 3 Features)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_feature_vs_attack.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 9. DATA QUALITY OVERVIEW
# =============================================================================
print("[9/12] Data Quality Overview...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total data points
total_cells = len(df) * len(df.columns)
valid_cells = total_cells - df.isnull().sum().sum() - np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
missing_cells = df.isnull().sum().sum()
inf_cells = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

quality_data = ['Valid\nData', 'Missing\n(NaN)', 'Infinity\nValues']
quality_counts = [valid_cells, missing_cells, inf_cells]
quality_colors = ['#27ae60', '#e74c3c', '#9b59b6']

axes[0, 0].pie(quality_counts, labels=quality_data, autopct='%1.2f%%',
              colors=quality_colors, startangle=90, explode=(0, 0.05, 0.05))
axes[0, 0].set_title('Data Quality Overview', fontsize=12, fontweight='bold')

# Rows and columns
axes[0, 1].bar(['Rows', 'Columns'], [len(df), len(df.columns)], color='#3498db', edgecolor='white')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Dataset Size', fontsize=12, fontweight='bold')
for i, v in enumerate([len(df), len(df.columns)]):
    axes[0, 1].text(i, v + v*0.02, f'{v:,}', ha='center', fontsize=12, fontweight='bold')

# Class balance
class_balance = label_counts / len(df) * 100
axes[1, 0].hist(class_balance.values, bins=20, color='#3498db', edgecolor='white')
axes[1, 0].axvline(class_balance.mean(), color='red', linestyle='--', label=f'Mean: {class_balance.mean():.2f}%')
axes[1, 0].set_xlabel('Percentage (%)')
axes[1, 0].set_ylabel('Number of Classes')
axes[1, 0].set_title('Class Balance Distribution', fontsize=12, fontweight='bold')
axes[1, 0].legend()

# Features info
top_features = df[FEATURES].notna().sum().sort_values() / len(df) * 100
axes[1, 1].barh(range(len(top_features)), top_features.values, color='#1abc9c', edgecolor='white')
axes[1, 1].set_yticks(range(len(top_features)))
axes[1, 1].set_yticklabels(top_features.index)
axes[1, 1].set_xlabel('Completeness (%)')
axes[1, 1].set_title('Feature Completeness (Our 6 Features)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlim(95, 100.5)

plt.suptitle('Data Quality Summary', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_data_quality_overview.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 10. FEATURE IMPORTANCE (After Training)
# =============================================================================
print("[10/12] Feature Statistics...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(FEATURES):
    data = df[feat].dropna()
    sample = data.sample(min(50000, len(data)), random_state=42)
    
    # KDE plot
    sns.kdeplot(sample, ax=axes[i], fill=True, color=sns.color_palette("husl", 6)[i], alpha=0.7)
    axes[i].axvline(sample.mean(), color='red', linestyle='--', label=f'Mean: {sample.mean():.2f}')
    axes[i].axvline(sample.median(), color='green', linestyle='--', label=f'Median: {sample.median():.2f}')
    axes[i].set_title(feat, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].legend(fontsize=8)

plt.suptitle('Feature Density Distributions (KDE)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_feature_density.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 11. ATTACK PATTERNS (Packet Size Analysis)
# =============================================================================
print("[11/12] Attack Patterns...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plot of packet sizes by attack
attack_sample = df.sample(min(50000, len(df)), random_state=42)
top_attacks = attack_sample['Attack_Category'].value_counts().head(6).index

plot_data = attack_sample[attack_sample['Attack_Category'].isin(top_attacks)]

sns.boxplot(data=plot_data, x='Attack_Category', y='Fwd Pkt Len Max', ax=axes[0], palette='Set2')
axes[0].set_title('Forward Packet Size by Attack Type', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)

# TCP Window analysis
sns.boxplot(data=plot_data, x='Attack_Category', y='Init Fwd Win Byts', ax=axes[1], palette='Set2')
axes[1].set_title('TCP Window Size by Attack Type', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "11_attack_patterns.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 12. SUMMARY STATISTICS TABLE
# =============================================================================
print("[12/12] Summary Statistics...")

stats_data = []
for feat in FEATURES:
    data = df[feat].dropna()
    stats_data.append({
        'Feature': feat,
        'Count': len(data),
        'Missing%': round((1 - len(data)/len(df))*100, 3),
        'Mean': round(data.mean(), 2),
        'Std': round(data.std(), 2),
        'Min': round(data.min(), 2),
        '25%': round(data.quantile(0.25), 2),
        '50%': round(data.quantile(0.5), 2),
        '75%': round(data.quantile(0.75), 2),
        'Max': round(data.max(), 2)
    })

stats_df = pd.DataFrame(stats_data)
stats_df.to_csv(os.path.join(OUTPUT_DIR, "feature_statistics.csv"), index=False)

# Plot as table image
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=stats_df.values,
                colLabels=stats_df.columns,
                cellLoc='center',
                loc='center',
                colColours=['#3498db']*len(stats_df.columns))

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.title('Feature Statistics Summary', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(os.path.join(OUTPUT_DIR, "12_feature_statistics_table.png"), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# COMPLETE
# =============================================================================
print("\n" + "="*60)
print("EDA COMPLETE!")
print("="*60)
print(f"\nGenerated {len(os.listdir(OUTPUT_DIR))} files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")

print(f"\nOutput folder: {os.path.abspath(OUTPUT_DIR)}")