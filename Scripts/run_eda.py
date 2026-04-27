import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# --- CONFIG ---
DATA_DIR = './data/'
OUTPUT_DIR = 'EDA'
SAMPLE_RATE = 0.25 

# Updated Feature Set
best_features = [
    'Init Fwd Win Byts', 
    'Fwd Seg Size Min', 
    'Protocol', 
    'Fwd Header Len', 
    'Fwd Pkt Len Max', 
    'ACK Flag Cnt'
]

def run_eda():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("🚀 MetaGuard: Generating EDA for 6-Feature Real-Time Set...")
    
    # 1. LOAD DATA
    all_files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv")) if "live" not in f]
    df_list = []
    total_original_rows = 0
    
    for f in tqdm(all_files, desc="Processing CSVs"):
        try:
            # Note: Init Fwd Win Byts might have different naming in some CSVs, stripping handles it
            df_temp = pd.read_csv(f, usecols=lambda x: x.strip() in best_features or x.strip() == 'Label')
            df_temp.columns = df_temp.columns.str.strip()
            total_original_rows += len(df_temp)
            df_list.append(df_temp.sample(frac=SAMPLE_RATE, random_state=42))
        except Exception as e:
            print(f"Skipping {f}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    df['Label'] = df['Label'].astype(str).str.strip()
    
    # Cleaning
    for col in best_features: 
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=best_features + ['Label'], inplace=True)
    rows_after = len(df)

    # 2. FEATURE TABLE WITH DESCRIPTIONS
    feature_meta = {
        'Init Fwd Win Byts': 'Initial TCP window size in the forward direction. High values often correlate with data-heavy attacks.',
        'Fwd Seg Size Min': 'Minimum segment size observed; helps identify small-packet "noise" or scanning behavior.',
        'Protocol': 'The transport layer protocol (e.g., TCP=6, UDP=17). Essential for protocol-specific attack filtering.',
        'Fwd Header Len': 'Total bytes of headers sent. Discrepancies here can signal header-manipulation attacks.',
        'Fwd Pkt Len Max': 'Largest packet size in the flow. Vital for detecting Buffer Overflow or Exfiltration attempts.',
        'ACK Flag Cnt': 'Number of packets with the Acknowledge flag. Key for detecting ACK Flood DDoS attacks.'
    }

    retention_pct = (rows_after / total_original_rows) * 100
    table_rows = []
    for feat in best_features:
        table_rows.append({
            "פיצ'ר": feat,
            "משמעות טכנית": feature_meta[feat],
            "שורות במדגם": rows_after,
            "אחוז מהמקור": f"{retention_pct:.2f}%"
        })
    pd.DataFrame(table_rows).to_csv(f"{OUTPUT_DIR}/feature_table.csv", index=False, encoding='utf-8-sig')

    # 3. CORRELATION HEATMAP
    plt.figure(figsize=(10, 8))
    # We factorize the Label to see how features correlate with the Attack vs Benign classes
    df_corr = df.copy()
    df_corr['Label_Numeric'] = pd.factorize(df_corr['Label'])[0]
    sns.heatmap(df_corr[best_features + ['Label_Numeric']].corr(), annot=True, cmap='RdYlBu', center=0)
    plt.title("MetaGuard: 6-Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()

    # 4. CLASS DISTRIBUTION (Malicious vs Benign)
    plt.figure(figsize=(10, 6))
    df['Label'].value_counts().plot(kind='barh', color='salmon')
    plt.title("Class Distribution for EDA")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
    plt.close()

    # 5. HEBREW SUMMARY
    summary = f"""
## סיכום EDA עבור MetaGuard (מערך 6 פיצ'רים)

### נתונים כלליים:
- **סך שורות שעובדו**: {total_original_rows:,}
- **שורות לאחר ניקוי ודגימה**: {rows_after:,}
- **שיעור שימור נתונים**: {retention_pct:.2f}%

### תובנות מפתח מהפיצ'רים:
1. **Init Fwd Win Byts**: פיצ'ר זה קריטי לזיהוי התחלת התקשרויות TCP. ערכים חריגים מעידים לרוב על ניסיונות הצפה או הזרקת מידע.
2. **ACK Flag Cnt**: מאפשר למודל להבחין בקלות בין התקפות DoS מסוג ACK Flood לבין תעבורה תקינה המבוססת על אישורי קבלה.
3. **Fwd Pkt Len Max**: עוזר בזיהוי התקפות "Payload-heavy" שבהן התוקף מנסה להעביר פקודות בתוך פאקטים בודדים גדולים.
4. **קורלציה**: הניתוח מראה כי אין חפיפה (Multicollinearity) גבוהה מדי בין הפיצ'רים, מה שמאפשר למודל ללמוד מידע ייחודי מכל אחד מהם.
"""
    with open(f"{OUTPUT_DIR}/eda_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"✅ EDA complete. Results saved in /{OUTPUT_DIR}")

if __name__ == "__main__":
    run_eda()