import pandas as pd
import numpy as np

# 1. Baca data
df = pd.read_excel("E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx")

# 2. Inisialisasi list untuk menyimpan info
summary = []

# 3. Loop tiap kolom
for col in df.columns:
    if col == 'Tanggal':
        continue  # lewati kolom tanggal

    nunique = df[col].nunique()
    dtype = df[col].dtype
    nulls = df[col].isnull().sum()
    null_percent = nulls / len(df) * 100
    min_val = df[col].min()
    try:
        skewness = df[col].skew()
    except:
        skewness = np.nan

    # 4. Buat saran
    suggestion = []

    if nulls > 0:
        suggestion.append("fill missing")
    if pd.api.types.is_numeric_dtype(df[col]):
        if abs(skewness) > 1:
            suggestion.append("skewed: consider transform")
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            suggestion.append("possible outliers")
    else:
        suggestion.append("check data type")

    summary.append({
        "Column": col,
        "Nuniques": nunique,
        "Dtype": dtype,
        "Nulls": nulls,
        "NullPercent": round(null_percent, 2),
        "Skewness": round(skewness, 2),
        "MinValue": min_val,
        "Suggestions": "; ".join(suggestion)
    })

# 5. Convert ke dataframe
summary_df = pd.DataFrame(summary)

# 6. Tampilkan hasil
print(summary_df)

# 7. Simpan ke Excel (opsional)
summary_df.to_excel("E:/STMKG/SKRIPSI/datahujan/ME_48/summary_me48_cleaning_suggestions_3Jam_After.xlsx", index=False)
