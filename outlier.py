import pandas as pd
import numpy as np

# === STEP 1: Load Data
file_path = "E:/STMKG/SKRIPSI/datahujan/Data_AWS_Per_Jam.xlsx"
df = pd.read_excel(file_path)

# Buat salinan untuk proses
df_cleaned = df.copy()

# === STEP 2: Identifikasi kolom selain 'rr'
columns_ex_rr = [col for col in df_cleaned.columns if col.lower() != 'rr']

# === STEP 3: Logika pembersihan nilai nol

# 3.1 Ganti 0 jadi NaN untuk semua kolom KECUALI 'rr'
df_cleaned[columns_ex_rr] = df_cleaned[columns_ex_rr].replace(0, np.nan)

# 3.2 Jika semua kolom lain NaN → anggap baris itu kosong → ganti rr jadi NaN juga
zero_rr_condition = df_cleaned['rr'] == 0
other_vars_nan = df_cleaned[columns_ex_rr].isna().all(axis=1)
df_cleaned.loc[zero_rr_condition & other_vars_nan, 'rr'] = np.nan

# === STEP 4: Deteksi outlier dengan metode IQR
df_iqr = df_cleaned.copy()
numeric_cols = df_iqr.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    Q1 = df_iqr[col].quantile(0.25)
    Q3 = df_iqr[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_iqr[col] = df_iqr[col].mask((df_iqr[col] < lower) | (df_iqr[col] > upper), np.nan)

# === STEP 5: Simpan hasil
output_path = "E:/STMKG/SKRIPSI/output_file/Data_AWS_Bersih_IQR_Outlier_FINAL.xlsx"
df_iqr.to_excel(output_path, index=False)

print("✅ Output berhasil disimpan tanpa kesalahan nilai rr:")
print(output_path)
