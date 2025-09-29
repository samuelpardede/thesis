import pandas as pd
import numpy as np

# Load data
df = pd.read_excel("E:/STMKG/SKRIPSI/datahujan/fix/Data_AWS_Per_3_Jam.xlsx")

# Pastikan kolom waktu bertipe datetime
datetime_column = 'Jam'  # Ganti jika nama kolom waktu berbeda
df[datetime_column] = pd.to_datetime(df[datetime_column])
df.set_index(datetime_column, inplace=True)

# Copy data untuk pembersihan
df_cleaned = df.copy()

# Tentukan variabel non-RR
non_rr_cols = df.columns.difference(['rr'])

# Langkah 1: Tangani nilai 0 sebagai NaN jika semua variabel (selain RR) bernilai 0
for i, row in df.iterrows():
    non_rr_values = row[non_rr_cols]
    if (non_rr_values.fillna(0) == 0).all():
        # Jika semua variabel selain RR bernilai 0
        df_cleaned.loc[i, non_rr_cols] = np.nan
        if row['rr'] == 0:
            df_cleaned.loc[i, 'rr'] = np.nan

# Langkah 2: Interpolasi awal
df_cleaned = df_cleaned.interpolate(method='time')

# Langkah 3: Tangani outlier
for col in df_cleaned.select_dtypes(include=[np.number]).columns:
    if col == 'rr':
        # Untuk RR, anggap nilai >50 sebagai outlier
        df_cleaned[col] = df_cleaned[col].where(df_cleaned[col] <= 200, np.nan)
    else:
        # Untuk kolom lain, gunakan metode IQR
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned[col] = df_cleaned[col].where(
            (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound), np.nan
        )

# Langkah 4: Interpolasi ulang setelah outlier
df_cleaned = df_cleaned.interpolate(method='time')

# Kembalikan index menjadi kolom biasa jika ingin disimpan
df_cleaned.reset_index(inplace=True)

# Simpan file hasil
df_cleaned.to_excel("E:/STMKG/SKRIPSI/datahujan/fix/Data_AWS_Per_3_Jam_Outlier.xlsx", index=False)
