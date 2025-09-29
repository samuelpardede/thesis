import pandas as pd
import numpy as np

# Load data
df = pd.read_excel("E:/STMKG/SKRIPSI/datahujan/ME_48/me48_3jam.xlsx")

# Pastikan kolom waktu bertipe datetime
datetime_column = 'DATA_TIMESTAMP'
df[datetime_column] = pd.to_datetime(df[datetime_column])
df.set_index(datetime_column, inplace=True)

# Copy data untuk pembersihan
df_cleaned = df.copy()

# Tentukan kolom kategorikal
categorical_cols = [
    'CLOUD_LOW_TYPE_CL',
    'CLOUD_LOW_MED_AMT_OKTAS',
    'CLOUD_MED_TYPE_CM',
    'CLOUD_HIGH_TYPE_CH',
    'CLOUD_COVER_OKTAS_M',
    'LAND_COND',
    'PRESENT_WEATHER_WW'
]

# Pastikan kolom-kolom kategorikal dalam bentuk string, lalu bersihkan
for col in categorical_cols:
    df_cleaned[col] = df_cleaned[col].astype(str).str.strip().replace('^0+', '', regex=True)
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')  # ubah ke numerik

# Validasi berdasarkan daftar kode WMO yang sah
valid_codes = {
    'CLOUD_LOW_TYPE_CL': list(range(0, 10)),
    'CLOUD_LOW_MED_AMT_OKTAS': list(range(0, 9)),
    'CLOUD_MED_TYPE_CM': list(range(0, 10)),
    'CLOUD_HIGH_TYPE_CH': list(range(0, 10)),
    'CLOUD_COVER_OKTAS_M': list(range(0, 9)),
    'LAND_COND': list(range(0, 3)),
    'PRESENT_WEATHER_WW': list(range(0, 100))
}
for col, valid in valid_codes.items():
    df_cleaned[col] = df_cleaned[col].apply(lambda x: x if x in valid else np.nan)

# Deteksi nilai kategori yang sangat jarang (frekuensi < 5)
for col in categorical_cols:
    rare_vals = df_cleaned[col].value_counts()[df_cleaned[col].value_counts() < 5].index
    df_cleaned[col] = df_cleaned[col].apply(lambda x: np.nan if x in rare_vals else x)

# Optional: isi NaN di kategori dengan ffill (atau mode)
# Imputasi kategori: ffill → bfill → mode
for col in categorical_cols:
    if df_cleaned[col].isna().sum() > 0:
        mode_val = df_cleaned[col].mode().iloc[0]
        df_cleaned[col].fillna(method='ffill', inplace=True)
        df_cleaned[col].fillna(method='bfill', inplace=True)
        df_cleaned[col].fillna(mode_val, inplace=True)


# Tentukan kolom numerik non-RR (hindari kategorikal)
non_rr_cols = df_cleaned.columns.difference(['RR'] + categorical_cols)

# Tangani nilai 0 sebagai NaN jika semua non-RR dan kategori bernilai 0
for i, row in df_cleaned.iterrows():
    non_rr_values = row[non_rr_cols]
    if (non_rr_values.fillna(0) == 0).all():
        df_cleaned.loc[i, non_rr_cols] = np.nan
        if row['RR'] == 0:
            df_cleaned.loc[i, 'RR'] = np.nan

# Interpolasi pertama untuk numerik
df_cleaned[non_rr_cols] = df_cleaned[non_rr_cols].interpolate(method='time')
df_cleaned['RR'] = df_cleaned['RR'].interpolate(method='time')

# Tangani outlier
for col in non_rr_cols:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned[col] = df_cleaned[col].where(
        (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound), np.nan
    )

# Outlier untuk RR (>200 dianggap outlier)
df_cleaned['RR'] = df_cleaned['RR'].where(df_cleaned['RR'] <= 200, np.nan)

# Interpolasi ulang setelah outlier
df_cleaned[non_rr_cols] = df_cleaned[non_rr_cols].interpolate(method='time')
df_cleaned['RR'] = df_cleaned['RR'].interpolate(method='time')

# Kembalikan indeks datetime ke kolom
df_cleaned.reset_index(inplace=True)

# Simpan hasil akhir
df_cleaned.to_excel("E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx", index=False)
