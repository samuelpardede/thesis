import pandas as pd

# Load data
df = pd.read_excel("E:/STMKG/SKRIPSI/datahujan/ME_48/me48fix.xlsx")

# Bersihkan kolom waktu dan konversi ke datetime
df["DATA_TIMESTAMP"] = df["DATA_TIMESTAMP"].str.split(",").str[0]
df["DATA_TIMESTAMP"] = pd.to_datetime(df["DATA_TIMESTAMP"])

# Set kolom waktu sebagai index
df.set_index("DATA_TIMESTAMP", inplace=True)

# Kolom kategori: gunakan modus saat agregasi
categorical_cols = [
    "CLOUD_LOW_TYPE_CL",
    "CLOUD_LOW_MED_AMT_OKTAS",
    "CLOUD_MED_TYPE_CM",
    "CLOUD_HIGH_TYPE_CH",
    "CLOUD_COVER_OKTAS_M",
    "LAND_COND",
    "RR"
]

# Tentukan metode agregasi
agg_dict = {}
for col in df.columns:
    if col in categorical_cols:
        agg_dict[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    elif "RR" in col:
        agg_dict[col] = "sum"  # Total curah hujan per 3 jam
    else:
        agg_dict[col] = "mean"  # Default: rata-rata

# Agregasi per 3 jam
df_3jam = df.resample("3H").agg(agg_dict).reset_index()

# Simpan ke file baru
df_3jam.to_excel("E:/STMKG/SKRIPSI/datahujan/ME_48/me48_3jam.xlsx", index=False)

print("✅ Data berhasil diagregasi per 3 jam dan disimpan sebagai 'data_me48_per_3jam.xlsx'")
