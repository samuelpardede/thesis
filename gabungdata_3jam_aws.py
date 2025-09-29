import pandas as pd

# 1. Load data
file_path = r"E:\STMKG\SKRIPSI\datahujan\Data_AWS_Per_Jam.xlsx"
df = pd.read_excel(file_path)

# 2. Pastikan kolom waktu dalam format datetime
# Gantilah 'tanggal' atau 'waktu' dengan nama kolom waktu di file Anda
# Misalnya jika nama kolom waktu adalah 'datetime':
df['Jam'] = pd.to_datetime(df['Jam'])  # Ubah sesuai nama kolom Anda

# 3. Set kolom waktu sebagai index
df.set_index('Jam', inplace=True)

# 4. Resample data per 3 jam
# Kolom non-rainfall dirata-rata, curah hujan dijumlahkan
# Sesuaikan nama kolom sesuai yang tersedia di data
aggregations = {
    'tt_air_avg': 'mean',
    'rh_avg': 'mean',
    'ws_avg': 'mean',
    'pp_air': 'mean',
    'rr': 'sum'  # Curah hujan dijumlahkan selama 3 jam
}

df_3jam = df.resample('3H').agg(aggregations)

# 5. Reset index jika ingin simpan ke file
df_3jam.reset_index(inplace=True)

# 6. Simpan ke file baru (opsional)
output_path = r"E:\STMKG\SKRIPSI\datahujan\fix\Data_AWS_Per_3_Jam.xlsx"
df_3jam.to_excel(output_path, index=False)

print("Transformasi ke data per 3 jam selesai dan disimpan di:")
print(output_path)
