import pandas as pd
import glob

# Ganti dengan path folder tempat file Excel disimpan
folder_path = "E:/STMKG/SKRIPSI/datahujan/AWS/*.xlsx"  # Sesuaikan dengan lokasi file Anda

# Ambil semua file Excel dalam folder
file_list = glob.glob(folder_path)

# Pastikan file diurutkan agar data berurutan dari Januari ke Desember
file_list = sorted(file_list)

# Pilih kolom yang ingin diambil
selected_columns = [
    "tt_air_min", "tt_air_min_flag",  # Temperatur minimum
    "tt_air_max", "tt_air_max_flag",  # Temperatur maksimum
    "tt_air_avg", "tt_air_avg_flag",  # Temperatur rata-rata
    "rh_avg", "rh_avg_flag",          # Kelembaban udara rata-rata
    "ws_avg", "ws_avg_flag",          # Kecepatan angin rata-rata
    "pp_air", "pp_air_flag",          # Tekanan udara
    "sr_avg", "sr_avg_flag",          # Intensitas radiasi matahari rata-rata
    "rr", "rr_flag"                   # Curah hujan
]

df_list = []

# Loop untuk membaca tiap file
for file in file_list:
    df = pd.read_excel(file)
    
    # Pastikan kolom 'Tanggal' ada dalam dataset
    if 'Tanggal' in df.columns:
        df['Tanggal'] = pd.to_datetime(df['Tanggal']).dt.tz_localize(None)  # Ubah ke format datetime tanpa timezone
    else:
        print(f"Peringatan: Tidak ada kolom 'Tanggal' dalam {file}")
        continue
    
    # Pilih hanya kolom yang diperlukan
    df_selected = df[['Tanggal'] + selected_columns]
    df_list.append(df_selected)

# Gabungkan semua data menjadi satu
if df_list:
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Urutkan berdasarkan waktu
    df_combined = df_combined.sort_values(by='Tanggal')
    
    # Simpan hasil ke dalam file Excel baru di lokasi yang ditentukan
    output_path = "E:/STMKG/SKRIPSI/datahujan/Gabungan_AWS_total.xlsx"
    df_combined.to_excel(output_path, index=False)
    print(f"Penggabungan selesai! Data tersimpan dalam '{output_path}'")
else:
    print("Tidak ada data yang berhasil diproses.")
