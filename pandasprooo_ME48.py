import pandas as pd

# Load data
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
df = pd.read_excel(file_path)

# Generate descriptive statistics
desc_stats = df.describe(include='all')

# Simpan hasilnya ke dalam Excel
output_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Deskripsi_Data_ME48_Bersih_Final.xlsx"
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    desc_stats.to_excel(writer, sheet_name='Descriptive Statistics')
    
print(f"Deskripsi statistik berhasil disimpan di: {output_path}")
