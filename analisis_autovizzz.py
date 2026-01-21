from autoviz.AutoViz_Class import AutoViz_Class
import pandas as pd

# 3️⃣ Load Data
file_path = "E:/STMKG/SKRIPSI/datahujan/Gabungan_AWS_total_cleaned.xlsx"
df = pd.read_excel(file_path)

# 5️⃣ Jalankan AutoViz
AV = AutoViz_Class()

# Target yang akan dianalisis (jika kamu ingin melihat prediktor terhadap curah hujan)
target_variable = 'rr'  # Jika tidak mau fokus ke target, ganti dengan depVar=''

# 6️⃣ Jalankan analisis
AV.AutoViz(
    filename="",     # Kosongkan jika pakai df langsung
    dfte=df,         # DataFrame
    depVar=target_variable,  # Target variabel (bisa juga "" jika eksplorasi umum)
    verbose=1
)
