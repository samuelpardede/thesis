import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class

# 1. Baca file Excel terlebih dahulu
data = pd.read_excel("E:/STMKG/SKRIPSI/datahujan/ME_48/me48_3jam.xlsx")


# 3. Inisialisasi dan jalankan AutoViz
AV = AutoViz_Class()

dfte = AV.AutoViz(
    filename='',           # Kosongkan karena pakai dfte
    dfte=data,             # DataFrame yang sudah dibaca
    depVar='RR',           # Kolom target
    header=0,
    verbose=2,
    lowess=False,
    chart_format='png',
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir='E:/STMKG/SKRIPSI/output_file/analisis/analisisme48sebelum',
)
