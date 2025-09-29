import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram

# Membaca data dari lokasi file
file_path = r"E:\STMKG\SMESTER_7\Komputasi_Meteorologi\PC_TS.csv"
PC = pd.read_csv(file_path)
rainfall = PC['PC1'].values  # Mengambil data curah hujan
n_days = len(rainfall)

# Plot data curah hujan
plt.figure(figsize=(10, 5))
plt.plot(rainfall, label='Data Curah Hujan', color='blue')
ticks = np.linspace(0, n_days - 1, 7, dtype=int)  # Pastikan tidak melebihi indeks
plt.xticks(ticks, labels=PC['index'].iloc[ticks], rotation=45)
plt.title("Data Curah Hujan Harian")
plt.xlabel("Waktu")
plt.ylabel("Curah Hujan")
plt.grid()
plt.tight_layout()
plt.show()

# Hitung Power Spectral Density (PSD)
freq, power = periodogram(rainfall, scaling='density')

# Mengonversi frekuensi ke periode (dalam siklus per tahun)
period_per_year = 1 / (freq * 12)

# Menyaring data dengan periode valid
valid = (period_per_year > 0) & (period_per_year < np.inf)
period_per_year = period_per_year[valid]
power = power[valid]

# Menambahkan smoothing pada PSD
window_size = 6
smoothed_power = np.convolve(power, np.ones(window_size) / window_size, mode='same')

# Plotting PSD
plt.figure(figsize=(10, 5))
plt.plot(period_per_year, power, label='Original PSD', color='blue', linewidth=2)
# Plot garis smoothing
plt.plot(period_per_year, smoothed_power, label='Smoothed PSD', color='red', linestyle='--', linewidth=2)

# Konfigurasi sumbu x dengan skala logaritmik
plt.xscale('log')
plt.title("Power Spectral Density (PSD) dengan Smoothing")
plt.xlabel("Periode (Siklus per Tahun) [Log Scale]")
plt.ylabel("PSD")
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Kustomisasi ticks pada sumbu x
ticks = [0.01, 0.1, 1, 10, 100, 1000]
plt.xticks(ticks, labels=[f"{tick:.0e}" for tick in ticks])

plt.tight_layout()
plt.show()
