import pandas as pd
import numpy as np
import joblib
import time

start = time.time()
print("Memuat model Random Forest untuk prediksi data ME 48...")

# --- 1. Load Model dan Scaler ---
try:
    model = joblib.load('rf_forecast_model.pkl')
    scaler_X = joblib.load('scaler_X_forecast.pkl')
    scaler_y = joblib.load('scaler_y_forecast.pkl')
    print("Model dan scaler berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model atau scaler: {e}")
    exit()

# --- 2. Input File ---
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"

try:
    data = pd.read_excel(file_path, engine="openpyxl")
    print("File input berhasil dimuat.")
except Exception as e:
    print(f"Error membaca file: {e}")
    exit()

# --- 3. Preprocessing ---
features = [
    "CLOUD_LOW_TYPE_CL", "CLOUD_LOW_MED_AMT_OKTAS", "CLOUD_MED_TYPE_CM", "CLOUD_HIGH_TYPE_CH",
    "CLOUD_COVER_OKTAS_M", "LAND_COND", "PRESENT_WEATHER_WW", "TEMP_DEWPOINT_C_TDTDTD",
    "TEMP_DRYBULB_C_TTTTTT", "TEMP_WETBULB_C", "WIND_SPEED_FF", "RELATIVE_HUMIDITY_PC",
    "PRESSURE_QFF_MB_DERIVED", "PRESSURE_QFE_MB_DERIVED"
]

missing_cols = [col for col in features if col not in data.columns]
if missing_cols:
    print(f"Kolom berikut tidak ditemukan di file input: {missing_cols}")
    exit()

# --- 4. Prediksi ---
X_new = data[features]
X_scaled = scaler_X.transform(X_new)

pred_scaled = model.predict(X_scaled)
pred_orig = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

data["Prediksi_RF"] = pred_orig

# --- 5. Fungsi Klasifikasi ---
def klasifikasi_hujan(rr):
    if rr < 0.2:
        return "Tidak Hujan"
    elif rr < 5:
        return "Hujan Ringan"
    elif rr < 20:
        return "Hujan Sedang"
    else:
        return "Hujan Lebat"

# --- 5a. Buat klasifikasi Observasi & Prediksi ---
if "RR" in data.columns:
    data["Klasifikasi_Observed"] = data["RR"].apply(klasifikasi_hujan)
else:
    print("Kolom 'RR' (observasi hujan) tidak ada di file input, hanya klasifikasi prediksi yang dibuat.")
    data["Klasifikasi_Observed"] = np.nan

data["Klasifikasi_Prediksi_Original"] = data["Prediksi_RF"].apply(klasifikasi_hujan)

# --- 5b. Adjust klasifikasi prediksi dengan toleransi ±3 mm ---
toleransi = 3.0
klasifikasi_adjusted = []

for obs, pred, klas_obs, klas_pred in zip(data["RR"], data["Prediksi_RF"], 
                                          data["Klasifikasi_Observed"], data["Klasifikasi_Prediksi_Original"]):
    # default klasifikasi prediksi
    klas_final = klas_pred
    
    # cek toleransi (hanya kalau ada nilai observasi)
    if not pd.isna(obs):
        if klas_obs != klas_pred:  # berbeda kategori
            if abs(pred - obs) <= toleransi:
                klas_final = klas_obs  # sesuaikan ke observasi
    
    klasifikasi_adjusted.append(klas_final)

data["Klasifikasi_Prediksi_Adjusted"] = klasifikasi_adjusted

# --- 6. Simpan File ---
output_path = r"E:\STMKG\SKRIPSI\output_file\studi_kasus_random_forest_klasifikasi_lengkap_fixxxx.xlsx"

try:
    data.to_excel(output_path, index=False, engine="xlsxwriter")
    print(f"\nPrediksi dan klasifikasi berhasil ditambahkan! File disimpan sebagai:\n{output_path}")
except Exception as e:
    print(f"Error menyimpan file: {e}")

print(f"Waktu total eksekusi: {time.time() - start:.2f} detik")
