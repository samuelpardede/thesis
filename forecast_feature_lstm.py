import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import shap

# --- 1. Persiapan Data untuk Forecasting ---
print("Mempersiapkan data ME 48 untuk FORECASTING...")
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
try:
    data = pd.read_excel(file_path, engine='openpyxl')
    print("File Excel berhasil dimuat.")
except Exception as e:
    print(f"Terjadi error saat membaca file: {e}")
    exit()

# Preprocessing
features = [
    "CLOUD_LOW_TYPE_CL", "CLOUD_LOW_MED_AMT_OKTAS", "CLOUD_MED_TYPE_CM", "CLOUD_HIGH_TYPE_CH",
    "CLOUD_COVER_OKTAS_M", "LAND_COND", "PRESENT_WEATHER_WW", "TEMP_DEWPOINT_C_TDTDTD",
    "TEMP_DRYBULB_C_TTTTTT", "TEMP_WETBULB_C", "WIND_SPEED_FF", "RELATIVE_HUMIDITY_PC",
    "PRESSURE_QFF_MB_DERIVED", "PRESSURE_QFE_MB_DERIVED"
]
target = "RR"

data.replace([8888, 9999], np.nan, inplace=True)
for col in features + [target]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.interpolate(method='linear', inplace=True)
data.dropna(subset=features + [target], inplace=True)

# Restrukturisasi Data
X_data = data[features]
y_data = data[target].shift(-1)
X_data = X_data[:-1]
y_data = y_data[:-1]

# Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_data)
y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1, 1))

# Membuat Sekuens
def create_sequences(X_data, y_data, time_steps=8):
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        Xs.append(X_data[i:(i + time_steps)])
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 8
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Pembagian Data Kronologis
split_percentage = 0.8
split_point = int(len(X_seq) * split_percentage)
X_train, X_test = X_seq[:split_point], X_seq[split_point:]
y_train_scaled, y_test_scaled = y_seq[:split_point], y_seq[split_point:]
print(f"Data sekuensial siap: {len(X_train)} data latih, {len(X_test)} data uji.")

# --- 2. Latih Model LSTM ---
print("Melatih model LSTM...")
# Menggunakan hyperparameter untuk ME 48
params_lstm = {'units': 96, 'learning_rate': 0.00125}
model = Sequential([
    LSTM(units=params_lstm['units'], input_shape=(time_steps, len(features))),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')
model.fit(X_train, y_train_scaled, epochs=50, batch_size=32, verbose=0)
print("Model LSTM selesai dilatih.")

# --- 3. Analisis SHAP ---
print("Menjalankan analisis SHAP...")
background_data = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
explainer = shap.GradientExplainer(model, background_data)
shap_values_3d = explainer.shap_values(X_test)
print("Analisis SHAP selesai.")

# --- 4. Visualisasi ---
# Ambil rata-rata nilai absolut SHAP
global_importance = np.mean(np.abs(shap_values_3d[0]), axis=(0, 1))
# Dapatkan indeks untuk mengurutkan fitur
sorted_indices = np.argsort(global_importance)

print("Membuat Grafik Feature Importance...")
plt.figure(figsize=(10, 8))
plt.barh(np.array(features)[sorted_indices], global_importance[sorted_indices], color='salmon')
plt.title("Feature Importance - LSTM Forecasting (ME 48)", fontsize=16)
plt.xlabel("Rata-rata Pengaruh Absolut (SHAP Value)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nSelesai.")