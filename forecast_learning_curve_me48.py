import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Persiapan Data ME 48 ---
print("Mempersiapkan data ME 48 untuk FORECASTING...")
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
try:
    data = pd.read_excel(file_path, engine='openpyxl')
    print("File Excel berhasil dimuat.")
except Exception as e:
    print(f"Terjadi error saat membaca file Excel: {e}")
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
split_point = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:split_point], X_seq[split_point:]
y_train_scaled, y_val_scaled = y_seq[:split_point], y_seq[split_point:]
print(f"Data siap: {len(X_train)} data latih, {len(X_val)} data validasi.")

# --- 2. Bangun dan Latih Model LSTM ---
print("\nMelatih model LSTM...")
params_lstm = {'units': 96, 'learning_rate': 0.00125, 'epochs': 200}

model_lstm = Sequential([
    LSTM(units=params_lstm['units'], input_shape=(time_steps, len(features))),
    Dropout(0.2), # Menambahkan dropout
    Dense(1)
])
model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')

# Menambahkan EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Latih model dan simpan history-nya
history = model_lstm.fit(
    X_train, 
    y_train_scaled, 
    epochs=params_lstm['epochs'], 
    batch_size=32, 
    validation_data=(X_val, y_val_scaled),
    callbacks=[early_stopping],
    verbose=1
)
print("Pelatihan model selesai.")

# --- 3. Visualisasi Learning Curve ---
print("\nMembuat Grafik Learning Curve...")
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='x')
plt.title('Learning Curve - LSTM Forecasting (Data ME 48)', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Selesai.")