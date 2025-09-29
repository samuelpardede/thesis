import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- 1. Persiapan Data ME 48 ---
print("Mempersiapkan data ME 48...")
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
data = pd.read_excel(file_path)

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

X = data[features].values
y = data[target].values.reshape(-1, 1)

# Normalisasi Fitur (X) dan Target (y)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshape input menjadi 3D [sampel, timesteps, fitur]
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split Data menjadi Training dan Validation (test_size 0.1 sesuai skrip ME 48)
X_train, X_val, y_train, y_val = train_test_split(
    X_lstm, y_scaled, test_size=0.1, random_state=42
)
print("Data siap.")

# --- 2. Bangun dan Latih Model LSTM ---
print("\nMelatih model LSTM...")

# Gunakan hyperparameter terbaik untuk ME 48
params_lstm = {
    'units': 32,
    'dropout': 0.0,
    'learning_rate': 0.00308,
    'epochs': 50
}

model = Sequential([
    LSTM(
        units=params_lstm['units'], 
        input_shape=(X_train.shape[1], X_train.shape[2]), 
        dropout=params_lstm['dropout']
    ),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=params_lstm['learning_rate']), 
    loss='mse'
)

# Latih model dan simpan history-nya
history = model.fit(
    X_train, 
    y_train, 
    epochs=params_lstm['epochs'], 
    batch_size=32, 
    validation_data=(X_val, y_val), # Menyediakan data validasi
    verbose=1
)
print("Pelatihan model selesai.")


# --- 3. Visualisasi Learning Curve ---
print("\nMembuat Grafik Learning Curve...")

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='x')
plt.title('Learning Curve - Model LSTM (Data ME 48)', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Selesai.")