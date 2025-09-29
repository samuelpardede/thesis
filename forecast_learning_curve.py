import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Persiapan Data untuk Forecasting ---
print("Mempersiapkan data AWS untuk FORECASTING...")
file_path = r"E:\STMKG\SKRIPSI\datahujan\fix\Data_AWS_Per_3_Jam_Outlier.xlsx"
try:
    data = pd.read_excel(file_path, engine='openpyxl')
    print("File Excel berhasil dimuat.")
except Exception as e:
    print(f"Terjadi error saat membaca file Excel: {e}")
    exit()

# Preprocessing
features = ['tt_air_avg', 'rh_avg', 'ws_avg', 'pp_air']
target = 'rr'
data.replace([8888, 9999], np.nan, inplace=True)
data.interpolate(method='linear', inplace=True)
data.dropna(subset=features + [target], inplace=True)

# Restrukturisasi Data
X_data = data[features]
y_data = data[target].shift(-1)
X_data = X_data[:-1]
y_data = y_data[:-1]

# Normalisasi X dan y
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_data)
y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1, 1))

# --- Membuat Sekuens dengan Jendela Waktu ---
def create_sequences(X_data, y_data, time_steps=8):
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        v = X_data[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 8
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# --- Pembagian Data Kronologis (Tidak Acak) ---
split_percentage = 0.8
split_point = int(len(X_seq) * split_percentage)
X_train, X_val = X_seq[:split_point], X_seq[split_point:]
y_train_scaled, y_val_scaled = y_seq[:split_point], y_seq[split_point:]
print(f"Data sekuensial siap: {len(X_train)} data latih, {len(X_val)} data validasi.")

# --- 2. Latih Model LSTM ---
print("\nMelatih model LSTM...")
params_lstm = {'units': 128, 'dropout': 0.3, 'learning_rate': 0.00534, 'epochs': 200}

model_lstm = Sequential([
    LSTM(units=params_lstm['units'], input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(params_lstm['dropout']),
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
plt.title('Learning Curve - LSTM Forecasting (Data AWS)', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\nSelesai.")