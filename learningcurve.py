import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- 1. Persiapan Data AWS ---
print("Mempersiapkan data AWS...")
file_path = r"E:\STMKG\SKRIPSI\datahujan\fix\Data_AWS_Per_3_Jam_Outlier.xlsx"
data = pd.read_excel(file_path)

# Preprocessing
features = ['tt_air_avg', 'rh_avg', 'ws_avg', 'pp_air']
target = 'rr'

data.replace([8888, 9999], np.nan, inplace=True)
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

# Split Data menjadi Training dan Validation
# Kita menggunakan X_val dan y_val sebagai data validasi untuk memonitor loss
X_train, X_val, y_train, y_val = train_test_split(
    X_lstm, y_scaled, test_size=0.2, random_state=42
)
print("Data siap.")

# --- 2. Bangun dan Latih Model LSTM ---
print("\nMelatih model LSTM...")

# Gunakan hyperparameter terbaik untuk AWS
params_lstm = {
    'units': 64,
    'dropout': 0.0,
    'learning_rate': 0.00702,
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
    loss='mse' # Mean Squared Error adalah loss function yang umum untuk regresi
)

# Latih model dan simpan history-nya
history = model.fit(
    X_train, 
    y_train, 
    epochs=params_lstm['epochs'], 
    batch_size=32, 
    validation_data=(X_val, y_val), # Menyediakan data validasi
    verbose=1 # Tampilkan log training per epoch
)
print("Pelatihan model selesai.")


# --- 3. Visualisasi Learning Curve ---
print("\nMembuat Grafik Learning Curve...")

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='x')
plt.title('Learning Curve - Model LSTM (Data AWS)', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Selesai.")