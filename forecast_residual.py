import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# Restrukturisasi Data: Menggeser target untuk prediksi 3 jam ke depan
X_data = data[features]
y_data = data[target].shift(-1)
X_data = X_data[:-1]
y_data = y_data[:-1]

# --- PERBAIKAN: PEMBAGIAN DATA & NORMALISASI YANG BENAR ---
# 1. Bagi data mentah (unscaled) menjadi latih dan uji terlebih dahulu
split_percentage = 0.8
split_point = int(len(X_data) * split_percentage)
X_train_raw, X_test_raw = X_data[:split_point], X_data[split_point:]
y_train_raw, y_test_raw = y_data[:split_point], y_data[split_point:]

# 2. Latih scaler HANYA pada data latih mentah
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1))

# 3. Gunakan scaler yang sama untuk mentransformasi data uji
X_test_scaled = scaler_X.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw.values.reshape(-1, 1))

# --- PERBAIKAN: Membuat Sekuens SETELAH Normalisasi ---
def create_sequences(X_data, y_data, time_steps=8):
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        v = X_data[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 8 # Gunakan data 24 jam terakhir (8 * 3 jam)
# 4. Buat sekuens dari data latih dan uji yang sudah di-scale
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

# --- PERBAIKAN: MEMBUAT SET VALIDASI TERPISAH ---
# 5. Bagi data sekuens latih menjadi latih akhir dan validasi
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_seq, y_train_seq, test_size=0.2, shuffle=False
)
print(f"Data sekuensial siap: {len(X_train_final)} data latih, {len(X_val)} data validasi, {len(X_test_seq)} data uji.")

# Buat dictionary untuk menyimpan hasil prediksi
results = {}

# --- 2. Latih Model dan Buat Prediksi ---
# Meratakan input 3D menjadi 2D untuk RF dan XGBoost
X_train_final_flat = X_train_final.reshape(X_train_final.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

# ----- Model 1: Random Forest -----
print("\nMelatih model Random Forest...")
params_rf = {'max_depth': 6, 'max_features': 'log2', 'max_leaf_nodes': 40, 'min_samples_leaf': 5, 'min_samples_split': 21, 'n_estimators': 413}
model_rf = RandomForestRegressor(**params_rf, random_state=42, n_jobs=-1)
model_rf.fit(X_train_final_flat, y_train_final.ravel())
results['Random Forest'] = {'test_pred': scaler_y.inverse_transform(model_rf.predict(X_test_flat).reshape(-1, 1)).flatten()}
print("Model Random Forest selesai.")

# ----- Model 2: XGBoost -----
print("Melatih model XGBoost dengan Early Stopping...")
params_xgb = {'colsample_bytree': 0.690, 'gamma': 0.322, 'learning_rate': 0.044, 'max_depth': 3, 'min_child_weight': 8, 'subsample': 0.807}
model_xgb = XGBRegressor(**params_xgb, n_estimators=1000, random_state=42, n_jobs=-1)
# PERBAIKAN: Gunakan set validasi untuk early stopping
model_xgb.fit(X_train_final_flat, y_train_final.ravel(), eval_set=[(X_val_flat, y_val)], early_stopping_rounds=30, verbose=False)
print(f"Pelatihan XGBoost berhenti pada iterasi ke-{model_xgb.best_iteration}")
results['XGBoost'] = {'test_pred': np.maximum(scaler_y.inverse_transform(model_xgb.predict(X_test_flat).reshape(-1, 1)).flatten(), 0)}
print("Model XGBoost selesai.")

# ----- Model 3: LSTM (Stacked Architecture) -----
print("Melatih model Stacked LSTM...")
params_lstm = {'units': 128, 'dropout': 0.2, 'learning_rate': 0.00534}
model_lstm = Sequential([
    LSTM(units=params_lstm['units'], return_sequences=True, input_shape=(X_train_final.shape[1], X_train_final.shape[2])),
    Dropout(params_lstm['dropout']),
    LSTM(units=int(params_lstm['units'] / 2)),
    Dropout(params_lstm['dropout']),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')
# PERBAIKAN: Tambahkan callback EarlyStopping dan gunakan set validasi
early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)
model_lstm.fit(X_train_final, y_train_final, epochs=100, batch_size=32, verbose=0,
               validation_data=(X_val, y_val),
               callbacks=[early_stopper])
test_pred_lstm_scaled = model_lstm.predict(X_test_seq)
results['LSTM'] = {'test_pred': np.maximum(scaler_y.inverse_transform(test_pred_lstm_scaled).flatten(), 0)}
print("Model LSTM selesai.")

# --- 3. Hitung Residual ---
y_test_orig = scaler_y.inverse_transform(y_test_seq).flatten()
residuals_rf = y_test_orig - results['Random Forest']['test_pred']
residuals_xgb = y_test_orig - results['XGBoost']['test_pred']
residuals_lstm = y_test_orig - results['LSTM']['test_pred']

# --- 4. Visualisasi Residual Plot Gabungan ---
print("\nMembuat Residual Plot...")
plt.figure(figsize=(12, 8))

plt.scatter(results['Random Forest']['test_pred'], residuals_rf, alpha=0.5, label='Random Forest', color='green')
plt.scatter(results['XGBoost']['test_pred'], residuals_xgb, alpha=0.5, label='XGBoost', color='red')
plt.scatter(results['LSTM']['test_pred'], residuals_lstm, alpha=0.5, label='LSTM', color='blue')

plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.title('Residual Plot Gabungan - Forecasting (Data AWS)', fontsize=16)
plt.xlabel('Nilai Prediksi Curah Hujan (mm)', fontsize=12)
plt.ylabel('Residual (Aktual - Prediksi)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
print("Selesai.")