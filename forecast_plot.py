import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

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

# Konversi kolom waktu dan set sebagai index
time_column = 'Jam'
data[time_column] = pd.to_datetime(data[time_column])
data.set_index(time_column, inplace=True)
data.dropna(subset=features + [target], inplace=True)

# --- RESTRUKTURISASI DATA UNTUK FORECASTING ---
X_data = data[features]
y_data = data[target].shift(-1)
# Hapus baris terakhir karena tidak memiliki target masa depan
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
# Simpan index waktu yang sesuai untuk data sekuensial
seq_indices = X_data.index[time_steps:]

# --- Pembagian Data Kronologis ---
split_percentage = 0.8
split_point = int(len(X_seq) * split_percentage)
X_train, X_test = X_seq[:split_point], X_seq[split_point:]
y_train_scaled, y_test_scaled = y_seq[:split_point], y_seq[split_point:]
train_indices, test_indices = seq_indices[:split_point], seq_indices[split_point:]
print(f"Data sekuensial siap: {len(X_train)} data latih, {len(X_test)} data uji.")

results = {}

# --- 2. Latih Model dan Buat Prediksi ---
# Meratakan input 3D menjadi 2D untuk RF dan XGBoost
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# ----- Model 1: Random Forest -----
print("\nMelatih model Random Forest...")
params_rf = {'n_estimators': 451, 'max_depth': 19, 'min_samples_split': 8, 'min_samples_leaf': 3, 'max_leaf_nodes': 139, 'max_features': 'sqrt'}
model_rf = RandomForestRegressor(**params_rf, random_state=42, n_jobs=-1)
model_rf.fit(X_train_flat, y_train_scaled.ravel())
results['Random Forest'] = {'test_pred': scaler_y.inverse_transform(model_rf.predict(X_test_flat).reshape(-1, 1)).flatten()}
print("Model Random Forest selesai.")

# ----- Model 2: XGBoost -----
print("Melatih model XGBoost...")
params_xgb = {'n_estimators': 271, 'max_depth': 7, 'learning_rate': 0.0386, 'subsample': 0.855, 'colsample_bytree': 0.94, 'gamma': 0.225, 'min_child_weight': 3}
model_xgb = XGBRegressor(**params_xgb, random_state=42, n_jobs=-1)
model_xgb.fit(X_train_flat, y_train_scaled.ravel())
results['XGBoost'] = {'test_pred': np.maximum(scaler_y.inverse_transform(model_xgb.predict(X_test_flat).reshape(-1, 1)).flatten(), 0)}
print("Model XGBoost selesai.")

# ----- Model 3: LSTM -----
print("Melatih model LSTM...")
params_lstm = {'units': 64, 'dropout': 0.0, 'learning_rate': 0.00702, 'epochs': 50}
model_lstm = Sequential([
    LSTM(units=params_lstm['units'], input_shape=(time_steps, len(features)), dropout=params_lstm['dropout']),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')
model_lstm.fit(X_train, y_train_scaled, epochs=params_lstm['epochs'], batch_size=32, verbose=0)
test_pred_lstm_scaled = model_lstm.predict(X_test)
results['LSTM'] = {'test_pred': np.maximum(scaler_y.inverse_transform(test_pred_lstm_scaled).flatten(), 0)}
print("Model LSTM selesai.")

# --- 3. Visualisasi Grafik Aktual vs. Prediksi (Periode 30 Hari) ---
print("\nMembuat Grafik Aktual vs. Prediksi untuk periode 30 hari...")

y_test_orig = scaler_y.inverse_transform(y_test_scaled).flatten()
plot_df = pd.DataFrame({
    'Aktual': y_test_orig,
    'Prediksi RF': results['Random Forest']['test_pred'],
    'Prediksi XGBoost': results['XGBoost']['test_pred'],
    'Prediksi LSTM': results['LSTM']['test_pred']
}, index=test_indices)

# Urutkan DataFrame berdasarkan indeks tanggalnya
plot_df.sort_index(inplace=True)

# ---- PERBAIKAN UTAMA: Filter rentang waktu yang lebih presisi ----
if not plot_df.empty:
    start_date = plot_df.index.min()
    # Gunakan Timedelta untuk durasi 30 hari yang pasti
    end_date = start_date + pd.Timedelta(days=30)
    plot_event_df = plot_df.loc[start_date:end_date]
    
    # Reset index agar sumbu-X menjadi urutan angka biasa
    plot_event_df.reset_index(drop=True, inplace=True)

    print(f"Menampilkan plot untuk periode 30 hari, dimulai dari {start_date.date()}")
    
    plt.figure(figsize=(15, 7))
    
    plt.plot(plot_event_df.index, plot_event_df['Aktual'], label='Aktual', color='black', linewidth=1.5, marker='o', markersize=3)
    plt.plot(plot_event_df.index, plot_event_df['Prediksi RF'], label='Prediksi Random Forest', color='green', linestyle='--')
    plt.plot(plot_event_df.index, plot_event_df['Prediksi XGBoost'], label='Prediksi XGBoost', color='red', linestyle=':')
    plt.plot(plot_event_df.index, plot_event_df['Prediksi LSTM'], label='Prediksi LSTM', color='blue', linestyle='-.')
    
    plt.title(f'Grafik Aktual vs. Prediksi - Periode 30 Hari (Data AWS)', fontsize=16)
    plt.xlabel('Indeks Data Uji', fontsize=12)
    plt.ylabel('Curah Hujan (mm)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print("Tidak ada data uji untuk diplot.")

print("Selesai.")