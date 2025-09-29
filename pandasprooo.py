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
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# --- 1. Persiapan Data ---
print("Mempersiapkan data AWS...")
file_path = r"E:\STMKG\SKRIPSI\datahujan\fix\Data_AWS_Per_3_Jam_Outlier.xlsx"
data = pd.read_excel(file_path)

# Preprocessing
features = ['tt_air_avg', 'rh_avg', 'ws_avg', 'pp_air']
target = 'rr'
data.replace([8888, 9999], np.nan, inplace=True)
data.interpolate(method='linear', inplace=True)
data['Jam'] = pd.to_datetime(data['Jam'])
data.set_index('Jam', inplace=True)
data.dropna(subset=features + [target], inplace=True)

X = data[features]
y = data[target]

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_scaled, y.values, data.index, test_size=0.2, random_state=42
)
print("Data siap.")

# Buat dictionary untuk menyimpan hasil
results = {}

# --- 2. Latih Model dan Buat Prediksi ---

# ----- Model 1: Random Forest -----
print("\nMelatih model Random Forest...")
params_rf = {'n_estimators': 451, 'max_depth': 19, 'min_samples_split': 8, 'min_samples_leaf': 3, 'max_leaf_nodes': 139, 'max_features': 'sqrt'}
model_rf = RandomForestRegressor(**params_rf, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
results['Random Forest'] = {'test_pred': model_rf.predict(X_test)}
print("Model Random Forest selesai.")

# ----- Model 2: XGBoost -----
print("Melatih model XGBoost...")
params_xgb = {'n_estimators': 271, 'max_depth': 7, 'learning_rate': 0.0386, 'subsample': 0.855, 'colsample_bytree': 0.94, 'gamma': 0.225, 'min_child_weight': 3}
model_xgb = XGBRegressor(**params_xgb, random_state=42, n_jobs=-1)
model_xgb.fit(X_train, y_train)
results['XGBoost'] = {'test_pred': np.maximum(model_xgb.predict(X_test), 0)}
print("Model XGBoost selesai.")

# ----- Model 3: LSTM -----
print("Melatih model LSTM...")
scaler_y = MinMaxScaler()
y_train_lstm_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

params_lstm = {'units': 64, 'dropout': 0.0, 'learning_rate': 0.00702, 'epochs': 50}
model_lstm = Sequential([LSTM(units=params_lstm['units'], input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), dropout=params_lstm['dropout']), Dense(1)])
model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')
model_lstm.fit(X_train_lstm, y_train_lstm_scaled, epochs=params_lstm['epochs'], batch_size=32, verbose=0)

test_pred_lstm_scaled = model_lstm.predict(X_test_lstm)
results['LSTM'] = {'test_pred': np.maximum(scaler_y.inverse_transform(test_pred_lstm_scaled).flatten(), 0)}
print("Model LSTM selesai.")

# --- 4. Visualisasi Grafik Aktual vs. Prediksi (Januari 2024) ---
print("\nMembuat Grafik Aktual vs. Prediksi untuk Januari 2024...")

# Buat DataFrame baru untuk plotting dengan index tanggal
plot_df = pd.DataFrame({
    'Aktual': y_test,
    'Prediksi RF': results['Random Forest']['test_pred'],
    'Prediksi XGBoost': results['XGBoost']['test_pred'],
    'Prediksi LSTM': results['LSTM']['test_pred']
}, index=test_indices)

# ---- PERBAIKAN DI SINI ----
# Pastikan index adalah tipe datetime sebelum memfilter
plot_df.index = pd.to_datetime(plot_df.index)
# ---------------------------

# Filter DataFrame hanya untuk Januari 2024 dan urutkan berdasarkan tanggal
# Menggunakan metode filter yang lebih eksplisit untuk menghindari error
plot_event_df = plot_df[(plot_df.index.year == 2024) & (plot_df.index.month == 1)].sort_index()

if not plot_event_df.empty:
    plt.figure(figsize=(15, 7))
    
    # Plot data dari DataFrame yang sudah difilter
    plt.plot(plot_event_df.index, plot_event_df['Aktual'], label='Aktual', color='black', linewidth=1, marker='o', markersize=3)
    plt.plot(plot_event_df.index, plot_event_df['Prediksi RF'], label='Prediksi Random Forest', color='green', linestyle='--')
    plt.plot(plot_event_df.index, plot_event_df['Prediksi XGBoost'], label='Prediksi XGBoost', color='red', linestyle=':')
    plt.plot(plot_event_df.index, plot_event_df['Prediksi LSTM'], label='Prediksi LSTM', color='blue', linestyle='-.')
    
    # Label dan Judul
    plt.title('Grafik Aktual vs. Prediksi - Januari 2024 (Data AWS)', fontsize=16)
    plt.xlabel('Tanggal', fontsize=12)
    plt.ylabel('Curah Hujan (mm)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Tidak ada data uji untuk Januari 2024 yang ditemukan setelah pemisahan data acak.")

print("Selesai.")