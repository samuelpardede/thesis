import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# --- 1. Persiapan Data AWS ---
print("Mempersiapkan data AWS...")
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

X = data[features]
y = data[target]

# Normalisasi X dan y
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split data yang sudah di-scale beserta indexnya
X_train, X_test, y_train_scaled, y_test_scaled, train_indices, test_indices = train_test_split(
    X_scaled, y_scaled, data.index, test_size=0.2, random_state=42
)
print("Data siap.")

results = {}

# --- 2. Latih Model dan Buat Prediksi ---

# ----- Model 1: Random Forest -----
print("\nMelatih model Random Forest...")
params_rf = {'n_estimators': 451, 'max_depth': 19, 'min_samples_split': 8, 'min_samples_leaf': 3, 'max_leaf_nodes': 139, 'max_features': 'sqrt'}
model_rf = RandomForestRegressor(**params_rf, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train_scaled.ravel())
results['Random Forest'] = {
    'test_pred': scaler_y.inverse_transform(model_rf.predict(X_test).reshape(-1, 1)).flatten()
}
print("Model Random Forest selesai.")

# ----- Model 2: XGBoost (dengan Early Stopping) -----
print("Melatih model XGBoost dengan Early Stopping...")
params_xgb_revised = {
    'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 1000,
    'subsample': 0.855, 'colsample_bytree': 0.94, 'gamma': 0.225, 'min_child_weight': 3
}
model_xgb = XGBRegressor(**params_xgb_revised, random_state=42, n_jobs=-1)
model_xgb.fit(X_train, y_train_scaled.ravel(), eval_set=[(X_test, y_test_scaled)], early_stopping_rounds=20, verbose=False)
print(f"Pelatihan XGBoost berhenti pada iterasi ke-{model_xgb.best_iteration}")
results['XGBoost'] = {
    'test_pred': np.maximum(scaler_y.inverse_transform(model_xgb.predict(X_test).reshape(-1, 1)).flatten(), 0)
}
print("Model XGBoost selesai.")

# ----- Model 3: LSTM -----
print("Melatih model LSTM...")
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
params_lstm = {'units': 64, 'dropout': 0.0, 'learning_rate': 0.00702, 'epochs': 50}
model_lstm = Sequential([
    LSTM(units=params_lstm['units'], input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), dropout=params_lstm['dropout']),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')
model_lstm.fit(X_train_lstm, y_train_scaled, epochs=params_lstm['epochs'], batch_size=32, verbose=0)
test_pred_lstm_scaled = model_lstm.predict(X_test_lstm)
results['LSTM'] = {
    'test_pred': np.maximum(scaler_y.inverse_transform(test_pred_lstm_scaled).flatten(), 0)
}
print("Model LSTM selesai.")
# --- 3. Visualisasi Grafik Aktual vs. Prediksi (November 2023) ---
print("\nMembuat Grafik Aktual vs. Prediksi untuk November 2023...")

y_test_orig = scaler_y.inverse_transform(y_test_scaled).flatten()
plot_df = pd.DataFrame({
    'Aktual': y_test_orig,
    'Prediksi RF': results['Random Forest']['test_pred'],
    'Prediksi XGBoost': results['XGBoost']['test_pred'],
    'Prediksi LSTM': results['LSTM']['test_pred']
}, index=test_indices)

# Urutkan DataFrame berdasarkan indeks tanggalnya
plot_df.sort_index(inplace=True)

# ---- PERUBAHAN DI SINI: Filter untuk rentang waktu November 2023 ----
plot_event_df = plot_df[(plot_df.index.year == 2023) & (plot_df.index.month == 11)]
# --------------------------------------------------------------------

if not plot_event_df.empty:
    # Reset index agar sumbu-X menjadi urutan angka biasa
    plot_event_df.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(15, 7))
    
    plt.plot(plot_event_df.index, plot_event_df['Aktual'], label='Aktual', color='black', linewidth=1.5, marker='o', markersize=3)
    plt.plot(plot_event_df.index, plot_event_df['Prediksi RF'], label='Prediksi Random Forest', color='green', linestyle='--')
    plt.plot(plot_event_df.index, plot_event_df['Prediksi XGBoost'], label='Prediksi XGBoost', color='red', linestyle=':')
    plt.plot(plot_event_df.index, plot_event_df['Prediksi LSTM'], label='Prediksi LSTM', color='blue', linestyle='-.')
    
    # ---- PERUBAHAN DI SINI: Judul grafik disesuaikan ----
    plt.title('Grafik Aktual vs. Prediksi - Data AWS', fontsize=16)
    # ----------------------------------------------------
    
    plt.xlabel('Indeks Data Uji', fontsize=12)
    plt.ylabel('Curah Hujan (mm)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print("Tidak ada data uji untuk November 2023 yang ditemukan setelah pemisahan data acak.")

print("Selesai.")