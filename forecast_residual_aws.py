import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Persiapan Data untuk Forecasting ---
print("Mempersiapkan data ME 48 untuk FORECASTING...")
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
try:
    data = pd.read_excel(file_path, engine='openpyxl')
    print("File Excel berhasil dimuat.")
except Exception as e:
    print(f"Error membaca file: {e}")
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
df_forecast = data[features].copy()
df_forecast['target_RR'] = data[target].shift(-1)
df_forecast.dropna(inplace=True)

X = df_forecast[features]
y = df_forecast['target_RR']

# --- PERBAIKAN: PEMBAGIAN DATA & NORMALISASI YANG KONSISTEN ---
print("\nMembagi dan menormalisasi data dengan benar...")
split_percentage = 0.8
split_point = int(len(X) * split_percentage)

# 1. Bagi data mentah menjadi set latih dan uji TERLEBIH DAHULU
X_train_raw, X_test_raw = X.iloc[:split_point], X.iloc[split_point:]
y_train_raw, y_test_raw = y.iloc[:split_point], y.iloc[split_point:]

# 2. Latih scaler HANYA pada data latih
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1))

# 3. Gunakan scaler yang sama untuk mentransformasi data uji
X_test = scaler_X.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw.values.reshape(-1, 1))

print(f"Data siap: {len(X_train)} data latih, {len(X_test)} data uji.")

# --- PERBAIKAN: BUAT SET VALIDASI TERPISAH UNTUK EARLY STOPPING ---
# Penting: Set validasi diambil dari data latih, bukan data uji
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train_scaled, test_size=0.2, shuffle=False
)

# Buat dictionary untuk menyimpan hasil prediksi
results = {}

# --- 2. Latih Model dan Buat Prediksi ---

# ----- Model 1: Random Forest -----
print("\nMelatih model Random Forest...")
params_rf = {'max_depth': 6, 'max_features': 'log2', 'max_leaf_nodes': 40, 'min_samples_leaf': 5, 'min_samples_split': 21, 'n_estimators': 413}
model_rf = RandomForestRegressor(**params_rf, random_state=42, n_jobs=-1)
# Latih pada data training final
model_rf.fit(X_train_final, y_train_final.ravel())
results['Random Forest'] = {'test_pred': scaler_y.inverse_transform(model_rf.predict(X_test).reshape(-1, 1)).flatten()}
print("Model Random Forest selesai.")

# ----- Model 2: XGBoost -----
print("\nMelatih model XGBoost...")
params_xgb = {'colsample_bytree': 0.888, 'gamma': 0.469, 'learning_rate': 0.010, 'max_depth': 14, 'min_child_weight': 5, 'subsample': 0.721}
model_xgb = XGBRegressor(**params_xgb, n_estimators=1000, random_state=42, n_jobs=-1)
# Latih dengan set validasi yang benar
model_xgb.fit(X_train_final, y_train_final.ravel(), eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
print(f"Pelatihan XGBoost berhenti pada iterasi ke-{model_xgb.best_iteration}")
results['XGBoost'] = {'test_pred': np.maximum(scaler_y.inverse_transform(model_xgb.predict(X_test).reshape(-1, 1)).flatten(), 0)}
print("Model XGBoost selesai.")

# ----- Model 3: LSTM -----
print("\nMelatih model LSTM...")
# Reshape data untuk input LSTM [samples, timesteps, features]
X_train_final_lstm = X_train_final.reshape((X_train_final.shape[0], 1, X_train_final.shape[1]))
X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

params_lstm = {'units': 96, 'dropout': 0.0, 'learning_rate': 0.00125}

# PERBAIKAN: `input_shape` yang benar
model_lstm = Sequential([
    LSTM(units=params_lstm['units'], input_shape=(X_train_final_lstm.shape[1], X_train_final_lstm.shape[2]), dropout=params_lstm['dropout']),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')

# PERBAIKAN: Gunakan Early Stopping untuk LSTM juga
early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)
model_lstm.fit(X_train_final_lstm, y_train_final,
               epochs=100, # Beri epoch lebih banyak, biarkan early stopping bekerja
               batch_size=32,
               validation_data=(X_val_lstm, y_val),
               callbacks=[early_stopper],
               verbose=0)

test_pred_lstm_scaled = model_lstm.predict(X_test_lstm)
results['LSTM'] = {'test_pred': np.maximum(scaler_y.inverse_transform(test_pred_lstm_scaled).flatten(), 0)}
print("Model LSTM selesai.")

# --- 3. Hitung Residual ---
y_test_orig = scaler_y.inverse_transform(y_test_scaled).flatten()
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
plt.title('Residual Plot Gabungan - Forecasting (Data ME 48)', fontsize=16)
plt.xlabel('Nilai Prediksi Curah Hujan (mm)', fontsize=12)
plt.ylabel('Residual (Aktual - Prediksi)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
print("Selesai.")