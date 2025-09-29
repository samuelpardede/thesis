import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# --- 1. Persiapan Data untuk Forecasting ---
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
df_forecast = data[features].copy()
df_forecast['target_RR'] = data[target].shift(-1)
df_forecast.dropna(inplace=True)

X = df_forecast[features]
y = df_forecast['target_RR']

# Normalisasi X dan y
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape input menjadi [sampel, 1, fitur]
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Pembagian Data Kronologis
split_percentage = 0.8
split_point = int(len(X_lstm) * split_percentage)
X_train, X_test = X_lstm[:split_point], X_lstm[split_point:]
y_train_scaled, y_test_scaled = y_scaled[:split_point], y_scaled[split_point:]
print(f"Data siap: {len(X_train)} data latih, {len(X_test)} data uji.")

# --- 2. Latih Model LSTM ---
print("\nMelatih model LSTM...")
params_lstm = {'units': 96, 'dropout': 0.0, 'learning_rate': 0.00125, 'epochs': 100}

# --- PERBAIKAN DI SINI: Ganti nama variabel 'model' menjadi 'model_lstm' ---
model_lstm = Sequential([
    LSTM(units=params_lstm['units'], input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(64, activation='relu'),
    Dense(1)
])
# -------------------------------------------------------------------------

model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')
model_lstm.fit(X_train, y_train_scaled, epochs=params_lstm['epochs'], batch_size=32, verbose=0)
print("Model LSTM selesai dilatih.")

# --- 3. Prediksi dan Inversi Skala ---
train_pred_scaled = model_lstm.predict(X_train)
test_pred_scaled = model_lstm.predict(X_test)

train_pred_orig = np.maximum(scaler_y.inverse_transform(train_pred_scaled).flatten(), 0)
test_pred_orig = np.maximum(scaler_y.inverse_transform(test_pred_scaled).flatten(), 0)

# --- 4. Hitung dan Tampilkan Metrik Evaluasi ---
y_train_orig = scaler_y.inverse_transform(y_train_scaled).flatten()
y_test_orig = scaler_y.inverse_transform(y_test_scaled).flatten()

print("\n" + "="*50)
print("HASIL EVALUASI KINERJA MODEL LSTM (FORECASTING - ME 48)")
print("="*50)

# Evaluasi pada data training
train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))
train_mae = mean_absolute_error(y_train_orig, train_pred_orig)
train_r2 = r2_score(y_train_orig, train_pred_orig)
    
# Evaluasi pada data testing
test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
test_mae = mean_absolute_error(y_test_orig, test_pred_orig)
test_r2 = r2_score(y_test_orig, test_pred_orig)
    
print(f"[Train] RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
print(f"[Test ] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R²: {test_r2:.3f}")

print("="*50)
print("\nSelesai.")