import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
X_train, X_test = X_seq[:split_point], X_seq[split_point:]
y_train_scaled, y_test_scaled = y_seq[:split_point], y_seq[split_point:]
print(f"Data sekuensial siap: {len(X_train)} data latih, {len(X_test)} data uji.")

# --- 2. Latih Model LSTM dengan Perbaikan ---
print("\nMelatih model LSTM...")
params_lstm = {'units': 96, 'dropout': 0.2, 'learning_rate': 0.00125, 'epochs': 200}

# Arsitektur disederhanakan dan Dropout ditambahkan
model_lstm = Sequential([
    LSTM(units=params_lstm['units'], input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(params_lstm['dropout']),
    Dense(1)
])

model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')

# Menambahkan EarlyStopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Latih model dengan callback
model_lstm.fit(
    X_train, 
    y_train_scaled, 
    epochs=params_lstm['epochs'], 
    batch_size=32, 
    verbose=1, 
    validation_data=(X_test, y_test_scaled),
    callbacks=[early_stopping]
)
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