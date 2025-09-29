import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import keras_tuner as kt

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

# --- PERUBAHAN UTAMA: Restrukturisasi Data untuk Forecasting ---
# Menggunakan data saat ini (t) untuk memprediksi curah hujan 3 jam berikutnya (t+1)
df_forecast = data[features].copy()
df_forecast['target_RR'] = data[target].shift(-1)
df_forecast.dropna(inplace=True) # Hapus baris terakhir yang tidak punya target

X = df_forecast[features]
y = df_forecast['target_RR']

# Normalisasi Fitur
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Inisialisasi TimeSeriesSplit untuk validasi silang pada data deret waktu
tscv = TimeSeriesSplit(n_splits=5)
print(f"Data siap untuk di-tuning. Jumlah data: {len(X_scaled)}")


# --- 2. Tuning Hyperparameter Random Forest ---
print("\nMemulai tuning untuk Random Forest...")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_dist_rf = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(10, 40),
    'min_samples_leaf': randint(5, 20),
    'max_features': ['sqrt', 'log2']
}
rf_search = RandomizedSearchCV(
    estimator=rf, param_distributions=param_dist_rf, n_iter=30, cv=tscv,
    scoring='neg_mean_squared_error', random_state=42, n_jobs=-1, verbose=1
)
rf_search.fit(X_scaled, y)
print("\n=== Hyperparameter Terbaik (Random Forest) ===")
print(rf_search.best_params_)


# --- 3. Tuning Hyperparameter XGBoost ---
print("\nMemulai tuning untuk XGBoost...")
xgb = XGBRegressor(random_state=42, n_jobs=-1)
param_dist_xgb = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'min_child_weight': randint(1, 10)
}
xgb_search = RandomizedSearchCV(
    estimator=xgb, param_distributions=param_dist_xgb, n_iter=30, cv=tscv,
    scoring='neg_mean_squared_error', random_state=42, n_jobs=-1, verbose=1
)
xgb_search.fit(X_scaled, y)
print("\n=== Hyperparameter Terbaik (XGBoost) ===")
print(xgb_search.best_params_)


# --- 4. Tuning Hyperparameter LSTM ---
print("\nMemulai tuning untuk LSTM...")
# Untuk LSTM, kita split data secara manual (kronologis) untuk Keras Tuner
split_point = int(len(X_scaled) * 0.8)
X_train_lstm, X_val_lstm = X_scaled[:split_point], X_scaled[split_point:]
y_train_lstm, y_val_lstm = y.values[:split_point], y.values[split_point:]

# Normalisasi target (y) khusus untuk LSTM
scaler_y = MinMaxScaler()
y_train_lstm_scaled = scaler_y.fit_transform(y_train_lstm.reshape(-1, 1))
y_val_lstm_scaled = scaler_y.transform(y_val_lstm.reshape(-1, 1))

# Reshape input untuk LSTM
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], 1, X_train_lstm.shape[1]))
X_val_lstm = X_val_lstm.reshape((X_val_lstm.shape[0], 1, X_val_lstm.shape[1]))

def build_model(hp):
    model = Sequential([
        LSTM(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
            dropout=hp.Float('dropout', 0.0, 0.3, step=0.1)
        ),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='mse'
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='tuning_lstm_forecast_me48',
    project_name='rainfall_forecast_me48',
    overwrite=True
)

tuner.search(X_train_lstm, y_train_lstm_scaled, epochs=50, validation_data=(X_val_lstm, y_val_lstm_scaled))
best_hps_lstm = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n=== Hyperparameter Terbaik (LSTM) ===")
print(f"Units: {best_hps_lstm.get('units')}")
print(f"Dropout: {best_hps_lstm.get('dropout')}")
print(f"Learning Rate: {best_hps_lstm.get('lr')}")

print("\nProses tuning untuk semua model selesai.")