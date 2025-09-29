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

# --- 1. Persiapan Data ME 48 ---
print("Mempersiapkan data ME 48...")
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
data = pd.read_excel(file_path)

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

X = data[features]
y = data[target]

# Normalisasi Fitur (X) dan Target (y)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler() # Scaler untuk y
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)) # y juga di-scale

# Split Data yang sudah di-scale
X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.1, random_state=42
)
print("Data siap.")

# Buat dictionary untuk menyimpan hasil prediksi (dalam skala asli)
results = {}

# --- 2. Latih Model dan Buat Prediksi ---

# ----- Model 1: Random Forest -----
print("\nMelatih model Random Forest...")
params_rf = {
    'n_estimators': 221, 'max_depth': 33, 'min_samples_split': 8,
    'min_samples_leaf': 5, 'max_leaf_nodes': 121, 'max_features': 'sqrt'
}
model_rf = RandomForestRegressor(**params_rf, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train_scaled.ravel())
results['Random Forest'] = {
    'train_pred': scaler_y.inverse_transform(model_rf.predict(X_train).reshape(-1, 1)).flatten(),
    'test_pred': scaler_y.inverse_transform(model_rf.predict(X_test).reshape(-1, 1)).flatten()
}
print("Model Random Forest selesai.")

# ----- Model 2: XGBoost -----
print("Melatih model XGBoost dengan Early Stopping...")
params_xgb_revised = {
    'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 1000,
    'subsample': 0.974, 'colsample_bytree': 0.796, 'gamma': 0.232, 'min_child_weight': 7
}
model_xgb = XGBRegressor(**params_xgb_revised, random_state=42, n_jobs=-1)
model_xgb.fit(X_train, y_train_scaled, eval_set=[(X_test, y_test_scaled)], early_stopping_rounds=20, verbose=False)
print(f"Pelatihan XGBoost berhenti pada iterasi ke-{model_xgb.best_iteration}")
results['XGBoost'] = {
    'train_pred': np.maximum(scaler_y.inverse_transform(model_xgb.predict(X_train).reshape(-1, 1)).flatten(), 0),
    'test_pred': np.maximum(scaler_y.inverse_transform(model_xgb.predict(X_test).reshape(-1, 1)).flatten(), 0)
}
print("Model XGBoost selesai.")

# ----- Model 3: LSTM -----
print("Melatih model LSTM...")
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

params_lstm = {'units': 32, 'dropout': 0.0, 'learning_rate': 0.00308, 'epochs': 50}
model_lstm = Sequential([
    LSTM(units=params_lstm['units'], input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), dropout=params_lstm['dropout']),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')
model_lstm.fit(X_train_lstm, y_train_scaled, epochs=params_lstm['epochs'], batch_size=32, verbose=0)

train_pred_lstm_scaled = model_lstm.predict(X_train_lstm)
test_pred_lstm_scaled = model_lstm.predict(X_test_lstm)
results['LSTM'] = {
    'train_pred': np.maximum(scaler_y.inverse_transform(train_pred_lstm_scaled).flatten(), 0),
    'test_pred': np.maximum(scaler_y.inverse_transform(test_pred_lstm_scaled).flatten(), 0)
}
print("Model LSTM selesai.")

# --- 3. Hitung dan Tampilkan Metrik Evaluasi ---
# Kita perlu mengembalikan y_train_scaled dan y_test_scaled ke skala asli untuk evaluasi
y_train_orig = scaler_y.inverse_transform(y_train_scaled).flatten()
y_test_orig = scaler_y.inverse_transform(y_test_scaled).flatten()

print("\n" + "="*50)
print("HASIL EVALUASI KINERJA MODEL (DATA ME 48)")
print("="*50)
for model_name, preds in results.items():
    train_rmse = np.sqrt(mean_squared_error(y_train_orig, preds['train_pred']))
    train_mae = mean_absolute_error(y_train_orig, preds['train_pred'])
    train_r2 = r2_score(y_train_orig, preds['train_pred'])
    
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, preds['test_pred']))
    test_mae = mean_absolute_error(y_test_orig, preds['test_pred'])
    test_r2 = r2_score(y_test_orig, preds['test_pred'])
    
    print(f"\n===== Model: {model_name} =====")
    print(f"[Train] RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
    print(f"[Test ] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R²: {test_r2:.3f}")

print("="*50)
print("\nSelesai.")