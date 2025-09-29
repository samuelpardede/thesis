import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Persiapan Data ---
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

# --- 2. RESTRUKTURISASI DATA UNTUK FORECASTING ---
df_forecast = data[features].copy()
df_forecast['target_RR'] = data[target].shift(-1)
df_forecast.dropna(inplace=True)

X = df_forecast[features]
y = df_forecast['target_RR']

# --- 3. Normalisasi & Pembagian Data Kronologis ---
# PERBAIKAN: Hanya menormalisasi X, biarkan y dalam skala aslinya
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Bagi data: X di-scale, y tetap asli
split_percentage = 0.8
split_point = int(len(X_scaled) * split_percentage)

X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
y_train, y_test = y.values[:split_point], y.values[split_point:]
print(f"Data siap: {len(X_train)} data latih, {len(X_test)} data uji.")

# --- 4. Latih Model XGBoost dengan Early Stopping ---
print("\nMelatih model XGBoost untuk forecasting...")
# Menggunakan hyperparameter hasil tuning forecasting
params_xgb = {
    'colsample_bytree': 0.888,
    'gamma': 0.469,
    'learning_rate': 0.0101, 
    'max_depth': 14,
    'min_child_weight': 10,
    'subsample': 0.721
}

model = XGBRegressor(**params_xgb, n_estimators=2000, random_state=42, n_jobs=-1)

# Latih model dengan early stopping, menggunakan y yang tidak di-scale
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=30,
    verbose=False
)
print(f"Pelatihan berhenti pada iterasi ke-{model.best_iteration}")

# --- 5. Prediksi dan Evaluasi ---
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Tidak perlu inversi skala untuk y
train_preds = np.maximum(train_preds, 0)
test_preds = np.maximum(test_preds, 0)

print("\n" + "="*50)
print("HASIL EVALUASI KINERJA MODEL XGBOOST (FORECASTING - ME 48)")
print("="*50)

# Evaluasi pada data training
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_mae = mean_absolute_error(y_train, train_preds)
train_r2 = r2_score(y_train, train_preds)
    
# Evaluasi pada data testing
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_mae = mean_absolute_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)
    
print(f"[Train] RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
print(f"[Test ] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R²: {test_r2:.3f}")

print("="*50)
print("\nSelesai.")