import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

# --- 3. PEMBAGIAN DATA & NORMALISASI (PERBAIKAN) ---
print("\nMembagi dan menormalisasi data dengan benar...")
split_percentage = 0.8
split_point = int(len(X) * split_percentage)

# 3.1. Bagi data mentah menjadi set latih dan uji TERLEBIH DAHULU
X_train_raw, X_test_raw = X.iloc[:split_point], X.iloc[split_point:]
y_train_raw, y_test_raw = y.iloc[:split_point], y.iloc[split_point:]

# 3.2. Latih scaler HANYA pada data latih untuk mencegah data leakage
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Lakukan `fit_transform` pada data latih
X_train = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1))

# Lakukan `transform` saja pada data uji menggunakan scaler yang sudah dilatih
X_test = scaler_X.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw.values.reshape(-1, 1))

print(f"Data siap: {len(X_train)} data latih, {len(X_test)} data uji.")

# --- 4. Latih Model XGBoost dengan Early Stopping yang BENAR (PERBAIKAN) ---
print("\nMelatih model XGBoost dengan set validasi terpisah...")

# 4.1. Buat set validasi dari data latih (bukan dari data uji)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train_scaled, test_size=0.2, shuffle=False
)

# 4.2. Konfigurasi dan latih model
params_xgb = {
    'colsample_bytree': 0.888,
    'gamma': 0.469,
    'learning_rate': 0.010,
    'max_depth': 14,
    'min_child_weight': 5,
    'subsample': 0.721,
    'n_estimators': 1000,
    'random_state': 42,
    'n_jobs': -1
}
model = XGBRegressor(**params_xgb)

# Latih model dengan eval_set yang benar
model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=False
)
print(f"Pelatihan berhenti pada iterasi ke-{model.best_iteration}")

# --- 5. EVALUASI MODEL PADA DATA UJI ---
print("\nMengevaluasi model pada data uji...")
y_pred_scaled = model.predict(X_test)

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# --- 6. VISUALISASI FEATURE IMPORTANCE ---
print("\nMembuat visualisasi Feature Importance...")
plt.figure(figsize=(10, 8)) # Ukuran disesuaikan untuk satu plot

importances = model.feature_importances_
indices = np.argsort(importances) # Urutkan dari yang terkecil ke terbesar
plt.barh(np.array(features)[indices], importances[indices], color='dodgerblue')
plt.title("Feature Importance - XGBoost Forecasting (ME 48)", fontsize=16)
plt.xlabel("Tingkat Kepentingan (Importance)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nSelesai.")