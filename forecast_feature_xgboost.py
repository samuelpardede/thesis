import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Persiapan Data ---
print("Mempersiapkan data AWS untuk FORECASTING...")
file_path = r"E:\STMKG\SKRIPSI\datahujan\fix\Data_AWS_Per_3_Jam_Outlier.xlsx"
try:
    data = pd.read_excel(file_path, engine='openpyxl')
except Exception as e:
    print(f"Error membaca file: {e}")
    exit()

# Preprocessing
features = ['tt_air_avg', 'rh_avg', 'ws_avg', 'pp_air']
target = 'rr'

data.replace([8888, 9999], np.nan, inplace=True)
data.interpolate(method='linear', inplace=True)
data.dropna(subset=features + [target], inplace=True)

# --- 2. RESTRUKTURISASI DATA UNTUK FORECASTING ---
df_forecast = data[features].copy()
df_forecast['target_RR'] = data[target].shift(-1)
df_forecast.dropna(inplace=True)

X = df_forecast[features]
y = df_forecast['target_RR']

# --- 3. Normalisasi & Pembagian Data Kronologis ---
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)) # y juga di-scale

# Bagi data yang sudah di-scale
split_percentage = 0.8
split_point = int(len(X_scaled) * split_percentage)

X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
y_train_scaled, y_test_scaled = y_scaled[:split_point], y_scaled[split_point:]
print(f"Data siap: {len(X_train)} data latih, {len(X_test)} data uji.")

# --- 4. Latih Model XGBoost dengan Early Stopping ---
print("\nMelatih model XGBoost untuk forecasting dengan Early Stopping...")
# Menggunakan hyperparameter hasil tuning forecasting
params_xgb = {
    'colsample_bytree': 0.690,
    'gamma': 0.322,
    'learning_rate': 0.044,
    'max_depth': 3,
    'min_child_weight': 8,
    'subsample': 0.807
}

# n_estimators diatur tinggi, biarkan early stopping yang menentukan jumlah terbaik
model = XGBRegressor(**params_xgb, n_estimators=2000, random_state=42, n_jobs=-1)

# Latih model dengan early stopping
model.fit(
    X_train,
    y_train_scaled,
    eval_set=[(X_test, y_test_scaled)],
    early_stopping_rounds=30,
    verbose=False
)
print(f"Pelatihan berhenti pada iterasi ke-{model.best_iteration}")

# --- 5. Hitung dan Visualisasi Feature Importance ---
print("Menghitung dan menampilkan Feature Importance...")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualisasi
plt.figure(figsize=(10, 6))
plt.barh(np.array(features)[indices], importances[indices], color='salmon')
plt.gca().invert_yaxis()
plt.title("Feature Importance - XGBoost Forecasting (AWS)", fontsize=16)
plt.xlabel("Tingkat Importance", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nSelesai.")