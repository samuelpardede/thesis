import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Load Data
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
data = pd.read_excel(file_path)

# 2. Preprocessing
data.replace([8888, 9999], np.nan, inplace=True)
features = [
    "CLOUD_LOW_TYPE_CL", "CLOUD_LOW_MED_AMT_OKTAS", "CLOUD_MED_TYPE_CM", "CLOUD_HIGH_TYPE_CH",
    "CLOUD_COVER_OKTAS_M", "LAND_COND", "PRESENT_WEATHER_WW", "TEMP_DEWPOINT_C_TDTDTD",
    "TEMP_DRYBULB_C_TTTTTT", "TEMP_WETBULB_C", "WIND_SPEED_FF", "RELATIVE_HUMIDITY_PC",
    "PRESSURE_QFF_MB_DERIVED", "PRESSURE_QFE_MB_DERIVED"
]
target = "RR"

for col in features + [target]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.interpolate(method='linear', inplace=True)
data.dropna(subset=features + [target], inplace=True)

X = data[features]
y = data[target]

# 3. Normalisasi & Split Data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.1, random_state=42
)

# 4. Model XGBoost dengan Penyesuaian & Early Stopping
params = {
    # Parameter yang disesuaikan untuk mengurangi overfitting
    'learning_rate': 0.1,  # Diturunkan agar model belajar lebih hati-hati
    'max_depth': 8,        # Dikurangi untuk membuat pohon lebih sederhana
    'n_estimators': 1000,  # Ditingkatkan untuk memberi ruang pada early stopping
    
    # Parameter regularisasi dari hasil tuning sebelumnya
    'colsample_bytree': 0.796,
    'gamma': 0.232,
    'min_child_weight': 7,
    'subsample': 0.974
}

model = XGBRegressor(**params, random_state=42, n_jobs=-1)

# Latih model dengan early stopping
print("Melatih model XGBoost dengan Early Stopping...")
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)], # Gunakan test set untuk memonitor performa
    early_stopping_rounds=20,     # Hentikan jika performa tidak membaik setelah 20 iterasi
    verbose=False                 # Set True jika ingin melihat prosesnya
)
print(f"Pelatihan berhenti pada iterasi ke-{model.best_iteration}")


# 5. Prediksi dan Evaluasi (menggunakan model terbaik dari early stopping)
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Inversi skala
train_preds_inv = scaler_y.inverse_transform(train_preds.reshape(-1, 1))
test_preds_inv = scaler_y.inverse_transform(test_preds.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train)
y_test_inv = scaler_y.inverse_transform(y_test)

# Hitung metrik
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_preds_inv))
train_mae = mean_absolute_error(y_train_inv, train_preds_inv)
train_r2 = r2_score(y_train_inv, train_preds_inv)

test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_preds_inv))
test_mae = mean_absolute_error(y_test_inv, test_preds_inv)
test_r2 = r2_score(y_test_inv, test_preds_inv)

print("\n=== Evaluasi Model (XGBoost dengan Early Stopping) ===")
print(f"[Train] RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
print(f"[Test ] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R²: {test_r2:.3f}")