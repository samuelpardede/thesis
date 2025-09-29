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

# 3. Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.1, random_state=42
)

# 5. Inisialisasi dan Training Model
model = XGBRegressor(random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Prediksi & Evaluasi
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_preds_inv = scaler_y.inverse_transform(train_preds.reshape(-1, 1))
test_preds_inv = scaler_y.inverse_transform(test_preds.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train)
y_test_inv = scaler_y.inverse_transform(y_test)

print("=== Evaluasi Model XGBoost ===")
print(f"[Train] RMSE: {np.sqrt(mean_squared_error(y_train_inv, train_preds_inv)):.3f}")
print(f"        MAE: {mean_absolute_error(y_train_inv, train_preds_inv):.3f}")
print(f"        R² : {r2_score(y_train_inv, train_preds_inv):.3f}")
print(f"[Test ] RMSE: {np.sqrt(mean_squared_error(y_test_inv, test_preds_inv)):.3f}")
print(f"        MAE: {mean_absolute_error(y_test_inv, test_preds_inv):.3f}")
print(f"        R² : {r2_score(y_test_inv, test_preds_inv):.3f}")

# 7. Visualisasi
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Aktual')
plt.plot(test_preds_inv, label='Prediksi XGBoost')
plt.xlabel('Index')
plt.ylabel('Curah Hujan (mm)')
plt.title('Prediksi vs Aktual - XGBoost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
