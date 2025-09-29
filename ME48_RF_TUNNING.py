import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Memuat Data
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
data = pd.read_excel(file_path)

# 2. Preprocessing Data
data.replace([8888, 9999], np.nan, inplace=True)

features = [
    "CLOUD_LOW_TYPE_CL",
    "CLOUD_LOW_MED_AMT_OKTAS",
    "CLOUD_MED_TYPE_CM",
    "CLOUD_HIGH_TYPE_CH",
    "CLOUD_COVER_OKTAS_M",
    "LAND_COND",
    "PRESENT_WEATHER_WW",
    "TEMP_DEWPOINT_C_TDTDTD",
    "TEMP_DRYBULB_C_TTTTTT",
    "TEMP_WETBULB_C",
    "WIND_SPEED_FF",
    "RELATIVE_HUMIDITY_PC",
    "PRESSURE_QFF_MB_DERIVED",
    "PRESSURE_QFE_MB_DERIVED"
]
target = "RR"

for col in features + [target]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.interpolate(method='linear', inplace=True)
data.dropna(subset=features + [target], inplace=True)

# 3. Split Data
X = data[features]
y = data[target]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.1, random_state=42
)

# 4. Inisialisasi Model dengan Hyperparameter Terbaik
best_params = {
    'max_depth': 33,
    'max_features': 'sqrt',
    'max_leaf_nodes': 121,
    'min_samples_leaf': 5,
    'min_samples_split': 8,
    'n_estimators': 221
}
model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
model.fit(X_train, y_train.ravel())

# 5. Prediksi & Evaluasi
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_preds_inv = scaler_y.inverse_transform(train_preds.reshape(-1, 1))
test_preds_inv = scaler_y.inverse_transform(test_preds.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train)
y_test_inv = scaler_y.inverse_transform(y_test)

train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_preds_inv))
train_mae = mean_absolute_error(y_train_inv, train_preds_inv)
train_r2 = r2_score(y_train_inv, train_preds_inv)

test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_preds_inv))
test_mae = mean_absolute_error(y_test_inv, test_preds_inv)
test_r2 = r2_score(y_test_inv, test_preds_inv)

print("=== Evaluasi Model (Tuned Random Forest) ===")
print(f"[Train] RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
print(f"[Test ] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R²: {test_r2:.3f}")

# 6. Visualisasi Prediksi vs Aktual
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Aktual')
plt.plot(test_preds_inv, label='Prediksi (Tuned)')
plt.xlabel('Data Index')
plt.ylabel('Curah Hujan (mm)')
plt.title('Prediksi vs Aktual Curah Hujan (Tuned Random Forest)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Hindcast Interaktif
def hindcast_rainfall():
    print("\nMasukkan nilai semua variabel prediktor:")
    user_input = []
    for f in features:
        val = float(input(f"{f}: "))
        user_input.append(val)

    user_input_scaled = scaler_X.transform([user_input])
    pred_scaled = model.predict(user_input_scaled)
    pred_inv = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

    print(f"\nPrediksi Curah Hujan: {pred_inv[0][0]:.2f} mm")

hindcast_rainfall()

# 8. Feature Importance
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

print("\n=== Feature Importances ===")
for idx in sorted_indices:
    print(f"{features[idx]}: {importances[idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(
    [features[i] for i in sorted_indices],
    [importances[i] for i in sorted_indices],
    color='lightgreen'
)
plt.xlabel('Importance')
plt.title('Feature Importance (Tuned Random Forest)')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
