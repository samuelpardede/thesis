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

# 5. Model XGBoost dengan Hyperparameter Terbaik
best_params = {
    'colsample_bytree': 0.7962072844310213,
    'gamma': 0.23225206359998862,
    'learning_rate': 0.1922634555704315,
    'max_depth': 11,
    'min_child_weight': 7,
    'n_estimators': 373,
    'subsample': 0.9744427686266666
}

model = XGBRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 6. Prediksi dan Evaluasi
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

print("=== Evaluasi Model (XGBoost Tuned) ===")
print(f"[Train] RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
print(f"[Test ] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R²: {test_r2:.3f}")

# 7. Visualisasi Prediksi
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Aktual')
plt.plot(test_preds_inv, label='Prediksi (XGBoost)')
plt.xlabel('Data Index')
plt.ylabel('Curah Hujan (mm)')
plt.title('Prediksi vs Aktual Curah Hujan (XGBoost Tuned)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Hindcast Interaktif
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

# 9. Feature Importance
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

print("\n=== Feature Importances ===")
for idx in sorted_indices:
    print(f"{features[idx]}: {importances[idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(
    [features[i] for i in sorted_indices],
    [importances[i] for i in sorted_indices],
    color='salmon'
)
plt.xlabel('Importance')
plt.title('Feature Importance (XGBoost)')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
