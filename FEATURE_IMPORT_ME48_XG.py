import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Load Data
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
data = pd.read_excel(file_path)

# 2. Preprocessing
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

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# 4. Model XGBoost dengan Hyperparameter dari Tabel
# Parameter ini diperbarui sesuai dengan tabel yang Anda berikan
best_params = {
    'n_estimators': 373,
    'max_depth': 11,
    'learning_rate': 0.192,
    'subsample': 0.974,
    'colsample_bytree': 0.796,
    'gamma': 0.232,
    'min_child_weight': 7
}
model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# 6. Visualisasi
plt.figure(figsize=(10, 6))
plt.barh([features[i] for i in indices], [importances[i] for i in indices], color='salmon')
plt.gca().invert_yaxis()
plt.title("Feature Importance - XGBoost (ME 48)")
plt.xlabel("Importance")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()