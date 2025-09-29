import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint

# 1. Memuat Data
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
data = pd.read_excel(file_path)

# 2. Preprocessing Data
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

# 4. Hyperparameter Tuning untuk Mengurangi Overfitting
print("Memulai proses hyperparameter tuning untuk Random Forest...")

# Tentukan ruang pencarian parameter yang lebih terkontrol
param_dist = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(5, 20),           # Mengurangi kedalaman maksimum pohon
    'min_samples_split': randint(10, 30),  # Menaikkan syarat minimum untuk split
    'min_samples_leaf': randint(5, 20),    # Menaikkan syarat minimum untuk daun
    'max_features': ['sqrt', 'log2']       # Menjaga keacakan fitur
}

# Inisialisasi model dan RandomizedSearchCV
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,  # Jumlah kombinasi yang akan dicoba
    cv=5,       # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Jalankan proses tuning
random_search.fit(X_train, y_train.ravel())

print("\n=== Hyperparameter Terbaik Hasil Tuning ===")
print(random_search.best_params_)

# 5. Prediksi dan Evaluasi dengan Model Terbaik
best_model = random_search.best_estimator_

train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

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

print("\n=== Evaluasi Model (Random Forest - Tuned untuk Overfitting) ===")
print(f"[Train] RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
print(f"[Test ] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R²: {test_r2:.3f}")