import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Persiapan Data ---
print("Mempersiapkan data AWS untuk FORECASTING...")
file_path = r"E:\STMKG\SKRIPSI\datahujan\fix\Data_AWS_Per_3_Jam_Outlier.xlsx"
try:
    data = pd.read_excel(file_path, engine='openpyxl')
    print("File Excel berhasil dimuat.")
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
X_scaled = scaler_X.fit_transform(X)

# Bagi data: X di-scale, y tetap asli
split_percentage = 0.8
split_point = int(len(X_scaled) * split_percentage)

X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
y_train, y_test = y.values[:split_point], y.values[split_point:]
print(f"Data siap: {len(X_train)} data latih, {len(X_test)} data uji.")

# --- 4. Latih Model XGBoost dengan Early Stopping ---
print("\nMelatih model XGBoost untuk forecasting dengan Early Stopping...")
# Menggunakan hyperparameter yang Anda berikan
params_xgb = {
    'colsample_bytree': 0.690,
    'gamma': 0.323,
    'learning_rate': 0.05, 
    'max_depth': 3,
    'min_child_weight': 8, # Diperbarui
    'subsample': 0.807
}

model = XGBRegressor(**params_xgb, n_estimators=2000, random_state=42, n_jobs=-1)

# Latih model dengan early stopping
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

train_preds = np.maximum(train_preds, 0)
test_preds = np.maximum(test_preds, 0)

print("\n" + "="*50)
print("HASIL EVALUASI KINERJA MODEL XGBOOST (FORECASTING - AWS)")
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