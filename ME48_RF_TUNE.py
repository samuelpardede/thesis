import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint

# 1. Load data ME 48
file_path = r"E:\STMKG\SKRIPSI\datahujan\ME_48\Data_ME48_Bersih_Final.xlsx"
data = pd.read_excel(file_path)

# 2. Preprocessing minimal
data.replace([8888, 9999], np.nan, inplace=True)
data.interpolate(method='linear', inplace=True)
data.dropna(inplace=True)

features = [
    "CLOUD_LOW_TYPE_CL", "CLOUD_LOW_MED_AMT_OKTAS", "CLOUD_MED_TYPE_CM",
    "CLOUD_HIGH_TYPE_CH", "CLOUD_COVER_OKTAS_M", "LAND_COND",
    "PRESENT_WEATHER_WW", "TEMP_DEWPOINT_C_TDTDTD", "TEMP_DRYBULB_C_TTTTTT",
    "TEMP_WETBULB_C", "WIND_SPEED_FF", "RELATIVE_HUMIDITY_PC",
    "PRESSURE_QFF_MB_DERIVED", "PRESSURE_QFE_MB_DERIVED"
]
target = "RR"

X = data[features]
y = data[target]

# 3. Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 4. Split train/test (hanya untuk tuning; model final Anda bisa pakai seluruh data latih)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.1, random_state=42
)

# 5. Model dasar & ruang hyperparameter
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_dist = {
    'n_estimators'     : randint(100, 600),
    'max_depth'        : randint(5, 40),
    'min_samples_split': randint(2, 12),
    'min_samples_leaf' : randint(1, 10),
    'max_leaf_nodes'   : randint(50, 150),
    'max_features'     : ['sqrt', 'log2']
}

# 6. RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,            # jumlah kombinasi yang diuji
    cv=3,                 # 3‑fold cross‑validation
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train.ravel())

# 7. Tampilkan hyperparameter terbaik
print("\n=== Hyperparameter Terbaik Random Forest (ME 48) ===")
print(random_search.best_params_)
