import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Load Data
file_path = r"E:\STMKG\SKRIPSI\datahujan\fix\Data_AWS_Per_3_Jam_Outlier.xlsx"
data = pd.read_excel(file_path)

# 2. Preprocessing
features = ['tt_air_avg', 'rh_avg', 'ws_avg', 'pp_air']
target = 'rr'

data.replace([8888, 9999], np.nan, inplace=True)
data.interpolate(method='linear', inplace=True)
data.dropna(subset=features + [target], inplace=True)

X = data[features]
y = data[target]

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Model
best_params = {
    'n_estimators': 451,
    'max_depth': 19,
    'min_samples_split': 8,
    'min_samples_leaf': 3,
    'max_leaf_nodes': 139,
    'max_features': 'sqrt'
}
model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# 6. Visualisasi
plt.figure(figsize=(8, 5))
plt.barh([features[i] for i in indices], [importances[i] for i in indices], color='skyblue')
plt.gca().invert_yaxis()
plt.title("Feature Importance - Random Forest (AWS)")
plt.xlabel("Importance")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
