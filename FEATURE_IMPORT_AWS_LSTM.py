import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import shap

# 1. Load & Preprocess Data
file_path = r"E:\STMKG\SKRIPSI\datahujan\fix\Data_AWS_Per_3_Jam_Outlier.xlsx"
data = pd.read_excel(file_path)

# Preprocessing
features = ['tt_air_avg', 'rh_avg', 'ws_avg', 'pp_air']
target = 'rr'

data.replace([8888, 9999], np.nan, inplace=True)
data.interpolate(method='linear', inplace=True)
data.dropna(subset=features + [target], inplace=True)

X = data[features].values
y = data[target].values.reshape(-1, 1)

# Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshape untuk LSTM [sampel, timesteps, fitur]
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 2. Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_scaled, test_size=0.2, random_state=42)

# 3. Bangun dan Latih Model LSTM dengan Hyperparameter Final
print("Membangun dan melatih model LSTM...")
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00702), loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)
print("Pelatihan model selesai.")

# 4. Analisis Feature Importance dengan SHAP
print("\nMenjalankan analisis SHAP...")

background_data = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

# --- PERUBAHAN DILAKUKAN DI SINI ---
# Mengganti DeepExplainer dengan GradientExplainer
explainer = shap.GradientExplainer(model, background_data)
# ------------------------------------

shap_values = explainer.shap_values(X_val)
print("Analisis SHAP selesai.")

# 5. Visualisasi Hasil SHAP
shap_values_2d = shap_values[0].reshape(-1, X_val.shape[2])
X_val_2d = X_val.reshape(-1, X_val.shape[2])

plt.figure()
shap.summary_plot(
    shap_values_2d, 
    X_val_2d, 
    feature_names=features,
    plot_type='bar',
    show=False
)
plt.title("Global Feature Importance - LSTM (AWS) via SHAP")
plt.tight_layout()
plt.show()