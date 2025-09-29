import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import shap

# 1. Load & Preprocess Data
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
data = pd.read_excel(file_path)

# Preprocessing
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
print("Membangun dan melatih model LSTM untuk data ME 48...")

# Definisikan hyperparameter dari tabel untuk ME 48
final_units = 32
final_dropout = 0.0
final_lr = 0.00308
final_epochs = 50

# Bangun arsitektur model
model = Sequential([
    LSTM(units=final_units, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=final_dropout),
    Dense(1)
])

# Kompilasi model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=final_lr),
    loss='mse'
)

# Latih model
model.fit(X_train, y_train, epochs=final_epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)
print("Pelatihan model selesai.")

# 4. Analisis Feature Importance dengan SHAP
print("\nMenjalankan analisis SHAP...")

background_data = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

# Gunakan GradientExplainer untuk menghindari error
explainer = shap.GradientExplainer(model, background_data)

shap_values = explainer.shap_values(X_val)
print("Analisis SHAP selesai.")

# 5. Visualisasi Hasil SHAP
# Meratakan data 3D dari LSTM menjadi 2D untuk visualisasi
shap_values_2d = shap_values[0].reshape(-1, X_val.shape[2])
X_val_2d = X_val.reshape(-1, X_val.shape[2])

# Buat plot ringkasan (bar chart)
plt.figure()
shap.summary_plot(
    shap_values_2d, 
    X_val_2d, 
    feature_names=features,
    plot_type='bar',
    show=False
)
plt.title("Global Feature Importance - LSTM (ME 48) via SHAP")
plt.tight_layout()
plt.show()

# Buat plot ringkasan (dot plot) untuk melihat distribusi pengaruh
plt.figure()
shap.summary_plot(
    shap_values_2d,
    X_val_2d,
    feature_names=features,
    show=False
)
plt.title("Distribusi Pengaruh Fitur - LSTM (ME 48) via SHAP")
plt.tight_layout()
plt.show()