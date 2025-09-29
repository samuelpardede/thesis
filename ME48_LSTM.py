import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

# 4. Ubah jadi format LSTM
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_scaled, test_size=0.2, random_state=42)

# 6. Model LSTM
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# 7. Evaluasi
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_preds_inv = scaler_y.inverse_transform(train_preds)
test_preds_inv = scaler_y.inverse_transform(test_preds)
y_train_inv = scaler_y.inverse_transform(y_train)
y_test_inv = scaler_y.inverse_transform(y_test)

print("=== Evaluasi Model LSTM ===")
print(f"[Train] RMSE: {np.sqrt(mean_squared_error(y_train_inv, train_preds_inv)):.3f}")
print(f"        MAE: {mean_absolute_error(y_train_inv, train_preds_inv):.3f}")
print(f"[Test ] RMSE: {np.sqrt(mean_squared_error(y_test_inv, test_preds_inv)):.3f}")
print(f"        MAE: {mean_absolute_error(y_test_inv, test_preds_inv):.3f}")

# 8. Visualisasi
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Aktual')
plt.plot(test_preds_inv, label='Prediksi LSTM')
plt.title('Prediksi vs Aktual (LSTM)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 9. Hindcast Interaktif
def hindcast_rainfall():
    print("\nMasukkan nilai semua variabel prediktor:")
    user_input = []
    for f in features:
        val = float(input(f"{f}: "))
        user_input.append(val)

    user_input_scaled = scaler_X.transform([user_input])
    input_lstm = user_input_scaled.reshape((1, 1, len(features)))
    pred_scaled = model.predict(input_lstm)
    pred_inv = scaler_y.inverse_transform(pred_scaled)

    print(f"\nPrediksi Curah Hujan: {pred_inv[0][0]:.2f} mm")

hindcast_rainfall()
