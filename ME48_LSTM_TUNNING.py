import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt

# 1. Load & Preprocess Data
file_path = "E:/STMKG/SKRIPSI/datahujan/ME_48/Data_ME48_Bersih_Final.xlsx"
data = pd.read_excel(file_path)
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

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_scaled, test_size=0.2, random_state=42)

# 2. Build Model Function
def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', 32, 128, step=32),
        input_shape=(X_train.shape[1], X_train.shape[2]),
        dropout=hp.Float('dropout', 0.0, 0.4, step=0.1)
    ))
    model.add(Dense(1))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')
        ),
        loss='mse'
    )
    return model

# 3. Tuner Setup
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    overwrite=True,
    directory='tuning_lstm',
    project_name='rainfall'
)

# 4. Jalankan Tuning (gunakan batch_size tetap)
tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 5. Print Best Hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n=== Hyperparameter Terbaik ===")
print(f"Units: {best_hps.get('units')}")
print(f"Dropout: {best_hps.get('dropout')}")
print(f"Learning Rate: {best_hps.get('lr')}")
