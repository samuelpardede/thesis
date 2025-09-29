import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Persiapan Data untuk Forecasting ---
print("Mempersiapkan data AWS untuk FORECASTING...")
file_path = r"E:\STMKG\SKRIPSI\datahujan\fix\Data_AWS_Per_3_Jam_Outlier.xlsx"
try:
    data = pd.read_excel(file_path, engine='openpyxl')
    print("File Excel berhasil dimuat.")
except Exception as e:
    print(f"Terjadi error saat membaca file Excel: {e}")
    exit()

# Preprocessing
features = ['tt_air_avg', 'rh_avg', 'ws_avg', 'pp_air']
target = 'rr'
data.replace([8888, 9999], np.nan, inplace=True)
data.interpolate(method='linear', inplace=True)
data.dropna(subset=features + [target], inplace=True)

# Restrukturisasi Data
X_data = data[features]
y_data = data[target].shift(-1)
X_data = X_data[:-1]
y_data = y_data[:-1]

# --- Metodologi yang Benar: Pembagian Data, Normalisasi, dan Sekuens ---
print("\nMembagi dan menormalisasi data dengan benar...")
split_percentage = 0.8
split_point = int(len(X_data) * split_percentage)
X_train_raw, X_test_raw = X_data[:split_point], X_data[split_point:]
y_train_raw, y_test_raw = y_data[:split_point], y_data[split_point:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1))
X_test_scaled = scaler_X.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw.values.reshape(-1, 1))

def create_sequences(X_data, y_data, time_steps=8):
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        Xs.append(X_data[i:(i + time_steps)])
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 8
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_seq, y_train_seq, test_size=0.2, shuffle=False
)
print(f"Data sekuensial siap: {len(X_train_final)} data latih, {len(X_val)} data validasi, {len(X_test_seq)} data uji.")

results = {}

# --- 2. Latih Model dan Buat Prediksi ---

# --- PERUBAHAN STRATEGI INPUT ---
# Untuk RF & XGBoost, kita hanya gunakan langkah waktu terakhir dari sekuens (data 2D)
# Ini membuat inputnya lebih sederhana dan cocok untuk model pohon
X_train_final_2D = X_train_final[:, -1, :]
X_val_2D = X_val[:, -1, :]
X_test_2D = X_test_seq[:, -1, :]

# ----- Model 1: Random Forest (dengan input 2D) -----
print("\nMelatih model Random Forest...")
# Menggunakan hyperparameter forecasting AWS Anda
params_rf = {'max_depth': 6, 'max_features': 'log2', 'max_leaf_nodes': 40, 'min_samples_leaf': 5, 'min_samples_split': 21, 'n_estimators': 413}
model_rf = RandomForestRegressor(**params_rf, random_state=42, n_jobs=-1)
model_rf.fit(X_train_final_2D, y_train_final.ravel())
results['Random Forest'] = {'test_pred': scaler_y.inverse_transform(model_rf.predict(X_test_2D).reshape(-1, 1)).flatten()}
print("Model Random Forest selesai.")

# ----- Model 2: XGBoost (dengan input 2D) -----
print("Melatih model XGBoost dengan Early Stopping...")
params_xgb = {'colsample_bytree': 0.690, 'gamma': 0.322, 'learning_rate': 0.044, 'max_depth': 3, 'min_child_weight': 8, 'n_estimators': 739, 'subsample': 0.807}
model_xgb = XGBRegressor(**params_xgb, random_state=42, n_jobs=-1)
model_xgb.fit(X_train_final_2D, y_train_final.ravel(), eval_set=[(X_val_2D, y_val)], early_stopping_rounds=30, verbose=False)
print(f"Pelatihan XGBoost berhenti pada iterasi ke-{model_xgb.best_iteration}")
results['XGBoost'] = {'test_pred': np.maximum(scaler_y.inverse_transform(model_xgb.predict(X_test_2D).reshape(-1, 1)).flatten(), 0)}
print("Model XGBoost selesai.")

# ----- Model 3: LSTM (tetap dengan input sekuens 3D) -----
print("Melatih model Stacked LSTM...")
params_lstm = {'units': 128, 'dropout': 0.2, 'learning_rate': 0.00534}
model_lstm = Sequential([
    LSTM(units=params_lstm['units'], return_sequences=True, input_shape=(X_train_final.shape[1], X_train_final.shape[2])),
    Dropout(params_lstm['dropout']),
    LSTM(units=int(params_lstm['units'] / 2)),
    Dropout(params_lstm['dropout']),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(learning_rate=params_lstm['learning_rate']), loss='mse')
early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)
model_lstm.fit(X_train_final, y_train_final, epochs=100, batch_size=32, verbose=0,
               validation_data=(X_val, y_val),
               callbacks=[early_stopper])
test_pred_lstm_scaled = model_lstm.predict(X_test_seq)
results['LSTM'] = {'test_pred': np.maximum(scaler_y.inverse_transform(test_pred_lstm_scaled).flatten(), 0)}
print("Model LSTM selesai.")

# --- 3. Visualisasi Scatter Plot Aktual vs Prediksi ---
print("\nMembuat Scatter Plot...")
y_test_orig = scaler_y.inverse_transform(y_test_seq).flatten()

plot_df = pd.DataFrame({
    'Aktual': y_test_orig,
    'Prediksi RF': results['Random Forest']['test_pred'],
    'Prediksi XGBoost': results['XGBoost']['test_pred'],
    'Prediksi LSTM': results['LSTM']['test_pred']
})

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
fig.suptitle('Scatter Plot: Aktual vs. Prediksi - Forecasting (Data AWS)', fontsize=18)
models_to_plot = {"Random Forest": ('Prediksi RF', 'green'), "XGBoost": ('Prediksi XGBoost', 'red'), "LSTM": ('Prediksi LSTM', 'blue')}

for i, (model_name, (pred_col, color)) in enumerate(models_to_plot.items()):
    ax = axes[i]
    ax.scatter(plot_df['Aktual'], plot_df[pred_col], alpha=0.5, color=color, edgecolors='k', s=50)
    ax.set_title(model_name, fontsize=14)
    ax.set_xlabel("Curah Hujan Aktual (mm)", fontsize=12)
    if i == 0:
        ax.set_ylabel("Curah Hujan Prediksi (mm)", fontsize=12)
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    min_val = np.min([xlims[0], ylims[0]])
    max_val = np.max([xlims[1], ylims[1]])
    lims = [min_val, max_val]

    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Prediksi Sempurna')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
print("Selesai.")