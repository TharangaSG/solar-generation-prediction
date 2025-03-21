import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import RegressionAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_pickle("../../data/processed/02_feature_data.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df.columns
df_train = df.drop(["month"], axis=1)

X = df_train.drop(["Power_kW"], axis=1)
y = df_train["Power_kW"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
df.columns
basic_features = ['Board temperature_℃', 'Radiation intensity_w',
       'Wind pressure_kgm2', 'Top wind speed_MS-2', 'Low wind speed_MS-2',
       'station pressure', 'sea level pressure', 'temperature',
       'humidity', 'precipitation', 'cloud amount', 'irradiance']
pca_features = ["pca_1", "pca_2", "pca_3", "pca_4"]

print("Basic features:", len(basic_features))
print("PCA features:", len(pca_features))


feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + pca_features))


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Assuming `df` is your DataFrame
df_train = df.drop(["month"], axis=1)  # Remove 'month' column
X = df_train.drop(["Power_kW"], axis=1)  # Features
y = df_train["Power_kW"]  # Target variable

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.columns
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Ensure target variable is converted to numpy arrays
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

# Build and train the model
# def create_spfnet(n_layers, n_activation, kernels):
#     model = tf.keras.models.Sequential()
#     for i, nodes in enumerate(n_layers):
#         if i == 0:
#             model.add(Dense(nodes, kernel_initializer=kernels, activation=n_activation, input_dim=X_train_scaled.shape[1]))
#         else:
#             model.add(Dense(nodes, activation=n_activation, kernel_initializer=kernels))
#     model.add(Dense(1))  # Output layer for regression
#     model.compile(loss='mse', 
#                   optimizer='adam',
#                   metrics=[tf.keras.metrics.RootMeanSquaredError()])
#     return model

# # Define the model
# spfnet = create_spfnet([32, 64], 'relu', 'normal')

# # Train the model
# history = spfnet.fit(
#     X_train_scaled, y_train,
#     validation_data=(X_val_scaled, y_val),
#     epochs=50,
#     batch_size=32,
#     verbose=1
# )

# # Summarize the model
# spfnet.summary()
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def create_spfnet(n_layers, n_activation, kernels):
    model = Sequential()
    for i, nodes in enumerate(n_layers):
        if i == 0:
            model.add(Dense(nodes, kernel_initializer=kernels, activation=n_activation, input_dim=X_train_scaled.shape[1]))
        else:
            model.add(Dense(nodes, activation=n_activation, kernel_initializer=kernels))
    model.add(Dense(1))  # Output layer for regression
    
    model.compile(loss='mse', 
                  optimizer='adam',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Define the model
spfnet = create_spfnet([32, 64], 'relu', 'normal')

# Save the best model during training
save_dir = "../../models/"  # Change this to your desired location
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Save the best model during training
checkpoint_path = os.path.join(save_dir, "spfnet_best_model.keras")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model
history = spfnet.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[checkpoint]  # Add checkpoint callback
)

final_model_path = os.path.join(save_dir, "spfnet_final_model.keras")
spfnet.save(final_model_path)

print(f"Model saved to: {final_model_path}")

# Summarize the model
spfnet.summary()

import joblib
import os

# Define save directory
save_dir = "../../models/"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

# Save the fitted scaler
scaler_path = os.path.join(save_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")



val_loss, val_rmse, val_mae = spfnet.evaluate(X_val_scaled, y_val, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation MAE: {val_mae:.4f}")

import matplotlib.pyplot as plt

# Predict using the validation set
y_pred = spfnet.predict(X_val_scaled)

# Reshape predictions and true values if necessary
y_pred = y_pred.flatten()
y_true = y_val.flatten()

# Create a scatter plot or line plot to compare predictions and true values
plt.figure(figsize=(10, 6))

# Scatter plot for comparison
plt.scatter(range(len(y_true)), y_true, color='blue', label='True Values', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', alpha=0.6)

# Optional: Line plot (works better if the values are ordered)
# plt.plot(range(len(y_true)), y_true, label='True Values', color='blue')
# plt.plot(range(len(y_pred)), y_pred, label='Predicted Values', color='red')

# Add labels, title, and legend
plt.title('True vs Predicted Outputs')
plt.xlabel('Sample Index')
plt.ylabel('Power (kW)')
plt.legend()
plt.show()

# Predict using the validation set
y_pred = spfnet.predict(X_val_scaled)

# Reshape predictions and true values if necessary
y_pred = y_pred.flatten()
y_true = y_val.flatten()

# Create a line plot for comparison
plt.figure(figsize=(12, 6))

# Plot true values
plt.plot(y_true, label="True Values", color="blue", linewidth=2)

# Plot predicted values
plt.plot(y_pred, label="Predicted Values", color="red", linestyle="dashed", linewidth=2)

# Add labels, title, and legend
plt.title('True vs Predicted Outputs (Line Plot)', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Calculate and display summary statistics
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
