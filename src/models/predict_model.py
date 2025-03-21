import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Define file paths
scaler_path = "../../models/scaler.pkl"
model_path = "../../models/spfnet_final_model.keras"

# Load the saved scaler
scaler = joblib.load(scaler_path)

# Load the trained model
model = load_model(model_path)

# Define new input data
# new_input = np.array([[27.3,79.5,0.09,2.72,0.35,999.3,1012.4,25.4,68.0,0.0,1.0,74.0,
# 0.03272739103434674,0.3259370852684888,-0.5456864731296713,-0.029936249878077564]])

new_input = np.array([[36.325,12.5,0.0,0.35,0.35,994.9,1007.8,29.2,79.0,4.0,
    9.0,115.0,-0.22079967620701818,-0.33008630190167276,-0.09238540752336476,-0.24817985963806385]])

# Scale the input data
new_input_scaled = scaler.transform(new_input)

# Make a prediction
prediction = model.predict(new_input_scaled)

# Print the predicted power output
print(f"Predicted Power Output: {prediction[0][0]:.2f} kW")

#0.02319558


##--------------------------use csv----------------------------------##
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Define file paths
scaler_path = "../../models/scaler.pkl"
model_path = "../../models/spfnet_final_model.keras"
input_csv_path = "../../data/processed/data_random.csv"  
output_csv_path = "../../data/predictions.csv"  

# Load the saved scaler and model
scaler = joblib.load(scaler_path)
model = load_model(model_path)

# Read input CSV
df = pd.read_csv(input_csv_path)
df = df.drop(["month"], axis=1)

# Ensure the correct features are used (modify if necessary)
features = ['Board temperature_℃', 'Radiation intensity_w', 'Wind pressure_kgm2', 
            'Top wind speed_MS-2', 'Low wind speed_MS-2', 'station pressure', 
            'sea level pressure', 'temperature', 'humidity', 'precipitation', 
            'cloud amount', 'irradiance', 'pca_1', 'pca_2', 'pca_3', 'pca_4']

# Process each row, predict, and store results
predictions = []

for index, row in df.iterrows():
    input_data = np.array(row[features]).reshape(1, -1)  # Convert row to 2D array
    input_scaled = scaler.transform(input_data)  # Scale input
    prediction = model.predict(input_scaled)  # Predict power output
    predictions.append(prediction[0][0])  # Store prediction

# Add predictions to the DataFrame
df["Predicted Power_kW"] = predictions

# Save predictions to a new CSV file
df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to: {output_csv_path}")
