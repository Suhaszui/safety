import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib # MODIFIED: Correct way to import joblib

# --- Step 1: Load the Combined Dataset ---
try:
    # Make sure this CSV file is in your 'safety-backend' folder
    file_name = "cleaned_combined_data.csv" 
    combined_df = pd.read_csv(file_name)
    print("✅ Combined dataset loaded successfully!")
except FileNotFoundError:
    print(f"🔴 Error: The file '{file_name}' was not found in the safety-backend folder.")
    exit()

# --- Step 2: Prepare the Data for the AI Model ---
combined_df.dropna(inplace=True)

# Define the single feature (X) and the target (y)
features = ['crime_rate_per_capita'] 
X = combined_df[features]
y = combined_df['safety_score']

# --- Step 3: Train the AI Model with a Pipeline ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# --- Step 4: Evaluate and Predict ---
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- AI Model Performance ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# --- Step 5: Make a Prediction for a New District ---
new_data = pd.DataFrame([{'crime_rate_per_capita': 1000}])
predicted_score = pipeline.predict(new_data)

print("\n--- Example Prediction ---")
print(f"A district with a crime rate of 1000 has a predicted safety score of: {predicted_score[0]:.2f}")

# --- Step 6: Save the Trained Model ---
model_filename = 'safety_model_pipeline.joblib'
joblib.dump(pipeline, model_filename)

print(f"\n✅ Model saved successfully as '{model_filename}' in the current folder!")




    
