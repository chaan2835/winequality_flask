import pandas as pd
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import os

# Load data for default values
data_path = os.path.join(os.path.dirname(__file__), 'data', 'WineQT.csv')
df = pd.read_csv(data_path)

# Features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the model for reuse
joblib.dump(model, 'wine_quality_model.pkl')

#streamlit app UI
st.title("Wine Quality Prediction")
st.write("Enter wine characteristics to predict quality:")

# Get feature names
features = df.drop('quality', axis=1).columns.tolist()

# Create input fields for each feature
input_data = []
for feature in features:
    value = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
    input_data.append(value)

# Predict button
if st.button("Predict Quality"):
    model = joblib.load('wine_quality_model.pkl')
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Wine Quality: {prediction}")