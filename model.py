import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib
import os
import dagshub
import mlflow
import streamlit as st
import json

# Prepare output directories
os.makedirs("model_output", exist_ok=True)
os.makedirs("log_metrics", exist_ok=True)

# Load data
df = pd.read_csv("D:\\mlops\\github-practises-mlops\\winequality_flask\\data\\WineQT.csv")

# Features and target
X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

# Model
model = RandomForestClassifier(random_state=35)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)

# Print
print(f"accuracy: {accuracy:.4f}")
print(f"f1 score: {f1:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall: {recall:.4f}")
print("Classification Report:\n", report)

# Save model to model_output/
model_path = os.path.join("model_output", "wine_quality_model.pkl")
joblib.dump(model, model_path)

# Save classification report and metrics to log_metrics/
with open(os.path.join("log_metrics", "classification_report.txt"), 'w') as f:
    f.write(report)

metrics_dict = {
    "accuracy": accuracy,
    "f1_score": f1,
    "precision": precision,
    "recall": recall
}
with open(os.path.join("log_metrics", "model_metrics.json"), 'w') as f:
    json.dump(metrics_dict, f, indent=4)

X_test.to_csv(os.path.join("log_metrics", "X_test.csv"), index=False)
y_test.to_csv(os.path.join("log_metrics", "y_test.csv"), index=False)

# MLflow and DagsHub tracking
dagshub.init(repo_name='wine-quality-check-flask', repo_owner='chaan2835', mlflow=True)

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("random_state", 35)
    mlflow.log_param("test_size", 0.3)
    mlflow.log_metrics(metrics_dict)

    mlflow.set_tag("author", "chaan2835")

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(os.path.join("log_metrics", "classification_report.txt"))
    mlflow.log_artifact(os.path.join("log_metrics", "model_metrics.json"))
    mlflow.log_artifact(os.path.join("log_metrics", "X_test.csv"))
    mlflow.log_artifact(os.path.join("log_metrics", "y_test.csv"))

# # Streamlit App
# st.title("Wine Quality Prediction")

# # Load the trained model
# model = joblib.load(model_path)

# # Get feature names
# features = X.columns.tolist()

# # Input fields
# input_data = []
# for feature in features:
#     value = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
#     input_data.append(value)

# # Predict button
# if st.button("Predict Quality"):
#     input_df = pd.DataFrame([input_data], columns=features)
#     prediction = model.predict(input_df)[0]
#     st.success(f"Predicted Wine Quality: {prediction}")
