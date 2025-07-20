import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib
import os
import dagshub
import mlflow


# Load data
data_path = os.path.join(os.path.dirname(__file__), 'data', 'WineQT.csv')
df = pd.read_csv("D:\mlops\github-practises-mlops\winequality_flask\data\WineQT.csv")

# Features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

# Model
model = RandomForestClassifier(random_state=35)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
# model_path = os.path.join(os.path.dirname(__file__), 'wine_quality_model.pkl')
# joblib.dump(model, model_path)

# model metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precession = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Print model metrics
print(f"accuracy: {accuracy:.4f}")
print(f"f1 score: {f1:.4f}")
print(f"precision: {precession:.4f}")
print(f"recall: {recall:.4f}")

dagshub.init(repo_name='wine-quality-check-flask',repo_owner='chaan2835', mlflow=True)

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("random_state", 35)
    mlflow.log_param("test_size", 0.3)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precession)
    mlflow.log_metric("recall", recall)
    mlflow.set_tag("author", "chaan2835")
   
   
    #save and log the model
    report = classification_report(y_test, y_pred)
    joblib.dump(model, 'wine_quality_model.pkl')
    mlflow.log_artifact('wine_quality_model.pkl')

    # save and log classification report
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('classification_report.txt')

    #optional: log the model to DagsHub
    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    mlflow.log_artifact('X_test.csv')   
    mlflow.log_artifact('y_test.csv')