from flask import Flask, request, jsonify, render_template_string
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("D:/mlops/github-practises-mlops/winequality_flask/model_output/wine_quality_model.pkl")

# Home route
@app.route("/")
def home():
    return "Welcome to the Wine Quality Prediction API!"

# HTML Form route for browser testing
@app.route("/predict_form", methods=["GET", "POST"])
def predict_form():
    form_html = """
    <h2>Wine Quality Prediction Form</h2>
    <form method="post">
        {% for feature in features %}
            <label>{{ feature }}:</label>
            <input type="text" name="{{ feature }}"><br><br>
        {% endfor %}
        <input type="submit" value="Predict">
    </form>
    {% if prediction is not none %}
        <h3>Predicted Quality: {{ prediction }}</h3>
    {% endif %}
    """
    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    prediction = None
    if request.method == "POST":
        try:
            input_data = [float(request.form[feature]) for feature in features]
            prediction = int(model.predict([input_data])[0])
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(form_html, features=features, prediction=prediction)

# JSON prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        input_data = [data[feature] for feature in features]
        prediction = int(model.predict([input_data])[0])
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
