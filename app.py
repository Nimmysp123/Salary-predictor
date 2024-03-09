from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models
loaded_model_lr = joblib.load('linear_regression_model.joblib')
loaded_model_rf = joblib.load('random_forest_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    years_experience = float(request.form['years_experience'])
    
    # Identify the model type from the query parameter
    model_type = request.args.get('model')

    # Debugging statements
    print(f"Received input: Years of Experience = {years_experience}, Model Type = {model_type}")

    # Make predictions using the specified model
    if model_type == 'lr':
        prediction = loaded_model_lr.predict(np.array([[years_experience]]))[0]
    else:
        return "Invalid model type"

    # Debugging statement
    print(f"Prediction: {prediction}")

    # Display the predictions on a new page
    return render_template('result.html', prediction=prediction, model_type=model_type)

if __name__ == '__main__':
    app.run(debug=True)
