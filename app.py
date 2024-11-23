from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np

# Create a Flask app
app = Flask(__name__)

# Load the saved model
model = load('insurance_dataset.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.form
    
    # Extract the input values
    age = int(data['age'])
    sex = int(data['sex'])
    bmi = float(data['bmi'])
    children = int(data['children'])
    smoker = int(data['smoker'])
    region = int(data['region'])
    
    # Create a numpy array of the input values
    features = np.array([[age, sex, bmi, children, smoker, region]])
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Return the result as a JSON response
    return render_template('index.html', prediction_text='Estimated Insurance Charges: ${:.2f}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
