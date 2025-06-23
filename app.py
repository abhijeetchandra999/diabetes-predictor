# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form and convert to float
    input_features = [float(x) for x in request.form.values()]
    
    # Convert to NumPy array
    input_array = np.array([input_features])
    
    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_array)
    
    # Predict using the model
    prediction = model.predict(input_scaled)
    output = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == '__main__':
    app.run(debug=True)
