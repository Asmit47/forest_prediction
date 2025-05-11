from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


# Load the model (ensure the model file is in the same directory)
model = pickle.load(open('model/ridge_model.pkl', 'rb'))
scaler= pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        input_features = [
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['wind_speed']),
            float(request.form['rainfall']),
            float(request.form['fuel_moisture']),
            float(request.form['drought_code']),
            float(request.form['initial_spread']),
            float(request.form['build_up']),
            float(request.form['ffmc']),
            float(request.form['dmc'])
        ]
        
        # Pad the input features with zeros for the missing features (fwi and dsr)
        while len(input_features) < 12:
            input_features.append(0.0)
        
        # Scale the input features
        input_features_scaled = scaler.transform([input_features])
        
        # Make prediction
        prediction = model.predict(input_features_scaled)
        
        return render_template('predict.html', prediction_text=f'Predicted Fire Weather Index (FWI): {prediction[0]}')
    return render_template('index.html')

if __name__ == '__main__':
    # Use host='0.0.0.0' for deployment
    app.run(debug=True, host='0.0.0.0', port=1230)