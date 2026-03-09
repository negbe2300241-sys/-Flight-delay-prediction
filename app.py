from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
le_airline = pickle.load(open("airline_encoder.pkl", "rb"))
le_origin = pickle.load(open("origin_encoder.pkl", "rb"))
le_dest = pickle.load(open("dest_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    airline = request.form['airline']
    origin = request.form['origin']
    destination = request.form['destination']
    departure_hour = int(request.form['departure_hour'])
    distance = int(request.form['distance'])
    day = int(request.form['day'])
    
    # Encode categorical variables
    airline_enc = le_airline.transform([airline])[0]
    origin_enc = le_origin.transform([origin])[0]
    dest_enc = le_dest.transform([destination])[0]
    
    # Predict
    prediction = model.predict(np.array([[airline_enc, origin_enc, dest_enc, departure_hour, distance, day]]))
    
    result = "Delayed" if prediction[0] == 1 else "On Time"
    
    return render_template('index.html', prediction_text=f"Flight Status: {result}")

if __name__ == "__main__":
    app.run(debug=True)