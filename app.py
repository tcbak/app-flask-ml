from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Expects JSON array
    prediction = model.predict([np.array(data)])
    return jsonify({'prediction': prediction.tolist()})

@app.route('/')
def home():
    return "Welcome to App"
    
if __name__ == "__main__":
    app.run(host='0.0.0.0')
