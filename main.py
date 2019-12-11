import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import xgboost

app = Flask(__name__)
model = pickle.load(open('data/model.pickle', 'rb'))

@app.route('/')
def home_endpoint():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    print("PREDICTION: ", output)
    return str(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)