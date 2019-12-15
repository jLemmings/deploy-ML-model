import numpy as np
from flask import Flask, request
import pickle
import xgboost
import os



app = Flask(__name__)
print("Loading Model")
model = pickle.load(open('data/model.pickle', 'rb'))
print("Done loading Model, waiting for requests")

@app.route('/')
def home_endpoint():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    print(prediction[0])

    output = round(prediction[0], 2)

    print("PREDICTION: ", output)
    return str(output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 33507))
    print("STARTING WEBSERVER PORT: ", port)
    app.run(host='127.0.0.1', port=port, threaded=True)