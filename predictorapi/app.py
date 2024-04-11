from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from datetime import date
import time
import sys
import json

with open('../suggestions/suggestions.json', 'r') as f:
  data = json.load(f)

print(data)
app = Flask(__name__)
app.json.sort_keys = False


model = pickle.load(open('model.pkl', 'rb'))

# Default HTML home page for model testing with form input
@app.route("/")
def Home():
    print(f)
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The predicted crop is    {}".format(prediction))


# USE BELOW ENDPOINT FOR API UTILIZATION!

# /prediction endpoint with POST request
@app.route("/prediction", methods = ["POST"])
def prediction():
    
    '''
    To make a prediction, call the /prediction endpoint, with the following sample JSON body:
    
    {
        "n": 2,
        "p": 15,
        "k": 7,
        "temp": 100,
        "h": 20,
        "ph": 30,
        "rain": 55
    }
    
    '''
    
    
    # Wrapped inside a try catch for error detection and handling
    try:
        
        '''
        Data is retrieved from the request as JSON, converted to a list or array -> np array -> reshaped to be accepted by the model
        '''
        data = request.get_json()
        pred_features = list(data.values())
        pred_features = np.array(pred_features)
        pred_features = pred_features.reshape(1, -1)

        # Takes in the time and day of when the request was made
        dt = date.today().strftime("%d/%m/%Y")
        curr_time = time.strftime("%H:%M:%S", time.localtime())

        # Calling our model to make a prediction
        prediction = model.predict(pred_features)

        # JSON response of all above(prediction input, prediction output, time, date)
        json_out = {
            "prediction_input" : data,
            "predicted_crop" : prediction[0],
            "time" : curr_time,
            "date" : dt,
        }
        
        return jsonify(json_out)

    except Exception as e:
        return jsonify("Error in request: " + str(e))   

if __name__ == '__main__':
    app.run(debug=True)