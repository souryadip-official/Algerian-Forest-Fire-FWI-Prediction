import pickle as pck
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, redirect, jsonify

application = Flask(__name__)
app = application

# Now, we need to import the model and the standard scaler pickle files
with open('models/proj1_model.pkl', 'rb') as file:
    model = pck.load(file)

with open('models/scaler.pkl', 'rb') as file:
    scaler: StandardScaler = pck.load(file)

# Creating the home route
@app.route('/')
def home():
    return render_template('home.html') # Flask already knows to look inside the templates/ folder

@app.route('/predict', methods = ['GET', 'POST'])
def details():
    if request.method.lower() == 'get':
        return render_template('details.html')
    else:
        Temperature = int(request.form.get('Temperature'))
        RH = int(request.form.get('RH'))
        Ws = int(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))
        
        new_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        new_data_scaled = scaler.transform(new_data)
        y_new_pred = model.predict(new_data_scaled)[0].round(4)
        
        return render_template('result.html', prediction = y_new_pred)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')