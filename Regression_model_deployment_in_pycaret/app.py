from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np
import webbrowser

app = Flask(__name__)
#url = "http://127.0.0.1:5000"
#webbrowser.open(url, new =0)

model = load_model('dt_deploy_test1')
#cols = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
#       'ptratio', 'black', 'lstat']

cols = ['gender', 'married', 'dependents', 'education', 'self_employed',
       'applicantincome', 'coapplicantincome', 'loanamount',
       'loan_amount_term', 'credit_history', 'property_area']

@app.route('/')
def home():
    return render_template("home_classification.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    if int(data_unseen.loanamount) > 50000:
        print('Sorry , we dont offer loans more than 50000')
        return render_template('home_classification.html',pred='Sorry , we dont offer loans more than 50000')
    else:
        prediction = predict_model(model, data=data_unseen)
        prediction = int(prediction.Label[0])
        if prediction == 1:
            message = 'Approved'
            return render_template('home_classification.html',pred='Congratulations, your loan is  {}'.format(message))
        else:
            message = 'Rejected'
            return render_template('home_classification.html',pred='Sorry , you loan is {}'.format(message))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
#data_unseen = pd.DataFrame([['male','no','4','graduate','no','5000','10000','25000','300','1','urban']], columns = cols)
