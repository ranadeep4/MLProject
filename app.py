
from flask import Flask, render_template, request
import pickle
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        input1 = request.form['input1']
        input2 = request.form['input2']
        input1 = float(input1)
        input2 = float(input2)
        data = np.array([[(input1),(input2)]])
        dt_model = pickle.load(open('ML1.pkl','rb'))
        prediction = dt_model.predict_proba(data)
        
    return render_template('index.html',prediction=float(prediction[0][1]))

if __name__ == '__main__':
    app.run(debug=True)