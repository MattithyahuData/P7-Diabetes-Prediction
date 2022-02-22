# Importing project dependencies 

# Model importing
import pickle

# Webapp creation and model use template rendering 
from flask import Flask,request, url_for, redirect, render_template # From flask import flask [Render-template refers to the html]

# Data handling
import pandas as pd

# Creating flask app / Initiating app
app = Flask(__name__)

# Load pickle model
model = pickle.load(open("svc_diabetes.pkl", "rb"))

# Define the home page
@app.route('/')
def hello_world():
    return render_template("index.html")

# route() decorator to tell Flask what URL should trigger our function.
@app.route('/predict',methods=['POST','GET'])
# Predict method
def predict():

    # Inputs from website 
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']
    
    # Inputs into dataframe 
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])
    

    # Finding the probability based on independent features
    prediction=model.predict_proba(row_df)

    # Formatting, selecting index 1 in predict_proba to get probability to churn or Churn == 1. Rounding to 2 d.p
    output=round(prediction[0][1], 2)

    # if output is greater than 50% proba
    if output> 0.5:

        # Converting output to string and adding % suffix
        output = str(float(output)*100)+'%'

        # Return if risk of diabetes is greater than 50%
        return render_template('result.html',pred=f'You have a higher risk of diabetes.\n\nYou have a {output} chance of having diabetes.')

    # Else if proba is less than or equal to 0.5 
    else:

        # Converting output to string and adding % suffix
        output = str(float(output)*100)+'%'

        # Return if risk of diabetes is less than or equal to 50%
        return render_template('result.html',pred=f'You have a lower risk of diabetes.\n\nYou have a {output} chance of having diabetes.')



# Only allowing file to run from this file 
if __name__ == '__main__':
    app.run(debug=True)
 