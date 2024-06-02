## Import all necessary
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

application =Flask(__name__)
app =application

## Import the model and standard scaler 
logistic_model =pickle.load(open("models/logistic_clf.pkl","rb"))
standardScaler =pickle.load(open("models/Standard_scaler.pkl","rb"))


## Go to home or index page
@app.get("/index")
def indexPage():
    return render_template("index.html")

@app.route("/index/predictdata",methods=["GET","POST"])
def predict_diabetes():
    ## define var to store result
    result =""
    ## Check the method
    if request.method == "POST":
        ## Take data from user
        Pregnancies =float(request.form.get("Pregnancies"))
        Glucose =float(request.form.get("Glucose"))
        BloodPressure =float(request.form.get("BloodPressure"))
        SkinThickness =float(request.form.get("SkinThickness"))
        Insulin =float(request.form.get("Insulin"))
        BMI =float(request.form.get("BMI"))
        DiabetesPedigreeFunction =float(request.form.get("DiabetesPedigreeFunction"))
        Age =float(request.form.get("Age"))
        
        ## Scale the data
        
        New_scale_data =standardScaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        
        ## Predict data
        
        predict =logistic_model.predict(New_scale_data)
        
        ## Logic to find person is diabetic or not
        if(predict[0] == 1):
            result ="Diabetics"
            
        else:
            result ="Non Diabetics"
        
        
        return render_template("single_prediction.html",result =result) 
    else:
        return render_template("home.html")
        


if __name__ =="__main__":
    app.run(host="0.0.0.0",debug=True)