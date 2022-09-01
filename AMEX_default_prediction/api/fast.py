from fastapi import FastAPI
import pickle
from pandas import DataFrame
import json
import numpy as np

app = FastAPI()

@app.get("/predict")
def predict(data):
    model = pickle.load(open('pickles/pp_pred_pipe', 'rb'))
    param_data = json.loads(data)
    X_pred = DataFrame(param_data,index=[0]).replace('',np.nan)
    prediction = model.predict(X_pred)[0]
    pred_probability = model.predict_proba(X_pred)

    if prediction == 1:
        defaulter = 'defaulter'
    else:
        defaulter = 'payer'

    return {'customer_ID':param_data['customer_ID'],
            'output':defaulter,
            'probability':round(pred_probability[0][1],3)}


@app.get("/")
def root():
    return {'customer_ID':'abc123customer',
            'output': 'defaulter',
            'probability': '0.999'}
