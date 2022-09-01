import pandas as pd
import pickle
import json

# X_pred = pd.read_csv('AMEX_default_prediction/api/test_row.csv')
# def load_model():
#     model = pickle.load(open('pickles/pp_pred_pipe', 'rb'))
#     return model

# print(load_model().predict(X_pred))

# def predict(params):
#     model = pickle.load(open('pickles/pp_pred_pipe', 'rb'))
#     X_pred = pd.DataFrame.from_dict(params,orient='index').transpose()
#     prediction = model.predict(X_pred)[0]
#     pred_probability = model.predict_proba(X_pred)

#     if prediction == 1:
#         defaulter = 'defaulter'
#     else:
#         defaulter = 'payer'

#     return {'customer_ID':params['customer_ID'],
#             'output':defaulter,
#             'probability':round(pred_probability[0][1],3)}


# def predict(data):
#     #model = pickle.load(open('pickles/pp_pred_pipe', 'rb'))
#     param_data = json.loads(data)
#     # X_pred = pd.DataFrame.from_dict(data,orient='index').transpose()
#     # prediction = model.predict(X_pred)[0]
#     # pred_probability = model.predict_proba(X_pred)

#     # if prediction == 1:
#     #     defaulter = 'defaulter'
#     # else:
#     #     defaulter = 'payer'

#     return param_data
#     # return {'customer_ID':param_data['customer_ID'],
#     #         'output':defaulter,
#     #         'probability':round(pred_probability[0][1],3)}

data = pd.read_csv('AMEX_default_prediction/api/test_row.csv',index_col=0).to_dict(orient='records')[0]
print(data)
# param_dict = {"data": json.dumps(params)}
# print(predict(param_dict))
