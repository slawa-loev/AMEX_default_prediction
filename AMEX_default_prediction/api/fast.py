from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# app.state.model = load_model()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
#)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
# @app.get("/predict")
# def predict(pickup_datetime: datetime,  # 2013-07-06 17:18:00
#             pickup_longitude: float,    # -73.950655
#             pickup_latitude: float,     # 40.783282
#             dropoff_longitude: float,   # -73.984365
#             dropoff_latitude: float,    # 40.769802
#             passenger_count: int):      # 1
#     """
#     we use type hinting to indicate the data types expected
#     for the parameters of the function
#     FastAPI uses this information in order to hand errors
#     to the developpers providing incompatible parameters
#     FastAPI also provides variables of the expected data type to use
#     without type hinting we need to manually convert
#     the parameters of the functions which are all received as strings
#     """

#     utc = pytz.utc
#     loc_dt=utc.localize(pd.to_datetime(pickup_datetime))
#     fmt='%Y-%m-%d %H:%M:%S UTC'
#     data=loc_dt.strftime(fmt)


#     input_data = {'key':["2013-07-06 17:18:00"],
#                 'pickup_datetime':data,
#                 'pickup_longitude':float(pickup_longitude),
#                 'pickup_latitude':float(pickup_latitude),
#                 'dropoff_longitude':float(dropoff_longitude),
#                 'dropoff_latitude':float(dropoff_latitude),
#                 'passenger_count':int(passenger_count)}

#     X_pred = pd.DataFrame(input_data)
#     X_processed = preprocess_features(X_pred)
#     y_pred = app.state.model.predict(X_processed)

#     return {'fare_amount': float(y_pred[0])}


@app.get("/")
def root():
    return {'output': 'defaulter',
            'probability': '0.999'}
