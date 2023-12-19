import random
import sys
import numpy as np
import pandas as pd
import dill

from flask import Flask, jsonify, request, Blueprint
from ctgan import CTGAN

bp = Blueprint('main', __name__)

def manifacture_weather_data():
    '''
    https://medium.com/the-research-nest/exploring-gans-to-generate-synthetic-data-ca48f8a4b518
    
    month = random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    temperature = round(random.uniform(0.0, 45.0), 2)
    relative_humidity = round(random.uniform(0.0, 100.0), 2)
    wind = round(random.uniform(0.0, 29.0), 2)
    rain = round(np.random.lognormal(0.317007, 1.257470), 2)

    return {'month': month, 'temp': temperature, 'RH': relative_humidity, 'wind': wind, 'rain': rain}'''

    generator = CTGAN.load("data_generator/generator.pkl")
    generated_sample = generator.sample(1)

    return generated_sample.to_dict(orient="records")[0]

@bp.route('/api/weather', methods=['GET'])
def get_weather_data():
    weather_data = manifacture_weather_data()

    return jsonify(weather_data)

def load_object(file_path:sys):
    '''
    This function returns the object in certain file path.
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        print("Error is ", e)

@bp.route('/predict', methods=['GET', 'POST'])
def predict_fire_occasion():

    data = request.get_json()

    data = pd.DataFrame(data, index=[0])
    data = data.drop(columns=["burned_area"])
    data = data[["month","temperature","RH","wind_speed","day_night","daily_cdd","daily_hdd"]]

    print(data)

    src_path = "app/static/ml_model/"

    ml_model_path = src_path + "model.pkl"
    ml_preprocess_path = src_path + "preprocessor.pkl"

    ml_model = load_object(ml_model_path)
    ml_preprocess = load_object(ml_preprocess_path)

    preprocessed_data = ml_preprocess.transform(data)
    y_pred = ml_model.predict(preprocessed_data)

    return list(y_pred)
    
