import random
import numpy as np
from flask import Flask, jsonify, request, Blueprint

bp = Blueprint('main', __name__)

def manifacture_weather_data():
    month = random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    temperature = round(random.uniform(0.0, 45.0), 2)
    relative_humidity = round(random.uniform(0.0, 100.0), 2)
    wind = round(random.uniform(0.0, 29.0), 2)
    rain = round(np.random.lognormal(0.317007, 1.257470), 2)

    return {'month': month, 'temp': temperature, 'RH': relative_humidity, 'wind': wind, 'rain': rain}

@bp.route('/api/weather', methods=['GET'])
def get_weather_data():
    weather_data = manifacture_weather_data()

    return jsonify(weather_data)
