import random
import sys
import numpy as np
import pandas as pd
import dill
import os

from flask import Flask, jsonify, request, Blueprint
from ctgan import CTGAN

bp = Blueprint('main', __name__)

def manifacture_weather_data():
    # Generate two classes data randomly
    class_of_fire = random.choice([0, 1])

    file_path = "data_generator/generator.pkl"

    generator = CTGAN.load(file_path)
    generated_sample = generator.sample(1)

    while generated_sample["burned_area"].item() != class_of_fire:
        generated_sample = generator.sample(1)

    return generated_sample.to_dict(orient="records")[0]

@bp.route('/api/weather', methods=['GET'])
def get_weather_data():
    weather_data = manifacture_weather_data()

    return jsonify(weather_data)
    
