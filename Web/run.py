from flask import Flask, request, render_template, url_for
import os
import glob
import subprocess
import threading
import time
from app.routes import manifacture_weather_data
from app.utils import start_drone
import requests
import dill
import sys
import pandas as pd
import concurrent.futures
from flask_socketio import SocketIO, emit, send
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__, static_url_path='', static_folder='app/static/', template_folder="app/templates/")
socketio = SocketIO(app)

app.config['SERVER_NAME'] = 'localhost:5000'  # Update with your server name and port
app.config['APPLICATION_ROOT'] = '/'  # Update with your application root
app.config['PREFERRED_URL_SCHEME'] = 'http'  # Update with your preferred URL scheme

drone_initialized = threading.Event()  # Event for synchronization

from app.routes import bp
app.register_blueprint(bp)

def load_object(file_path:sys):
    '''
    This function returns the object in certain file path.
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        print("Error is ", e)

def predict_fire_occasion(data):

    data = pd.DataFrame(data, index=[0])
    data = data.drop(columns=["burned_area"])
    data = data[["month","temperature","RH","wind_speed","day_night","daily_cdd","daily_hdd"]]

    src_path = "app/static/ml_model/"

    ml_model_path = src_path + "model.pkl"
    ml_preprocess_path = src_path + "preprocessor.pkl"

    ml_model = load_object(ml_model_path)
    ml_preprocess = load_object(ml_preprocess_path)

    preprocessed_data = ml_preprocess.transform(data)
    y_pred = ml_model.predict(preprocessed_data)

    return list(y_pred)

result = 0

@socketio.on('connect')
def background_task():
    try:
        while True:
            try:
                weather_data_response = requests.get("http://localhost:5001/api/weather")
                weather_data = weather_data_response.json()
            except Exception as e:
                raise Exception("Connection has been failed!") from e

            socketio.emit('weather-info', {'data': weather_data})

            result = predict_fire_occasion(weather_data).pop()
            result = int(result)

            socketio.emit('result', {'result': result})
            socketio.emit('connected', {'message': 'Connected!'})  # Emit 'connected' event

            break  # Exit the loop after initial actions
    finally:
        drone_initialized.set()


def initiate_drone():
    with app.app_context():
        drone_initialized.wait()

        print("Initiate drone function has been started.")
        socketio.emit('update_message', {'message': "Drone is preparing..."})

        diameter = 0.003
        altitude = 50
        num_points = 36

        frame, output, prob, (coordinate_lat, coordinate_lon, altitude) = start_drone(socketio, diameter, altitude, num_points)
        photo_path = glob.glob(os.path.join("app/static/data/predicted/", "*"))
        photo_path = photo_path[0].replace("app/static/", "")
        image_url = url_for('static', filename=photo_path)
        socketio.emit('output', {'output': output, 'prob': prob, 'lat': coordinate_lat, 'lon': coordinate_lon, 'photo_path': image_url})



@app.route("/")
def index():
    return render_template("index.html")




if __name__ == '__main__':

    socketio.start_background_task(target=background_task)
    socketio.start_background_task(target=initiate_drone)
    socketio.run(app, debug=True)
