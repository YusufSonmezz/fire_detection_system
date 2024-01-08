from flask import Flask, request, render_template
import subprocess
import threading
from app.routes import manifacture_weather_data
import requests

app = Flask(__name__)

from app.routes import bp
app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
