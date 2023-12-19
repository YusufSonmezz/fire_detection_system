from flask import Flask, request, render_template

app = Flask(__name__, template_folder="app/templates/")

from app.routes import bp
app.register_blueprint(bp)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':

    app.run(debug=True)