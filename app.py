import os
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Safely resolve absolute path to C_df.pkl in the same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "C_df.pkl")

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", prediction=None, error="Model file not loaded.")
        
    try:
        hp = float(request.form["hp"])
        vol = float(request.form["vol"])
        sp = float(request.form["sp"])

        loghp = np.log(hp)
        logsp = np.log(sp)

        X = np.array([[loghp, vol, logsp]])

        pred = model.predict(X)[0]
        prediction = round(float(pred), 2)

        return render_template("index.html", prediction=prediction, hp=hp, vol=vol, sp=sp)
    except Exception as e:
        return render_template("index.html", prediction=None, error=str(e))