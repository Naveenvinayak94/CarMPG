
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model (must be in the same folder as this file)
model = joblib.load("C_df.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Raw inputs from the user
        hp = float(request.form["hp"])
        vol = float(request.form["vol"])
        sp = float(request.form["sp"])

        # Same feature engineering used during training
        loghp = np.log(hp)
        logsp = np.log(sp)

        # Order must match the columns the model was trained on
        X = np.array([[loghp, vol, logsp]])

        pred = model.predict(X)[0]
        prediction = round(float(pred), 2)

        return render_template("index.html", prediction=prediction,
                                hp=hp, vol=vol, sp=sp)
    except Exception as e:
        return render_template("index.html", prediction=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)