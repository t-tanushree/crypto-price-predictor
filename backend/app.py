from flask import Flask, jsonify, render_template
import pandas as pd
import os
from utils.predict import predict_next_price

app = Flask(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "BTC-USD.csv"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict():
    try:
        predicted_price = predict_next_price()

        return jsonify({
            "symbol": "BTC-USD",
            "predicted_price": round(predicted_price, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/history", methods=["GET"])
def history():
    df = pd.read_csv(DATA_PATH)

    # If Date column doesn't exist, use first column
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df = df[["Date", "Close"]]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()

    return jsonify({
        "dates": df["Date"].astype(str).tolist(),
        "prices": df["Close"].tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)
