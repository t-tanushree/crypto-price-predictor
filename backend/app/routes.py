from flask import Blueprint

main = Blueprint('main', __name__)

@main.route("/")
def home():
    return "Crypto Price Predictor Backend is Running (Modular)!"
