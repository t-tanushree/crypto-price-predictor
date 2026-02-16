🚀 Cryptocurrency Price Prediction Web Application
📌 Project Overview

This is a production-ready full-stack machine learning web application that predicts the next closing price of Bitcoin using an LSTM (Long Short-Term Memory) deep learning model.

The system includes:

Data ingestion from Yahoo Finance

Data preprocessing pipeline

LSTM model training and saving

REST API built with Flask

Interactive frontend dashboard with Chart.js visualization

🧠 Tech Stack

Backend

Python

Flask

TensorFlow / Keras (LSTM)

Pandas & NumPy

Scikit-learn

yfinance

Frontend

HTML

CSS

JavaScript

Chart.js

⚙️ Features

Historical Bitcoin price visualization

Deep learning–based next-day price prediction

REST API endpoints:

/predict

/history

Modular backend architecture

Interactive UI dashboard

📂 Project Structure
crypto-price-predictor/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── data/
│   ├── models/
│   ├── utils/
│   ├── templates/
│   └── venv/
│
├── README.md
└── .gitignore

🚀 How To Run Locally
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py


Then open:

http://127.0.0.1:5000

🎯 Future Improvements

Multi-cryptocurrency support

Prediction confidence intervals

Deployment to cloud (Render)

Model performance metrics dashboard

👩‍💻 Author

Tanushree Tavakari
BCA Graduate | Aspiring Data & ML Engineer