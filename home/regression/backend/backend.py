from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Define regression models
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Support Vector Machine": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor()
}

MODEL_PATH = "best_model.pkl"

@app.route('/train', methods=['POST'])
def train():
    file = request.files.get("file")
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Read dataset
    df = pd.read_csv(file)

    # Identify target column (last column)
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        X[categorical_cols] = X[categorical_cols].apply(lambda col: LabelEncoder().fit_transform(col))

    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    best_model_name = None
    best_score = float("inf")  # Minimize MSE

    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if mse < best_score:
            best_score = mse
            best_model_name = model_name
            joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)

        results[model_name] = {"Mean Squared Error": round(mse, 4), "R² Score": round(r2, 4)}

    return jsonify({"best_model": best_model_name, "results": results})

@app.route("/download_model", methods=["GET"])
def download_model():
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True)
    return jsonify({"error": "Model not found. Please train first!"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5002)  # Use port 5001 for classification

