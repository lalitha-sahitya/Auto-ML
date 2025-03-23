from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Define models
MODELS = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
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

    # Encode categorical target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate all models
    results = {}
    best_model = None
    best_accuracy = 0

    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
            joblib.dump(model, MODEL_PATH)

        results[model_name] = {
            "accuracy": round(accuracy * 100, 2),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

    return jsonify({
        "best_model": best_model,
        "results": results
    })

@app.route("/download_model", methods=["GET"])
def download_model():
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True)
    else:
        return jsonify({"error": "Model not found. Please train first!"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5001)
