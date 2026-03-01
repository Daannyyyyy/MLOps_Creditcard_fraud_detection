import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
from utils import load_data, split_features_target, preprocess_data

def main():
    mlflow.set_experiment("fraud-detection")
    df = load_data("data/creditcard.csv")
    X, y = split_features_target(df, target_column="Class")
    X = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]

    # --- Threshold tuning ---
    best_threshold = 0.5
    best_f1 = 0
    for t in np.arange(0.3, 0.95, 0.05):
        y_pred_t = (y_probs >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        print(f"Threshold: {t:.2f} | F1: {f1:.3f} | Precision: {precision_score(y_test, y_pred_t):.3f} | Recall: {recall_score(y_test, y_pred_t):.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\nBest threshold: {best_threshold:.2f} with F1: {best_f1:.3f}")

    # --- Log best threshold run ---
    y_pred = (y_probs >= best_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("threshold", round(best_threshold, 2))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", best_f1)
        mlflow.sklearn.log_model(model, "model", registered_model_name="fraud-detection-model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/fraud_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        print("Model and scaler saved successfully")
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()