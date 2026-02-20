import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import load_data, split_features_target, preprocess_data
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score


def main():
    df = load_data("data/creditcard.csv")
    print("Pipeline has started")
    X,y = split_features_target(df, target_column= "Class")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    X = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit(X_test)


    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    y_probs = model.predict_proba(X_test)[:,1]
    best_threshold = 0.9
    print("\n Threshold tuning results")
    for t in [0.9,0.8,0.7,0.6,0.5,0.4]:
        y_pred_thres = (y_probs>=t).astype(int)
        precision = precision_score(y_test,y_pred_thres)
        recall = recall_score(y_test, y_pred_thres)
        print("Threshold(t):", t)
        print("precision:", precision)
        print("Recall:", recall)
        print("-" * 20)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_model.pkl")
    print("Model saved Successfully")
    y_pred = (y_probs>=best_threshold).astype(int)



    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy score:",accuracy)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()