import pandas as pd
from sklearn.preprocessing import StandardScaler
import os 
import joblib

def load_data(path):
    df = pd.read_csv(path)
    print("Shape of the dataset:", df.shape)
    print("Dataset is loaded Successfully")
    return df

def split_features_target(df, target_column):
    X = df.drop(columns = [target_column])
    y = df[target_column]
    print("Features and Target are fuckin' separated")
    return X,y


def preprocess_data(X, scaler=None, training=True):
    X = X.copy()

    if "Time" in X.columns:
        X.drop(columns=["Time"], inplace=True)

    if "Amount" in X.columns:
        if training:
            scaler = StandardScaler()
            X["Amount"] = scaler.fit_transform(X[["Amount"]])
            os.makedirs("models", exist_ok=True)
            joblib.dump(scaler, "models/scaler.pkl")
        else:
            X["Amount"] = scaler.transform(X[["Amount"]])

    return X
