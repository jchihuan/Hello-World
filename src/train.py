import argparse
import os
import pandas as pd
import joblib
from typing import Any
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier

TARGET_COL = 'target'
MODEL_NAME = 'model.pkl'

@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@task
def separate_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y

@task
def train_model(X_train: pd.DataFrame, y_train: pd.Series, args: argparse.Namespace) -> RandomForestClassifier:    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1  # Usar todos los procesadores
    )
    
    model.fit(X_train, y_train)    
    return model

def save_model(model: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

@flow(name="Model Training Pipeline")
def train_flow(args: argparse.Namespace):
    # Rutas
    train_path = os.path.join(args.input_dir, "train.csv")
    model_path = os.path.join(args.model_dir, MODEL_NAME)

    # Pipeline
    df_train = load_data(train_path)
    X_train, y_train = separate_features_target(df_train, TARGET_COL)
    
    model = train_model(X_train, y_train, args)
    save_model(model, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo RandomForestClassifier.")    
    parser.add_argument("input_dir", type=str)
    parser.add_argument("model_dir", type=str)
    
    # Hiperparámetros (params.yaml)
    parser.add_argument("--n-estimators", type=int, default=100, help="Numero de arboles en el bosque")
    parser.add_argument("--max-depth", type=int, default=10, help="Profundidad maxima de los arboles")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")

    args = parser.parse_args()
    train_flow(args)