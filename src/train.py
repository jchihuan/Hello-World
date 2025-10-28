import argparse
import os
import pandas as pd
import joblib
from typing import Any
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Constantes globales
TARGET_COL = 'target'
MODEL_NAME = 'model.pkl'
REGISTERED_MODEL_NAME = 'cu_venta_random_forest'


# ----------------------------- #
#            TASKS              #
# ----------------------------- #

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
        n_jobs=-1  # Usa todos los nucleos disponibles
    )
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


# ----------------------------- #
#            FLOW               #
# ----------------------------- #

@flow(name="Model Training Pipeline")
def train_flow(args: argparse.Namespace):    
    # Configuracion de MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "random_forest_experiment"))

    with mlflow.start_run(run_name="train_random_forest"):
        # Registro de hiperparametros
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "seed": args.seed
        })

        # Rutas
        train_path = os.path.join(args.input_dir, "train.csv")
        model_path = os.path.join(args.model_dir, MODEL_NAME)

        # Ejecucion del pipeline
        df_train = load_data(train_path)
        X_train, y_train = separate_features_target(df_train, TARGET_COL)
        model = train_model(X_train, y_train, args)

        # Guardar modelo localmente
        save_model(model, model_path)

        # Log en MLflow con firma y ejemplo
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model( # type: ignore
            sk_model=model,
            name="model",
            input_example=X_train.head(1),
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME
        )

        # Registrar el artefacto del modelo
        mlflow.log_artifact(model_path, artifact_path="artifacts")

# ----------------------------- #
#          MAIN ENTRY           #
# ----------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo RandomForestClassifier.")
    parser.add_argument("input_dir", type=str, help="Directorio de entrada con train.csv")
    parser.add_argument("model_dir", type=str, help="Directorio donde guardar el modelo entrenado")

    # Hiperparametros
    parser.add_argument("--n-estimators", type=int, default=100, help="Numero de arboles en el bosque")
    parser.add_argument("--max-depth", type=int, default=10, help="Profundidad maxima de los arboles")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")

    args = parser.parse_args()
    train_flow(args)
