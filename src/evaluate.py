import argparse
import os
import pandas as pd
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Dict
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

TARGET_COL = 'target'
METRICS_FILE = 'metrics.json'
CM_PLOT_FILE = 'confusion_matrix.png'
FEAT_IMP_PLOT_FILE = 'feature_importance.png'

@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@task
def load_artifact(path: str) -> Any:
    return joblib.load(path)

@task
def separate_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y

@task
def make_predictions(model: RandomForestClassifier, X: pd.DataFrame) -> tuple:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] # Probabilidad de la clase '1'
    return y_pred, y_proba

@task
def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }
    return metrics

@task(retries=3, retry_delay_seconds=5)
def save_metrics_json(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

@task
def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['Real 0', 'Real 1'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.savefig(path)
    plt.close()

@task
def plot_feature_importance(model: RandomForestClassifier, preprocessor: ColumnTransformer, path: str, top_n: int = 20):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Obtener nombres de features del preprocesador
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    
    feat_imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    # Graficar solo las top_n features
    top_features = feat_imp_series.head(top_n)
    
    plt.figure(figsize=(10, top_n / 3))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f'Top {top_n} Features más Importantes')
    plt.xlabel('Importancia')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

@flow(name="Model Evaluation Pipeline")
def evaluate_flow(args: argparse.Namespace):
    # rutas
    test_data_path = os.path.join(args.input_dir, "test.csv")
    model_path = os.path.join(args.model_dir, "model.pkl")
    preprocessor_path = os.path.join(args.model_dir, "preprocessor.pkl")
    
    metrics_json_path = os.path.join(args.output_dir, METRICS_FILE)
    cm_plot_path = os.path.join(args.output_dir, CM_PLOT_FILE)
    feat_imp_plot_path = os.path.join(args.output_dir, FEAT_IMP_PLOT_FILE)

    # cargar datos y artefactos
    df_test = load_data(test_data_path)
    model = load_artifact(model_path)
    preprocessor = load_artifact(preprocessor_path) # Necesario para feature importance

    # preparar datos
    X_test, y_test = separate_features_target(df_test, TARGET_COL)
    
    # evaluar
    y_pred, y_proba = make_predictions(model, X_test)
    metrics_dict = calculate_metrics(y_test, y_pred, y_proba)
    
    # guardar metricas y graficos
    save_metrics_json(metrics_dict, metrics_json_path)
    plot_confusion_matrix(y_test, y_pred, cm_plot_path)
    plot_feature_importance(model, preprocessor, feat_imp_plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa un modelo RandomForestClassifier entrenado.")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("model_dir", type=str)
    parser.add_argument("output_dir", type=str)

    args = parser.parse_args()
    evaluate_flow(args)