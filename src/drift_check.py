import argparse
import os
import json
import pandas as pd
import mlflow
from prefect import flow, task
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

DRIFT_JSON = "data_drift.json"
DRIFT_HTML = "data_drift.html"

# ----------------------------- #
#            TASKS              #
# ----------------------------- #

@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@task
def generate_drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = train_df.select_dtypes(include=["number", "category"])
    test_df = test_df.select_dtypes(include=["number", "category"])

    schema = DataDefinition(
        numerical_columns=["p_codmes", 
                           "monto", 
                           "prm_sldvigrstsf12m",
                           "rec_campecs06m", 
                           "min_difsaltottcr12m", 
                           "prm_pctusosaltottcr03m"],
        categorical_columns=["target"],
    )

    eval_ref = Dataset.from_pandas(pd.DataFrame(train_df), data_definition=schema)
    eval_cur = Dataset.from_pandas(pd.DataFrame(test_df), data_definition=schema)

    report = Report([
        DataDriftPreset(method="kl_div")
    ])
    snapshot = report.run(eval_cur, eval_ref)
    
    json_path = os.path.join(output_dir, "data_drift.json")
    html_path = os.path.join(output_dir, "data_drift.html")
    snapshot.save_json(json_path)
    snapshot.save_html(html_path)
    return json_path

@task
def extract_drift_summary(json_path: str) -> dict:
    import json
    with open(json_path, "r") as f:
        data = json.load(f)

    drift_info = {
        "share_drifted_features": 0.0,
        "n_drifted_features": 0,
        "n_features": 0,
        "dataset_drift": False,
    }

    metrics = data.get("metrics", [])

    for metric in metrics:
        metric_id = metric.get("metric_id", "")
        value = metric.get("value", {})

        if metric_id.startswith("DriftedColumnsCount"):
            drift_info["n_drifted_features"] = int(value.get("count", 0))
            drift_info["share_drifted_features"] = float(value.get("share", 0.0))
            drift_info["dataset_drift"] = drift_info["share_drifted_features"] > 0.0
        elif metric_id.startswith("ValueDrift("):
            drift_info["n_features"] += 1

    return drift_info


# ----------------------------- #
#             FLOW              #
# ----------------------------- #
@flow(name="Data Drift Detection Pipeline")
def drift_check_flow(args: argparse.Namespace):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "random_forest_experiment"))

    with mlflow.start_run(run_name="data_drift_check"):
        train_df = load_data(args.train_path)
        test_df = load_data(args.test_path)
        output_dir = args.output_dir

        drift_json_path = generate_drift_report(train_df, test_df, output_dir)
        drift_summary = extract_drift_summary(drift_json_path)

        mlflow.log_metrics(drift_summary)
        mlflow.log_artifact(os.path.join(output_dir, DRIFT_JSON), artifact_path="artifacts")
        mlflow.log_artifact(os.path.join(output_dir, DRIFT_HTML), artifact_path="artifacts")
        mlflow.set_tag("data_drift_detected", str(drift_summary["dataset_drift"]))


# ----------------------------- #
#          MAIN ENTRY           #
# ----------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chequeo de Data Drift con Evidently y MLflow")
    parser.add_argument("train_path", type=str)
    parser.add_argument("test_path", type=str)
    parser.add_argument("output_dir", type=str, help="Directorio donde guardar los reportes de drift")
    args = parser.parse_args()

    drift_check_flow(args)
