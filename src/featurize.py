import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from typing import Any, Tuple
from prefect import flow, task

COLS_CAT_OTHERS = ['grp_camptot06m', 'region', 'ubigeo_buro', 'grp_riesgociiu']
COLS_CAT_GRUPO0 = ['grp_campecs06m']
TARGET_COL = 'target' # <--- AÑADIR ESTO

@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@task
def get_num_columns(df: pd.DataFrame, target_col: str = TARGET_COL) -> list: # <--- Leve ajuste
    cols_numericas = df.select_dtypes(include=np.number).columns.drop(target_col).tolist()
    return cols_numericas

@task
def create_preprocessor(numeric_cols: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ('imputer_num', SimpleImputer(strategy='mean'))
    ])
    cat_others_pipeline = Pipeline(steps=[
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Otros')),
        ('encoder_ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    cat_grupo0_pipeline = Pipeline(steps=[
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='grupo_0')),
        ('encoder_ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat_otros', cat_others_pipeline, COLS_CAT_OTHERS),
            ('cat_grupo0', cat_grupo0_pipeline, COLS_CAT_GRUPO0),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    preprocessor.set_output(transform="pandas")
    return preprocessor

@task
def separate_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y

@task
def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer: # <--- Cambiado a X_train
    preprocessor.fit(X_train)
    return preprocessor

@task
def transform_data(preprocessor: ColumnTransformer, X_df: pd.DataFrame) -> pd.DataFrame: # <--- Cambiado a X_df
    processed_df = preprocessor.transform(X_df)
    return processed_df # type: ignore

@task(retries=3, retry_delay_seconds=5)
def save_artifact(artifact: Any, path: str):
    joblib.dump(artifact, path)

@task(retries=3, retry_delay_seconds=5)
def save_data(df: pd.DataFrame, path: str):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    df.to_csv(path, index=False)

@flow(name="Feature Engineering Pipeline")
def featurize_flow(args):    
    train_path = os.path.join(args.input_dir, "train.csv")
    test_path = os.path.join(args.input_dir, "test.csv")    
    preprocessor_path = os.path.join(args.model_dir, "preprocessor.pkl")

    train_df = load_data(train_path)
    test_df = load_data(test_path)

    # 1. separar X e Y
    X_train, y_train = separate_features_target(train_df, TARGET_COL)
    X_test, y_test = separate_features_target(test_df, TARGET_COL)

    # 2. columnas numericas
    num_columns = get_num_columns(train_df)
    
    # 3. crear y ajustar el preprocesador solo en X_train
    preprocessor = create_preprocessor(num_columns)
    fitted_preprocessor = fit_preprocessor(preprocessor, X_train)

    # 4. transformar X_train y X_test
    X_train_processed = transform_data(fitted_preprocessor, X_train)
    X_test_processed = transform_data(fitted_preprocessor, X_test)

    # 5. re-unir los datos procesados con sus 'target'
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    train_final_df = pd.concat([X_train_processed, y_train], axis=1)
    test_final_df = pd.concat([X_test_processed, y_test], axis=1)
    
    # 6. guardar artefactos y datos finales
    save_artifact(fitted_preprocessor, preprocessor_path)
    save_data(train_final_df, os.path.join(args.output_dir, "train.csv"))
    save_data(test_final_df, os.path.join(args.output_dir, "test.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("model_dir", type=str)
    args = parser.parse_args()

    featurize_flow(args)