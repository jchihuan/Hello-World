import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from typing import Any, Tuple
from prefect import task

COLS_CAT_OTHERS = ['grp_camptot06m', 'region', 'ubigeo_buro', 'grp_riesgociiu']
COLS_CAT_GRUPO0 = ['grp_campecs06m']
TARGET_COL = 'target'

# prepare.py

@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@task
def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas irrelevantes."""
    cols_to_drop = ['key_value', 'grp_camptottlv06m', 'grp_campecstlv06m']
    # Dropear solo las columnas que existen, para evitar errores
    cols_exist = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_exist, axis=1)
    return df

@task
def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas con >= 80% de valores nulos."""
    df_nan = df.isna().mean() * 100
    df_nan_ge_80 = df_nan[df_nan >= 80].round(2)
    df = df.drop(columns=df_nan_ge_80.index)
    return df

@task
def split_and_save_data(df: pd.DataFrame, output_dir: str, split_ratio: float, seed: int):
    """Divide los datos en train/test."""
    train_df, test_df = train_test_split(df, test_size=1 - split_ratio, random_state=seed)
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[split_and_save_data] Saved data: {train_path} y {test_path}")

# featurize.py

@task
def get_num_columns(df: pd.DataFrame, target_col: str = TARGET_COL) -> list:
    """Obtiene columnas numericas excluyendo el target."""
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
    """Separa features (X) y target (y)."""
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y

@task
def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    preprocessor.fit(X_train)
    return preprocessor

@task
def transform_data(preprocessor: ColumnTransformer, X_df: pd.DataFrame) -> pd.DataFrame:
    processed_df = preprocessor.transform(X_df)
    return processed_df # type: ignore

@task(retries=3, retry_delay_seconds=5)
def save_artifact(artifact: Any, path: str):
    joblib.dump(artifact, path)

@task(retries=3, retry_delay_seconds=5)
def save_data(df: pd.DataFrame, path: str):
    """Guarda un DataFrame en CSV."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    df.to_csv(path, index=False)

@task(retries=3, retry_delay_seconds=5)
def load_artifact(path: str) -> Any:
    return joblib.load(path)