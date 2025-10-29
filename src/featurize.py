import argparse
import os
import pandas as pd
from tasks import (
    load_data,
    get_num_columns,
    create_preprocessor,
    separate_features_target,
    fit_preprocessor,
    transform_data,
    save_artifact,
    save_data,
    TARGET_COL
)
from prefect import flow

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