import argparse
import os
from prefect import flow
from tasks import (
    load_data,
    drop_irrelevant_columns,
    drop_na,
    load_artifact,
    transform_data,
    save_data
)

@flow(name="Apply Preprocessor Flow")
def apply_preprocessor_flow(args):    
    # 1. Cargar datos crudos
    df_raw = load_data(args.raw_data_path)
    
    # 2. Aplicar la logica de prepare.py
    df_prepared = (
        df_raw.pipe(drop_irrelevant_columns)
              .pipe(drop_na)
    )
    
    # 3. Cargar el preprocesador entrenado
    preprocessor = load_artifact(args.preprocessor_path)
    
    # 4. Aplicar la logica de featurize.py
    df_processed = transform_data(preprocessor, df_prepared)
    
    # 5. Guardar los datos de drift ya procesados
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_data(df_processed, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessor_path", type=str)
    parser.add_argument("raw_data_path", type=str, help="Path a los datos crudos de drift (CSV)")
    parser.add_argument("output_path", type=str, help="Path donde guardar el CSV procesado")
    args = parser.parse_args()

    apply_preprocessor_flow(args)