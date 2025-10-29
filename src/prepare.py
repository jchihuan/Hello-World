import argparse
from tasks import (
    load_data,
    drop_irrelevant_columns,
    drop_na,
    split_and_save_data
)
from prefect import flow

@flow(name="Process data")
def process_data(args):
    df = load_data(args.input)
    df_prepared = (
        df.pipe(drop_irrelevant_columns)
          .pipe(drop_na)
    )
    split_and_save_data(df_prepared, args.output, args.split_ratio, args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to dataset")
    parser.add_argument("output", type=str, help="Output directory")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    process_data(args)
