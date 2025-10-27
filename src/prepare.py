import os
import argparse
import pandas as pd
import numpy as np
from prefect import flow, task
from sklearn.model_selection import train_test_split

@task
def read_data(args) -> pd.DataFrame:
    return pd.read_csv(args.input)

@task
def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['key_value', 'grp_camptottlv06m', 'grp_campecstlv06m'], axis=1)
    return df

@task
def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    df_nan = df.isna().mean() * 100
    df_nan_ge_80 = df_nan[df_nan >= 80].round(2)
    df = df.drop(columns=df_nan_ge_80.index)
    return df

@task
def save_process_data(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=1 - args.split_ratio, random_state=args.seed)
    
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.csv")
    test_path = os.path.join(args.output, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[prepare] Saved data: {train_path} y {test_path}")

@flow(name="Process data")
def process_data(args):
    df = read_data(args)
    df = (
        df.pipe(drop_irrelevant_columns)
          .pipe(drop_na)
    )
    save_process_data(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to dataset")
    parser.add_argument("output", type=str, help="Output directory")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    process_data(args)
