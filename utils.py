import os
import pandas as pd


def standardize(dataset: pd.DataFrame) -> pd.DataFrame:
    """Standardize a dataset"""
    df = dataset.copy(deep=True)
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            df[col] = (df[col].dropna().subtract(df[col].mean()).divide(df[col].std()))
    return df


def normalize(dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dataset"""
    df = dataset.copy(deep=True)
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            df[col] = (df[col].dropna() - df[col].min()) / (
                df[col].max() - df[col].min()
            )
    return df


def load(path: str) -> pd.DataFrame | None:
    """Load a data file using pandas library"""
    try:
        assert isinstance(path, str), "your path is not valid."
        assert os.path.exists(path), "your file doesn't exist."
        assert os.path.isfile(path), "your 'file' is not a file."
        assert path.lower().endswith(".csv"), "file format is not .csv."
        data = pd.read_csv(path)
        print(f"Loading dataset of dimensions {data.shape}")
        return data
    except AssertionError as msg:
        print(f"{msg.__class__.__name__}: {msg}")
        return None
