"""This file includes several utils functions."""

import sys
import os
import pandas as pd


def standardize(dataset: pd.DataFrame) -> pd.DataFrame:
    """Standardize a dataset"""
    df = dataset.copy(deep=True)
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].dropna().subtract(df[col].mean()).divide(df[col].std())

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


def load(path: str) -> pd.DataFrame:
    """Load a data file using pandas library"""
    assert isinstance(path, str), "your path is not valid."
    assert os.path.exists(path), "your file doesn't exist."
    assert os.path.isfile(path), "your 'file' is not a file."
    assert path.lower().endswith(".csv"), "file format is not .csv."
    data = pd.read_csv(path)
    print(f"Loading dataset of dimensions {data.shape}")

    return data


def check_args(help_msg: str) -> tuple[int, float]:
    """Checks the program arguments. If there are no optional arguments
    then it returns a predefined number of epochs and learning rate.

    Returns
    -------
    epochs: nb of one complete pass of the training data set through our training algorithm.
        Predefined at 100.
    learning_rate: a hyper-parameter used to govern the pace at which
        our algorithm updates thetas.
        Predefined at 3.0.
    """
    av = sys.argv
    ac = len(av)
    if ac == 1 or av[1] == "-h" or av[1] == "-help":
        print(help_msg)
        sys.exit()
    if ac != 2 and ac != 4:
        print("Incorrect input.")
        sys.exit()
    if ac == 2:
        return 100, 3.0
    return int(sys.argv[2]), float(sys.argv[3])
