"""This program will predict which house each Hogwarts student
belongs to and generates a csv file with results."""

import sys
import json
import pandas as pd
import numpy as np
from numpy import ndarray

from tools.logreg_config import FEATURES_TO_REMOVE
from tools.logreg_utils import load


def take_best_score(preds: dict[str, ndarray[float]], idx: int) -> str:
    """This function find the max value for student with index idx.
    
    Parameters
    ----------
    preds: a dict of predictions for each students.
    idx: Index of student.
    
    Returns
    -------
    house: the name of the Hogwarts House where is our student.
    """
    keys = list(preds.keys())
    best_score = preds[keys[0]][idx]
    house = keys[0]

    for key in keys[1:]:
        if preds[key][idx] >= best_score:
            best_score = preds[key][idx]
            house = key

    return house


def evaluate(X: ndarray[float], thetas: ndarray[float]) -> ndarray[float]:
    """Evaluate function: creates the log(odds) value for each line
    of X.

    Parameters
    ----------
    X: a matrix of features + biais,
        its shape is m (nb of students) * n (nb of features
        + 1 column for biais).
    thetas: a vector of regression coefficients,
        its size is (nb of features + 1).

    Returns
    -------
    z: a vector of log(odds) values,
        its size is (nb of students).
    """
    return X.dot(thetas)


def prepare_df_prediction(
    df: pd.DataFrame, scalers: dict[str, dict[str, float]]
) -> ndarray[float]:
    """Prepare dataset: drop unwanted features,
    standardize data,
    and create a matrix of our selected features.

    Parameters
    ----------
    df: our DataFrame.
    scalers: for each features a dict of mean and std,
        which will allow me to standardize my data.

    Returns
    -------
    X: a matrix of features + biais,
        its shape is m (nb of students) * n (nb of features
        + 1 column for biais).
    """
    new_df = df.select_dtypes(["number"])
    new_df.drop(columns=FEATURES_TO_REMOVE, inplace=True)
    new_df.drop(columns="Hogwarts House", inplace=True)

    for col in new_df.columns:
        mean = scalers[col]["mean"]
        std = scalers[col]["std"]
        new_df[col] = new_df[col].map(lambda x: 0 if pd.isna(x) else (x - mean) / std)

    datas = new_df.to_numpy()
    biais = np.ones((datas.shape[0], 1), dtype=float)
    X = np.hstack((datas, biais))

    return X


def predict(df: pd.DataFrame, model: dict):
    """Predict function.

    Parameters
    ----------
    df: our DataFrame.
    model: a dict with features informations (mean, std) and
        lists of thetas for each house.
    """
    result = pd.DataFrame(columns=["Index", "Hogwarts House"])
    X = prepare_df_prediction(df, model["features"])
    predictions_per_house = {}

    for key in model["thetas"]:
        predictions_per_house[key] = evaluate(X, model["thetas"][key])

    for i in range(len(X)):
        result.loc[i] = [i, take_best_score(predictions_per_house, i)]

    return result


def main():
    """Main function"""
    av = sys.argv
    ac = len(av)
    if ac == 1 or av[1] == "-h" or av[1] == "-help":
        print(
            """
            usage : python logreg_predict.py path_csv_file.

            rule : this program will predict the house membership
                   for each line in a dataset containing a set of
                   school student in Hogwarts.
            """
        )
    elif ac != 2:
        print(
            """
            Only one argument required. try -help for details.
            """
        )
    else:
        try:
            df = load(sys.argv[1])
            with open("weigths.json", "r", encoding="utf8") as file:
                model = json.load(file)
            predictions = predict(df, model)
            predictions.to_csv("houses.csv", index=False)

        except UnicodeDecodeError as e:
            print(f"{e.__class__.__name__}: {e.args[4]}")
        except AssertionError as msg:
            print(f"{msg.__class__.__name__}: {msg}")
        # except Exception as e:
        #     print(f"{e.__class__.__name__}: {e.args}")


if __name__ == "__main__":
    main()
