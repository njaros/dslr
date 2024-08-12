import sys
import pandas as pd
import json
import numpy as np
from tools.logreg_config import FEATURE_TO_DROP


def take_best_score(preds: dict, idx: int) -> str:
    """function take_best_score
    blablabla
    """
    keys = list(preds.keys())
    best_score = preds[keys[0]][idx]
    house = keys[0]
    for key in keys[1:]:
        if preds[key][idx] >= best_score:
            best_score = preds[key][idx]
            house = key
    return house


def evaluate(X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """evaluate function
    create the log(odds) value for each line of X
    returns it as a np.ndarray
    """
    return X.dot(thetas)


def prepare_df_prediction(df: pd.DataFrame, scalers: dict) -> np.ndarray:
    """prepare_df_prediction

    blablabla
    """
    df.drop(columns=FEATURE_TO_DROP, inplace=True)
    df.drop(columns="Hogwarts House", inplace=True)
    for col in df.columns:
        mean = scalers[col]["mean"]
        std = scalers[col]["std"]
        df[col] = df[col].map(lambda x: 0 if pd.isna(x) else (x - mean) / std)
    datas = df.to_numpy()
    biais = np.ones((datas.shape[0], 1), dtype=float)
    X = np.hstack((datas, biais))
    return X


def predict(df: pd.DataFrame, model: dict):
    """predict function"""
    result = pd.DataFrame(columns=["Index", "Hogwarts House"])
    X = prepare_df_prediction(df, model["features"])
    predictions_per_house = {}
    for key in model["thetas"]:
        predictions_per_house[key] = evaluate(X, model["thetas"][key])
    for i in range(len(X)):
        result.loc[i] = [i, take_best_score(predictions_per_house, i)]
    return result


if __name__ == "__main__":
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
            df = pd.read_csv(av[1])
            with open("model.json", "r") as io:
                model = json.load(io)
            predictions = predict(df, model)
            predictions.to_csv("houses.csv", index=False)
        except UnicodeDecodeError as e:
            print(f"{e.__class__.__name__}: {e.args[4]}")
        # except Exception as e:
        #     print(f"{e.__class__.__name__}: {e.args}")
