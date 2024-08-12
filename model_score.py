import sys
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from logreg_train import prepare_df, train
from tools.logreg_config import FEATURE_TO_DROP


def evaluate(value, theta0, theta1) -> float:
    return value * theta1 + theta0


def score(csv_path):
    """score function

    - prepare the two datasets: one for train model, the other
    to evaluate the score of the model

    - train model

    - evaluate the score
    """
    model = {}
    model["features"] = dict()
    model["thetas"] = dict()
    X, Y = prepare_df(csv_path, model["features"])
    train_df, test_df, res_train, res_test = train_test_split(
        X, Y, test_size=0.3, random_state=1
    )
    print(f"{train_df=}, {test_df=}, {res_train=}, {res_test=}")


if __name__ == "__main__":
    av = sys.argv
    ac = len(av)
    if ac == 1 or av[1] == "-h" or av[1] == "-help":
        print(
            """
            usage: python model_score.py path_to_dataset.

            rule: this program will calculate the efficiency of a logistic
                   regression algorithm using a train dataset.

            how: program split into two part the dataset.
                 > one part for train the model using the algorithm
                 > the other part for calculate the efficiency of the model
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
            score(av[1])
        except UnicodeDecodeError as e:
            print(f"{e.__class__.__name__}: {e.args[4]}")
        # except Exception as e:
        #     print(f"{e.__class__.__name__}: {e.args}")
