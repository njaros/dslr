import sys
import pandas as pd
import json
from sklearn.model_selection import train_test_split


def evaluate(value, theta0, theta1) -> float:
    return value * theta1 + theta0


def is_member(line, params) -> bool:
    """is_member function"""
    value = (line[params["feature"]] - params["mean"]) / params["std"]
    return evaluate(value, params["theta0"], params["theta1"]) >= 0


def predict_one(line, model) -> str:
    """predict_one function"""
    if is_member(line, model["Slytherin"]):
        return "Slytherin"
    if is_member(line, model["Ravenclaw"]):
        return "Ravenclaw"
    if is_member(line, model["Gryffindor"]):
        return "Gryffindor"
    return "Hufflepuff"


def prepare(df: pd.DataFrame):
    return df.drop(["Index", "Hogwarts House", "First Name", "Last Name", "Birthday"])


def predict(csv_path):
    """predict function"""
    df = pd.read_csv(csv_path)
    response = df["Hogwarts House"]
    prepared_train = prepare(df)
    train_df, test_df, res_train, res_test = train_test_split(
        prepared_train, response, test_size=0.3, random_state=1
    )
    with open("model.json", "r") as io:
        model = json.load(io)
        io.close()
    result = pd.DataFrame(columns=["Index", "Hogwarts House"])
    for index, line in df.iterrows():
        result.loc[index] = [index, predict_one(line, model)]
    result.to_csv("houses.csv", index=False)


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
            predict(av[1])
        except UnicodeDecodeError as e:
            print(f"{e.__class__.__name__}: {e.args[4]}")
        except Exception as e:
            print(f"{e.__class__.__name__}: {e.args}")
