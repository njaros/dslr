import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import json
import tqdm

pd.options.mode.chained_assignment = None


FEATURE_TO_DROP = [
    "Arithmancy",
    "Potions",
    "Care of Magical Creatures",
    "Defense Against the Dark Arts",
    "Index",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
    "Transfiguration",
]


def harry_plotter(df: pd.DataFrame, house: str, theta0: float, theta1: float):
    """harry_plotter function

    The magicien of the plots !
    Create a house_name.png which shows the result of a training
    """
    feat = df.columns[1]
    res = df.columns[0]
    min = int(np.floor(df[feat].min()))
    max = int(np.ceil(df[feat].max()))
    x = []
    y = []
    plt.clf()
    plt.scatter(df[feat], df[res], c="r", s=1)
    for i in range(min, max):
        for j in range(10):
            x.append(i + 0.1 * j)
            y.append(sigmoide(estimate_log_odds(i + 0.1 * j, theta0, theta1)))
    plt.plot(x, y, "b")
    plt.plot([min, max], [0.5, 0.5], "black")
    plt.xlabel(feat)
    plt.ylabel(f"{house} member")
    plt.savefig(f"{house}.png")


def estimate_log_odds(X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """estimate_member function

    X: np.ndarray: dataset as a matrix + the bias.
    thetas: np.ndarray: current theta values as a vector of the in training model.

    return: np.ndarray: a vector corresponding of log(odds) for each line to be a membership.
    """
    return X.dot(thetas)


def sigmoide(log_odd: np.ndarray) -> np.ndarray:
    """sigmoide function

    odds: np.ndarray: a vector of log(odds)

    return: np.ndarray: a vector of probabilities
    """
    return 1.0 / (1.0 + np.exp(-log_odd))


def log_likelihood(df: pd.DataFrame, theta0: float, theta1: float) -> float:
    """log_likelihood function

    df: Dataframe: the dataset with only one feature x and results y
    theta0: float: the y-intercept of the in-training model.
    theta1: float: the slope of the in-training model.

    returns: float: the log_likelihood value.
    """
    x = df[df.columns[1]]
    y = df[df.columns[0]]
    log_sum = 0
    for xi, yi in zip(x, y):
        h = sigmoide(estimate_log_odds(xi, theta0, theta1))
        if yi == 1:
            log_sum += np.log(h)
        else:
            log_sum += np.log(1 - h)
    return -log_sum / df.count()


def gradients(X: np.ndarray, Y: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """gradient_slope function

    bla bla bla
    """
    probs = sigmoide(estimate_log_odds(X, thetas))
    return X.T.dot(np.subtract(probs, Y)) / Y.size


def build_thetas(
    X: np.ndarray, Y: np.ndarray, learning_rate: float, iterations: int, thetas: dict
):
    """build_thetas function

    bla bla bla
    """
    np_thetas = np.zeros(X.shape[1])
    for i in tqdm.tqdm(range(iterations)):
        np_thetas = np.subtract(np_thetas, learning_rate * gradients(X, Y, np_thetas))
    for idx in range(np_thetas.shape[0]):
        thetas[idx] = float(np_thetas[idx])


def house_training(house: str, df: pd.DataFrame, json_data: dict):
    """
    df: Dataframe: original dataframe
    house: str: targeted house
    json_data: dict: store results to build the model.json file later

    rule: train a model for a specific house and store it in the json_data
    """
    json_data[house] = dict()
    datas = df.drop(columns="Hogwarts House").to_numpy()
    biais = np.ones((datas.shape[0], 1), dtype=float)

    Y = df["Hogwarts House"].map(lambda x: 1 if x == house else 0).to_numpy()
    X = np.hstack((datas, biais))

    print(f"training model for {house}...")
    build_thetas(X, Y, 0.5, 100, json_data[house])


def prepare_df(path: str, json_data: dict):
    df = pd.read_csv(path)
    df.drop(columns=FEATURE_TO_DROP, inplace=True)
    df.dropna(inplace=True)
    for col in df.columns.drop("Hogwarts House"):
        mean = df[col].mean()
        std = df[col].std()
        json_data[col] = dict()
        json_data[col]["mean"] = mean
        json_data[col]["std"] = std
        df[col] = df[col].subtract(mean).divide(std)
    return df


def train(path):
    """
    path: string -> it must be a relative or absolute path to a csv file.

    rule: train a model to predict the Hogwarts House membership

    exceptions: raise exception if the file isn't readable
                or is to incorrect format, depending of pandas exceptions.
    """
    json_data = {}
    json_data["features"] = dict()
    json_data["thetas"] = dict()
    df = prepare_df(path, json_data["features"])
    house_training("Slytherin", df, json_data["thetas"])
    house_training("Ravenclaw", df, json_data["thetas"])
    house_training("Gryffindor", df, json_data["thetas"])
    house_training("Hufflepuff", df, json_data["thetas"])
    with open("model.json", "w") as io:
        json.dump(json_data, io, indent=4)
        io.close()
        print("model trained and saved in model.json")


if __name__ == "__main__":
    av = sys.argv
    ac = len(av)
    if ac == 1 or av[1] == "-h" or av[1] == "-help":
        print(
            """
            usage : python logreg_train.py path_csv_file.

            rule : this program will train a model to predict
                   the house membership of a Hogwarts student
                   and generates a file with weights for each house
                   named weights.json.
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
            train(av[1])
        except UnicodeDecodeError as e:
            print(f"{e.__class__.__name__}: {e.args[4]}")
        # except Exception as e:
        #     print(f"{e.__class__.__name__}: {e.args}")
