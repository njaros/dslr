import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import json
import tqdm

pd.options.mode.chained_assignment = None


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
            y.append(sigmoide(estimate_member(i + 0.1 * j, theta0, theta1)))
    plt.plot(x, y, "b")
    plt.plot([min, max], [0.5, 0.5], "black")
    plt.xlabel(feat)
    plt.ylabel(f"{house} member")
    plt.savefig(f"{house}.png")


def estimate_member(value: float, theta0: float, theta1: float) -> float:
    """estimate_member function

    value: float: a value of a x on a specific line of the dataset.
    theta0: float: the y-intercept of the in-training model.
    theta1: float: the slope of the in-training model.

    return: float: the estimation of the model with the entry x.
                   if the result is > 0 then x is member,
                   else he is not.
    """
    return theta0 + theta1 * value


def sigmoide(x: float) -> float:
    """sigmoide function

    x: float: the parameter given to the sigmoide function

    return: the value of sigmoide(x)
    """
    return 1.0 / (1.0 + np.exp(-x))


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
        h = sigmoide(estimate_member(xi, theta0, theta1))
        if yi == 1:
            log_sum += np.log(h)
        else:
            log_sum += np.log(1 - h)
    return -log_sum / df.count()


def gradient_slope(
    datas: list[tuple[float, float]], theta0: float, theta1: float
) -> float:
    """gradient_slope function

    bla bla bla
    """
    log_sum = 0
    for xi, yi in datas:
        h = sigmoide(estimate_member(xi, theta0, theta1))
        log_sum += xi * (h - yi)
    return log_sum / len(datas)


def gradient_y_intercept(
    datas: list[tuple[float, float]], theta0: float, theta1: float
) -> float:
    """gradient_y_intercept function

    bla bla bla
    """
    log_sum = 0
    for xi, yi in datas:
        h = sigmoide(estimate_member(xi, theta0, theta1))
        log_sum += h - yi
    return log_sum / len(datas)


def build_thetas(
    df: pd.DataFrame, learning_rate: float, iterations: int
) -> tuple[float, float]:
    """build_thetas function

    bla bla bla
    """
    theta0 = 0
    theta1 = 0
    x = df[df.columns[1]]
    y = df[df.columns[0]]
    datas = list(zip(x, y))
    for i in tqdm.tqdm(range(iterations)):
        temp_theta0 = theta0
        theta0 -= learning_rate * gradient_y_intercept(datas, theta0, theta1)
        theta1 -= learning_rate * gradient_slope(datas, temp_theta0, theta1)
    return (theta0, theta1)


def house_training(df: pd.DataFrame, house: str, feat: str, json_data: dict):
    """
    df: Dataframe: original dataframe
    house: str: targeted house
    feat: str: feature which wille be used for train the model
    json_data: dict: store results to build the model.json file later

    rule: create an exploitable dataframe to train the model
    return: a new dataframe
    """
    new_df: pd.DataFrame = df.dropna(subset=feat).get(["Hogwarts House", feat])
    new_df["Hogwarts House"] = new_df["Hogwarts House"].map(
        lambda x: 1 if x == house else 0
    )
    new_df.rename(columns={"Hogwarts House": f"{house} member"}, inplace=True)
    mean = new_df[feat].mean()
    std = new_df[feat].std()
    json_data[house] = dict()
    json_data[house]["feature"] = feat
    json_data[house]["mean"] = mean
    json_data[house]["std"] = std
    new_df[feat] = new_df[feat].subtract(mean).divide(std)
    print(f"training for {house} with feature {feat}...")
    theta0, theta1 = build_thetas(new_df, 0.5, 100)
    json_data[house]["theta0"] = theta0
    json_data[house]["theta1"] = theta1
    print("training succesfully done.")
    print(f"creating feedback plots file {house}.png")
    harry_plotter(new_df, house, theta0, theta1)
    print(f"{house}.png created")


def train(path):
    """
    path: string -> it must be a relative or absolute path to a csv file.

    rule: train a model to predict the Hogwarts House membership

    exceptions: raise exception if the file isn't readable
                or is to incorrect format, depending of pandas exceptions.
    """
    df = pd.read_csv(path)
    json_data = {}
    house_training(df, "Slytherin", "Divination", json_data)
    house_training(df, "Ravenclaw", "Charms", json_data)
    house_training(df, "Gryffindor", "Transfiguration", json_data)
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
        except Exception as e:
            print(f"{e.__class__.__name__}: {e.args}")
