import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import json

pd.options.mode.chained_assignment = None


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
    for i in range(iterations):
        print(i)
        temp_theta0 = theta0
        theta0 -= learning_rate * gradient_y_intercept(datas, theta0, theta1)
        theta1 -= learning_rate * gradient_slope(datas, temp_theta0, theta1)
    return (theta0, theta1)


def separate_house(df: pd.DataFrame, house, feat) -> pd.DataFrame:
    """
    df: Dataframe: original dataframe
    house: str: targeted house
    feat: str: feature which wille be used for train the model

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
    with open("weigths.json", "w") as io:
        json.dump({f"{feat}_scale": {"mean": mean, "std": std}}, io, indent=4)
        io.close()
    new_df[feat] = new_df[feat].subtract(mean).divide(std)
    return new_df


def do_plot(df: pd.DataFrame, filename):
    plt.scatter(df[df.columns[1]], df[df.columns[0]])
    plt.xlabel(df.columns[1])
    plt.ylabel(df.columns[0])
    plt.savefig(filename)


def do_plot_sigmoide(thetas):
    results = []
    x_axis = []
    x = -5
    while x < 5:
        x_axis.append(x)
        results.append(sigmoide(estimate_member(x, thetas[0], thetas[1])))
        x += 0.1
    plt.plot(x_axis, results)
    plt.savefig("test_log.png")


def train(path):
    """
    path: string -> it must be a relative or absolute path to a csv file.

    rule: train a model to predict the Hogwarts House membership

    exceptions: raise exception if the file isn't readable
                or is to incorrect format, depending of pandas exceptions.
    """
    df = pd.read_csv(path)
    df_sly = separate_house(df, "Slytherin", "Divination")
    df_rav = separate_house(df, "Ravenclaz", "Charms")
    df_gry = separate_house(df, "Gryffindor", "Transfiguration")
    do_plot(df_sly, "SERPENTAR.png")
    sly_thetas = build_thetas(df_sly, 0.3, 100)
    do_plot_sigmoide(sly_thetas)
    print(sly_thetas)


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
