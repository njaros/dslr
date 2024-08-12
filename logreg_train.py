"""This program will train my models and
generates a file containing the weights
that will be used for the prediction."""

import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import ndarray

from tools.logreg_utils import load, standardize
from tools.logreg_config import FEATURES_TO_REMOVE


def sigmoid_function(
    X: ndarray[float], thetas: ndarray[float]
) -> tuple[ndarray[float], ndarray[float]]:
    """Sigmoid function also called logistic function.
    Transform results of linear function into probablities
    of belonging to class 1, between 0 and 1.

    Parameters
    ----------
    X: a matrix of features + biais,
        its shape is m (nb of students) * n (nb of features
        + 1 column for biais).
    thetas: a vector of regression coefficients,
        its size is (nb of features + 1).

    Returns
    -------
    probabilities: a vector of probabilities,
        its size is (nb of students).
    z: a vector corresponding to the results of the linear function,
        its size is (nb of students).
    """
    z = np.array(X.dot(thetas), dtype=float)
    probabilities = 1 / (1 + np.exp(-z))

    return probabilities, z


def cost_function(sigma: ndarray[float], target: ndarray[int]) -> float:
    """Here we evaluate the performance of our model.
    We calculate the likelihood: the plausibility of our model
    in relation to real data, with the log loss function.

    Parameters
    ----------
    sigma: a vector of probabilities,
        its size is (nb of students).
    target: a vector of our target 1 or 0 (1 is for class membership),
        its size is (nb of students).

    Returns
    -------
    res: product of all probabilities, using Bernoulli’s law.
    """
    res = -sum(target * np.log(sigma) + (1 - target) * np.log(1 - sigma)) / target.size

    return res


def find_gradients(
    X: ndarray[float], sigma: ndarray[float], target: ndarray[int]
) -> ndarray[float]:
    """Calcul of gradients (or partial derivatives) of the cost function.

    Parameters
    ----------
    X: a matrix of features + biais,
        its shape is m (nb of students) * n (nb of features
        + 1 column for biais).
    sigma: a vector of probabilities,
        its size is (nb of students).
    target: a vector of our target 1 or 0 (1 is for class membership),
        its size is (nb of students).

    Returns
    -------
    gradients: a vector of (nb of features + 1),
        a gradient per theta.
    """
    X_t = np.transpose(X)  # ndarray(n-features + 1, m)
    gradients = X_t.dot(np.subtract(sigma, target)) / target.size

    return gradients


def training(
    features: ndarray[float],
    target: ndarray[int],
    epochs: int,
    learning_rate: float,
    house: str,
):
    """Our learning algorithm, using gradient descent.
    We will find the most optimized thetas to minimize the cost function.

    Parameters
    ----------
    features: a vector of size (nb of features).
    target: a vector of our target 1 or 0 (1 is for class membership),
        its size is (nb of students).
    epochs: nb of one complete pass of the training data set through our training algorithm.
    learning_rate: a hyper-parameter used to govern the pace at which
        our algorithm updates thetas.
    house: name of the house for which we train our model.
    """
    cost_history: list[float] = []

    m = features.shape[0]
    X = np.hstack((features, np.ones((m, 1))))  # ndarray(m, n-features + 1)
    thetas = np.zeros(X.shape[1])  # ndarray(n-features + 1)

    for i in range(epochs):
        proba, z = sigmoid_function(X, thetas)
        cost_history.append(cost_function(proba, target))
        gradients = find_gradients(X, proba, target)
        thetas = np.subtract(thetas, (learning_rate * gradients))

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    axes[0].scatter(z, target, color="b", s=5, alpha=0.4)
    axes[0].scatter(z, proba, color="r", s=4, alpha=0.6)
    axes[0].set_title(f"Data training for {house}")
    axes[1].plot(list(range(i + 1)), cost_history)
    axes[1].set_title("Cost history")
    plt.show()

    return list(thetas)


def get_ready(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare dataset for training with multi-factor logistic regression:
    drop nan values,
    drop unwanted features,
    standardize data,
    and create a new DataFrame for Hogwarts Houses.

    Parameters
    ----------
    df: our DataFrame.

    Returns
    -------
    df_stand: a new DataFrame with our selected features.
    houses_df: a new DataFrame for houses: each column
        represents one house (with 1 and 0).
    """
    df.dropna(inplace=True)
    houses_df = pd.get_dummies(df["Hogwarts House"])
    # print(houses_df)

    new_df = df.select_dtypes(["number"])
    new_df.drop(columns=FEATURES_TO_REMOVE, inplace=True)
    df_stand = standardize(new_df)

    return (df_stand, houses_df)


def check_args() -> tuple[int, float]:
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
        print(
            """
            usage : python logreg_train.py path_csv_file (epochs learning_rate).

            rule : this program will train a model to predict
                   the house membership of a Hogwarts student
                   and generates a file with weights for each house
                   named weights.json.
            """
        )
        sys.exit()
    if ac != 2 and ac != 4:
        print("Incorrect input.")
        sys.exit()
    if ac == 2:
        return 100, 3.0
    return int(sys.argv[2]), float(sys.argv[3])


def main():
    """Main function"""
    try:
        epochs, learning_rate = check_args()
        dataset = load(sys.argv[1])
        pd.set_option("future.no_silent_downcasting", True)
        df, houses = get_ready(dataset)
        weights = {}

        for col in houses.columns:
            thetas = training(
                df.to_numpy(),
                houses[col].to_numpy().astype(int),
                epochs,
                learning_rate,
                col,
            )
            weights[col] = thetas

        with open("weigths.json", "w", encoding="utf8") as file:
            json.dump(weights, file, indent=4)

    except ValueError:
        print("There is a problem in your input parameters.")
    except AssertionError as msg:
        print(f"{msg.__class__.__name__}: {msg}")


if __name__ == "__main__":
    main()
