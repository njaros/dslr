"""This program will train my models and
generates a file containing the weights
that will be used for the prediction."""

import sys
import json
import pandas as pd
import numpy as np
from numpy import ndarray

from tools.logreg_utils import load, check_args, harry_plotter
from tools.logreg_config import (
    FEATURES_TO_REMOVE,
    HELP_TRAIN,
    CHOOSEN_ALGORITHM,
    NUMBER_OF_BATCH,
)


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
    res = -sum([np.log(x) if y == 1 else np.log(1 - x) for x, y in zip(sigma, target)])
    # res = -sum(target * np.log(sigma) + (1 - target) * np.log(1 - sigma)) / target.size

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


def batch_gradient_descent(
    features: ndarray[float],
    target: ndarray[int],
    epochs: int,
    learning_rate: float,
    house: str,
):
    """Our learning algorithm, using batch gradient descent.
    We will find the most optimized thetas to minimize the cost function.

    Parameters
    ----------
    features: a vector of size (nb of features).
    target: a vector of our target 1 or 0 (1 is for class membership),
        its size is (nb of students).
    epochs: nb of one complete pass of the training data set through
        our training algorithm.
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

    harry_plotter(z, proba, target, cost_history, house, i)

    return list(thetas)


def mini_batch_gradient_descent(
    features: ndarray[float],
    target: ndarray[int],
    epochs: int,
    learning_rate: float,
    house: str,
):
    """Our learning algorithm, using mini batch gradient descent.
    We will find the most optimized thetas to minimize the cost function.

    Parameters
    ----------
    features: a vector of size (nb of features).
    target: a vector of our target 1 or 0 (1 is for class membership),
        its size is (nb of students).
    epochs: nb of one complete pass of the training data set through
        our training algorithm.
    learning_rate: a hyper-parameter used to govern the pace at which
        our algorithm updates thetas.
    house: name of the house for which we train our model.
    """
    cost_history: list[float] = []
    nb_chunk = NUMBER_OF_BATCH

    m = features.shape[0]
    chunk_size = int(m / nb_chunk)
    X = np.hstack((features, np.ones((m, 1))))  # ndarray(m, n-features + 1)
    thetas = np.zeros(X.shape[1])  # ndarray(n-features + 1)

    for i in range(epochs):
        j = chunk_size
        full_proba = np.empty(0)
        full_z = np.empty(0)

        for c in range(0, m, chunk_size):
            chunk_X = X[c:j]
            chunk_target = target[c:j]
            proba, z = sigmoid_function(chunk_X, thetas)
            full_proba = np.append(full_proba, proba)
            full_z = np.append(full_z, z)
            # cost_history.append(cost_function(proba, chunk_target))
            gradients = find_gradients(chunk_X, proba, chunk_target)
            thetas = np.subtract(thetas, (learning_rate * gradients))
            j += chunk_size
        cost_history.append(cost_function(full_proba, target))

    # blop = list(range(len(cost_history)))
    # ticks = blop[::nb_chunk]
    # tick_labels = [int(tick / nb_chunk) for tick in ticks]
    # _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    # axes[0].scatter(full_z, target, color="b", s=5, alpha=0.4)
    # axes[0].scatter(full_z, full_proba, color="r", s=4, alpha=0.6)
    # axes[0].set_title(f"Data training for {house}")
    # axes[1].plot(blop, cost_history)
    # axes[1].set_title("Cost history")
    # axes[1].set_xticks(ticks, labels=tick_labels)
    # plt.show()

    harry_plotter(full_z, full_proba, target, cost_history, house, i)

    return list(thetas)


def stochastic_gradient_descent(
    features: ndarray[float],
    target: ndarray[int],
    epochs: int,
    learning_rate: float,
    house: str,
):
    """Our learning algorithm, using stochastic gradient descent.
    We will find the most optimized thetas to minimize the cost function.

    Parameters
    ----------
    features: a vector of size (nb of features).
    target: a vector of our target 1 or 0 (1 is for class membership),
        its size is (nb of students).
    epochs: nb of one complete pass of the training data set through
        our training algorithm.
    learning_rate: a hyper-parameter used to govern the pace at which
        our algorithm updates thetas.
    house: name of the house for which we train our model.
    """
    cost_history: list[float] = []

    m = features.shape[0]
    X = np.hstack((features, np.ones((m, 1))))  # ndarray(m, n-features + 1)
    thetas = np.zeros(X.shape[1])  # ndarray(n-features + 1)
    for i in range(epochs):
        j = 0
        proba_array = np.ndarray((X.shape[0]))
        z_array = np.ndarray((X.shape[0]))
        for line in X:
            proba, z = sigmoid_function(line, thetas)
            proba_array[j] = proba
            z_array[j] = z
            gradients = find_gradients(line, proba, target[j])
            thetas = np.subtract(thetas, (learning_rate * gradients))
            j += 1
        cost_history.append(cost_function(proba_array, target))

    harry_plotter(z_array, proba_array, target, cost_history, house, i)

    return list(thetas)


def train(
    df: pd.DataFrame,
    houses: pd.DataFrame,
    epochs: int,
    learning_rate: float,
    json_data: dict[str, list[float]],
):
    """Trains dataset for each Hogwarts houses, and fill
    json_data dict with our optimals thetas.

    Parameters
    ----------
    df: DataFrame with our selected features.
    houses: DataFrame for houses: each column
        represents one house (with 1 and 0).
    epochs: nb of one complete pass of the training data set through
        our training algorithm.
    learning_rate: a hyper-parameter used to govern the pace at which
        our algorithm updates thetas.
    json_data: a dict to fill : for each houses => a list of thetas.
    """
    if CHOOSEN_ALGORITHM == 1:
        print("Algorithm chosen: Batch gradient descent")
        f = batch_gradient_descent
    elif CHOOSEN_ALGORITHM == 2:
        print("Algorithm chosen: Stochastic gradient descent")
        f = stochastic_gradient_descent
    elif CHOOSEN_ALGORITHM == 3:
        print("Algorithm chosen: Mini batch gradient descent")
        f = mini_batch_gradient_descent
    else:
        raise AssertionError(f"{CHOOSEN_ALGORITHM} must be 1, 2, or 3")
    for col in houses.columns:
        thetas = f(
            df.to_numpy(),
            houses[col].to_numpy().astype(int),
            epochs,
            learning_rate,
            col,
        )
        json_data[col] = thetas


def get_ready(
    df: pd.DataFrame, json_data: dict[str, dict[str, float]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare dataset for training with multi-factor logistic regression:
    drop nan values,
    drop unwanted features,
    standardize data,
    and create a new DataFrame for Hogwarts Houses.

    Parameters
    ----------
    df: our DataFrame.
    json_data: a dict to fill : for each features
        => a dict of mean and std.

    Returns
    -------
    df_stand: a new DataFrame with our selected features.
    houses_df: a new DataFrame for houses: each column
        represents one house (with 1 and 0).
    """
    df.dropna(inplace=True)
    houses_df = pd.get_dummies(df["Hogwarts House"])

    new_df = df.select_dtypes(["number"])
    new_df.drop(columns=FEATURES_TO_REMOVE, inplace=True)
    for col in new_df:
        mean = new_df[col].mean()
        std = new_df[col].std()
        json_data[col] = dict()
        json_data[col]["mean"] = mean
        json_data[col]["std"] = std
        new_df[col] = new_df[col].subtract(mean).divide(std)

    return (new_df, houses_df)


def main():
    """Main function"""
    try:
        epochs, learning_rate = check_args(HELP_TRAIN)
        dataset = load(sys.argv[1])
        pd.set_option("future.no_silent_downcasting", True)
        json_data = {}
        json_data["features"] = dict()
        json_data["thetas"] = dict()
        df, houses = get_ready(dataset, json_data["features"])
        train(df, houses, epochs, learning_rate, json_data["thetas"])

        with open("weigths.json", "w", encoding="utf8") as file:
            json.dump(json_data, file, indent=4)

    except ValueError:
        print("There is a problem in your input parameters.")
    except AssertionError as msg:
        print(f"{msg.__class__.__name__}: {msg}")


if __name__ == "__main__":
    main()
