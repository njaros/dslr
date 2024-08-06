"""This program will train my models and
generates a file containing the weights
that will be used for the prediction."""

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import load, standardize
from describe import describe


FEATURES_TO_REMOVE = [
    "Index",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
    "Arithmancy",
    "Defense Against the Dark Arts",
    # "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
    "Astronomy",
    "Herbology"
]


def sigmoid_function(features: np.ndarray[np.float64], coefs: np.ndarray[np.float64], target):
    """Sigmoid function that returns probabilities of an individual
    being in a particular house"""
    z = features.dot(coefs)
    p = 1 / (1 + np.exp(-z))
    res = [1 for x in p if x > 0.5]
    blop = [1 for y in target if y == 1]
    print(len(blop))
    print(len(res))
    lg = np.log(p / (1 - p))

    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    axes[1, 0].scatter(features.T[0], target)
    axes[1, 0].set_title("Dataset with Astronomy")
    axes[0, 0].scatter(features.T[0], z)
    axes[0, 0].set_title("Linear function")
    axes[1, 1].scatter(features.T[0], p)
    axes[1, 1].set_title("Sigmoid function")
    axes[0, 1].scatter(features.T[0], lg)
    axes[0, 1].set_title("Log(odds)")
    plt.show()
# def cost_function()


def training(dataset: pd.DataFrame):
    """This function will find the most optimized thetas to minimize cost function"""
    # cost_history: list[float] = []
    a = dataset.drop(columns="Hogwarts House").to_numpy()
    b = np.ones((a.shape[0], 1))
    features = np.hstack((a, b))  # ndarray(m, 3)
    coefs = np.random.randn(features.shape[1], 1)  # ndarray(3, 1)
    target = dataset["Hogwarts House"].to_numpy()[:, np.newaxis]  # ndarray(m, 1)
    sigmoid_function(features, coefs, target)
    # print(hey)


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            pd.set_option("future.no_silent_downcasting", True)
            dataset.drop(columns=FEATURES_TO_REMOVE, inplace=True)
            dataset.dropna(inplace=True)
            df_stand = standardize(dataset)
            df_stand.replace(
                {
                    "Hogwarts House": {
                        "Ravenclaw": 0,
                        "Slytherin": 1,
                        "Gryffindor": 0,
                        "Hufflepuff": 0,
                    }
                },
                inplace=True,
            )
            # print(df_stand.head(10))
            # describe(df_stand)
            training(df_stand)


if __name__ == "__main__":
    main()
