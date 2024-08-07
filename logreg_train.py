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
    "Herbology",
]


def sigmoid_function(features: np.ndarray[np.float64], coefs: np.ndarray[np.float64]):
    """Sigmoid function that returns probabilities of an individual
    being in a particular house"""
    z = np.array(features.dot(coefs), dtype=float)
    probabilities = 1 / (1 + np.exp(-z))
    # res = [1 if x > 0.5 else 0 for x in probabilities]

    return probabilities


def cost_function(sigma, target):
    """Log-likelihood : probability of observing the good results"""
    return -sum(target * np.log(sigma) + (1 - target) * np.log(1 - sigma)) / target.size


def find_gradients(features, sigma, target):
    """Find gradients with partial differential of thetas"""
    features_T = np.transpose(features)  # ndarray(n-features + 1, m)
    return features_T.dot(np.subtract(sigma, target)) / target.size


def training(dataset: pd.DataFrame):
    """This function will find the most optimized thetas to minimize cost function"""
    cost_history: list[float] = []
    nb_iter = 100
    learning_rate = 3

    a = dataset.drop(columns="Hogwarts House").to_numpy()
    b = np.ones((a.shape[0], 1))
    features = np.hstack((a, b))  # ndarray(m, n-features + 1 for biais)
    coefs = np.zeros(features.shape[1])  # ndarray(n-features + 1)
    # coefs = np.array([-1.19, -8])
    target = dataset["Hogwarts House"].to_numpy()  # ndarray(m)

    for i in range(nb_iter):
        proba = sigmoid_function(features, coefs)  # ndarray(m)
        cost_history.append(cost_function(proba, target))
        gradients = find_gradients(features, proba, target)  # ndarray(n-features + 1)
        coefs = np.subtract(
            coefs, (learning_rate * gradients)
        )  # ndarray(n-features + 1)

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    axes[0].scatter(features.T[0], target, color="b", s=5, alpha=0.4)
    axes[0].scatter(features.T[0], proba, color="r", s=4, alpha=0.6)
    axes[0].set_title("Slytherin relative to Divination")
    axes[1].plot(list(range(i + 1)), cost_history)
    axes[1].set_title("Cost history")
    plt.show()


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
            training(df_stand)


if __name__ == "__main__":
    main()
