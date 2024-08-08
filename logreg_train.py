"""This program will train my models and
generates a file containing the weights
that will be used for the prediction."""

import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import load, standardize


FEATURES_TO_REMOVE = [
    "Index",
    "Arithmancy",
    "Defense Against the Dark Arts",
    # "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    # "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    # "Charms",
    "Flying",
    "Astronomy",
    "Herbology",
]


def sigmoid_function(X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Sigmoid function that returns probabilities of an individual
    being in a particular house"""
    z = np.array(X.dot(thetas), dtype=float)
    probabilities = 1 / (1 + np.exp(-z))
    # res = [1 if x > 0.5 else 0 for x in probabilities]

    return probabilities, z


def cost_function(sigma: np.ndarray, target: np.ndarray) -> float:
    """Log-likelihood : probability of observing the good results"""
    return -sum(target * np.log(sigma) + (1 - target) * np.log(1 - sigma)) / target.size


def find_gradients(X: np.ndarray, sigma: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Find gradients with partial differential of thetas"""
    X_t = np.transpose(X)  # ndarray(n-features + 1, m)
    return X_t.dot(np.subtract(sigma, target)) / target.size


def training(features: np.ndarray, target: np.ndarray):
    """This function will find the most optimized thetas to minimize cost function"""
    nb_iter = 100
    learning_rate = 3.0
    cost_history: list[float] = []

    m = features.shape[0]
    biais = np.ones((m, 1))
    X = np.hstack((features, biais))  # ndarray(m, n-features + 1)
    thetas = np.zeros(X.shape[1])  # ndarray(n-features + 1)

    for i in range(nb_iter):
        proba, z = sigmoid_function(X, thetas)  # ndarray(m)
        cost_history.append(cost_function(proba, target))
        gradients = find_gradients(X, proba, target)  # ndarray(n-features + 1)
        thetas = np.subtract(
            thetas, (learning_rate * gradients)
        )  # ndarray(n-features + 1)

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    axes[0].scatter(z, target, color="b", s=5, alpha=0.4)
    axes[0].scatter(z, proba, color="r", s=4, alpha=0.6)
    axes[1].plot(list(range(i + 1)), cost_history)
    axes[1].set_title("Cost history")
    plt.show()

    return list(thetas)


def get_ready(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare dataset for training : drop unwanted features,
    drop nan values, standardize data
    Return a DataFrame with our selected features,
    and a DataFrame for houses: each column represents one house (with 1 and 0)"""
    df.dropna(inplace=True)
    h = df["Hogwarts House"]
    hr = np.fromiter((1 if x == "Ravenclaw" else 0 for x in h), dtype=float)
    hg = np.fromiter((1 if x == "Gryffindor" else 0 for x in h), dtype=float)
    hs = np.fromiter((1 if x == "Slytherin" else 0 for x in h), dtype=float)
    hh = np.fromiter((1 if x == "Hufflepuff" else 0 for x in h), dtype=float)

    houses = np.stack((hr, hg, hs, hh), axis=1)
    h_names = ["Ravenclaw", "Gryffindor", "Slytherin", "Hufflepuff"]
    houses_df = pd.DataFrame(houses, columns=h_names)
    # print(houses_df)

    new_df = df.select_dtypes(["number"])
    new_df.drop(columns=FEATURES_TO_REMOVE, inplace=True)
    df_stand = standardize(new_df)

    return (df_stand, houses_df)


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            pd.set_option("future.no_silent_downcasting", True)
            df, houses = get_ready(dataset)
            weights = {}
            for col in houses.columns:
                thetas = training(df.to_numpy(), houses[col].to_numpy())
                weights[col] = thetas

            with open("weigths.json", "w", encoding="utf8") as file:
                json.dump(weights, file)


if __name__ == "__main__":
    main()
