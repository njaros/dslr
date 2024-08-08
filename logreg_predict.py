import sys
import json
import pandas as pd
import numpy as np


from utils import load, standardize


FEATURES_TO_REMOVE = [
    "Index",
    "Hogwarts House",
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


def predict(X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Sigmoid function that returns probabilities of an individual
    being in a particular house"""
    z = np.array(X.dot(thetas), dtype=float)
    probabilities = 1 / (1 + np.exp(-z))
    # res = [1 if x > 0.5 else 0 for x in probabilities]

    return probabilities


def get_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get data for predictions"""
    new_df = df.select_dtypes(["number"])
    new_df.drop(columns=FEATURES_TO_REMOVE, inplace=True)
    df.dropna(inplace=True)
    df_stand = standardize(new_df)

    return df_stand


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            try:
                with open(sys.argv[2], "rb") as file:
                    weights = json.load(file)
                features = get_data(dataset).to_numpy()
                m = features.shape[0]
                biais = np.ones((m, 1))
                X = np.hstack((features, biais))
                predictions = pd.DataFrame(columns=weights.keys())
                for house, thetas in weights.items():
                    p = predict(X, np.array(thetas))
                    predictions[house] = p.tolist()
                predictions['Hogwarts House'] = predictions.idxmax(axis=1)
                new_df = predictions.reset_index()[['index', 'Hogwarts House']]
                new_df.to_csv('houses.csv', index=False)
            except OSError:
                print("There is a problem with your jsonfile. You must first train your model.")
            except ValueError:
                print("There is a problem in you thetas file.")


if __name__ == "__main__":
    main()
