"""this program take a dataset as a parameter and display information
for all numerical features"""

import os
import sys
import math
import pandas as pd


def my_mean(vector: list) -> float:
    """Calculate Mean"""
    return sum(x for x in vector) / 1600


def my_median(vector: list) -> float:
    """Calculate Median"""
    n = len(vector)
    if n % 2 != 0:
        index = n / 2
        return vector[int(index)]

    first_nb = vector[int(n / 2) - 1]
    second_nb = vector[int((n / 2))]
    return (first_nb + second_nb) / 2


def my_quartile_25(vector: list) -> float: 
    """Calculate Quartile (25% and 75%)"""
    quart = len(vector) / 4
    return vector[int(quart)]


def my_quartile_75(vector: list) -> float:
    """Calculate Quartile (25% and 75%)"""
    quart = len(vector) / 4
    return vector[int(3 * quart)]


def my_variance(vector: list) -> float:
    """Calculate Variance"""
    mean = my_mean(vector)
    return sum(pow((x - mean), 2) for x in vector) / len(vector)


def my_std(vector: list) -> float:
    """Calculate Standard Deviation"""
    return my_variance(vector) ** 0.5


def load(path: str):
    """Load a data file using pandas library"""
    try:
        assert isinstance(path, str), "your path is not valid."
        assert os.path.exists(path), "your file doesn't exist."
        assert os.path.isfile(path), "your 'file' is not a file."
        assert path.lower().endswith(".csv"), "file format is not .csv."
        data = pd.read_csv(path)
        print(f"Loading dataset of dimensions {data.shape}")
        return data
    except AssertionError as msg:
        print(f"{msg.__class__.__name__}: {msg}")
        return None


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        data = load(sys.argv[1])
        if data is not None:
            # print(data.head(10))
            # print(data.describe())
            dscb = pd.DataFrame(index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]) and col == "Charms":
                    data_lst = sorted([x for x in data[col].tolist() if not math.isnan(x)])
                    dscb[col] = [len(data_lst),
                                 my_mean(data_lst),
                                 my_std(data_lst),
                                 min(data_lst),
                                 my_quartile_25(data_lst),
                                 my_median(data_lst),
                                 my_quartile_75(data_lst),
                                 max(data_lst)
                                 ]

            print(dscb)
if __name__ == "__main__":
    main()
