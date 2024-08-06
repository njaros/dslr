"""This program take a dataset as a parameter and display information
for all numerical features"""

import sys
import math
import pandas as pd

from utils import load


def my_min(col: pd.Series) -> float:
    """Return the minimun value of this column"""
    assert len(col) != 0, f"{col.name} is empty."
    min_val = col[0]
    for x in col:
        if x < min_val:
            min_val = x
    return min_val


def my_max(col: pd.Series) -> float:
    """Return the maximum value of this column"""
    assert len(col) != 0, f"{col.name} is empty."
    max_val = col[0]
    for x in col:
        if x > max_val:
            max_val = x
    return max_val


def my_mean(col: pd.Series) -> float:
    """Calculate Mean"""
    assert len(col) != 0, f"{col.name} is empty."
    return sum(x for x in col) / len(col)


def my_median(col: pd.Series) -> float:
    """Calculate Median"""
    assert len(col) != 0, f"{col.name} is empty."
    m = len(col)
    if m % 2 != 0:
        return col.iloc[m // 2]

    first_nb = col.iloc[m // 2 - 1]
    second_nb = col.iloc[m // 2]
    return (first_nb + second_nb) / 2


def my_quartile_25(col: pd.Series) -> float:
    """Calculate Quartile (25%)"""
    assert len(col) != 0, f"{col.name} is empty."
    mid = len(col) // 2
    return my_median(col.iloc[:mid])


def my_quartile_75(col: pd.Series) -> float:
    """Calculate Quartile (75%)"""
    assert len(col) != 0, f"{col.name} is empty."
    mid = len(col) // 2
    return my_median(col.iloc[mid:])


def my_variance(col: pd.Series) -> float:
    """Calculate Variance"""
    mean = my_mean(col)
    return sum(pow((x - mean), 2) for x in col) / len(col)


def my_std(col: pd.Series) -> float:
    """Calculate Standard Deviation"""
    return math.sqrt(my_variance(col))


def describe(dataset: pd.DataFrame):
    """Display information for all numerical features"""
    dscb = pd.DataFrame(
        index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    )
    for col in dataset.columns:
        try:
            if pd.api.types.is_numeric_dtype(dataset[col]) and col != "Index":
                # data_lst = sorted([x for x in data[col].tolist() if not math.isnan(x)])
                data = dataset[col].sort_values().dropna()
                dscb[col] = [
                    len(data),
                    my_mean(data),
                    my_std(data),
                    my_min(data),
                    my_quartile_25(data),
                    my_median(data),
                    my_quartile_75(data),
                    my_max(data),
                ]
        except AssertionError as msg:
            print(f"{msg.__class__.__name__}: {msg}")
    print(dscb)


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            # print(data.head(10))
            # print(data.describe())
            describe(dataset)

if __name__ == "__main__":
    main()
