"""This program take a dataset as a parameter and display information
for all numerical features"""

import sys
import math
import pandas as pd

from load_csv import load


def my_count(col: pd.Series) -> int:
    """Return size of this column"""
    cnt = 0
    for _ in col:
        cnt += 1
    return cnt


def my_min(col: pd.Series) -> float:
    """Return the minimun value of this column"""
    min_val = col[0]
    for x in col:
        if x < min_val:
            min_val = x
    return min_val


def my_max(col: pd.Series) -> float:
    """Return the maximum value of this column"""
    max_val = col[0]
    for x in col:
        if x > max_val:
            max_val = x
    return max_val


def my_mean(col: pd.Series) -> float:
    """Calculate Mean"""
    return sum(x for x in col) / col.count()


def my_median(col: pd.Series) -> float:
    """Calculate Median"""
    n = col.count()
    if n % 2 != 0:
        index = n / 2
        return col[int(index)]

    first_nb = col[int(n / 2) - 1]
    second_nb = col[int((n / 2))]
    return (first_nb + second_nb) / 2


def my_quartile_25(col: pd.Series) -> float:
    """Calculate Quartile (25% and 75%)"""
    quart = col.count() / 4
    return col[int(quart)]


def my_quartile_75(col: pd.Series) -> float:
    """Calculate Quartile (25% and 75%)"""
    quart = col.count() / 4
    return col[int(3 * quart)]


def my_variance(col: pd.Series) -> float:
    """Calculate Variance"""
    mean = my_mean(col)
    return sum(pow((x - mean), 2) for x in col) / col.count()


def my_std(col: pd.Series) -> float:
    """Calculate Standard Deviation"""
    return math.sqrt(my_variance(col))


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            # print(data.head(10))
            # print(data.describe())
            dscb = pd.DataFrame(index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
            for col in dataset.columns:
                if pd.api.types.is_numeric_dtype(dataset[col]) and col != "Index":
                    # data_lst = sorted([x for x in data[col].tolist() if not math.isnan(x)])
                    data = dataset[col].sort_values().dropna()
                    dscb[col] = [my_count(data),
                                 my_mean(data),
                                 my_std(data),
                                 my_min(data),
                                 my_quartile_25(data),
                                 my_median(data),
                                 my_quartile_75(data),
                                 my_max(data)
                                 ]
            print(dscb)


if __name__ == "__main__":
    main()
