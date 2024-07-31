"""Statistics library"""

from typing import Any


def calculate_mean(vector: list):
    """Calculate Mean"""
    return sum(x for x in vector) / len(vector)


def calculate_median(vector: list):
    """Calculate Median"""
    n = len(vector)
    if n % 2 != 0:
        index = n / 2
        return vector[int(index)]

    first_nb = vector[int(n / 2) - 1]
    second_nb = vector[int((n / 2))]
    return (first_nb + second_nb) / 2


def calculate_quartile(vector: list):
    """Calculate Quartile (25% and 75%)"""
    n = len(vector)
    quart = n / 4
    quartile = list()
    quartile.append(vector[int(quart)])
    quartile.append(vector[int(3 * quart)])

    return quartile


def calculate_variance(vector: list):
    """Calculate Variance"""
    mean = calculate_mean(vector)
    return sum(pow((x - mean), 2) for x in vector) / len(vector)


def calculate_standard_deviation(vector: list):
    """Calculate Standard Deviation"""
    return calculate_variance(vector) ** 0.5


def ft_statistics(*args: Any, **kwargs: Any) -> None:
    """This function takes an indeterminate number of numbers
    and according to what is asked in the kwargs it calculates:
    Mean, Median, Quartile, Standard Deviation and Variance of this numbers"""

    vector = list(args)
    assert all(
        not isinstance(n, bool) and isinstance(n, (float, int)) for n in vector
    ), "Your list must contain only float or int."
    vector.sort()

    for _, value in kwargs.items():
        try:
            if value == "mean":
                print(f"mean: {calculate_mean(vector)}")
            if value == "median":
                print(f"median: {calculate_median(vector)}")
            if value == "quartile":
                print(f"quartile: {calculate_quartile(vector)}")
            if value == "std":
                print(f"std: {calculate_standard_deviation(vector)}")
            if value == "var":
                print(f"var: {calculate_variance(vector)}")
        except Exception:
            print("ERROR")
