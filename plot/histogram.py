"""This script displays a histogram answering the question :
Which Hogwarts course has a homogeneous score distribution
between all four houses?"""

import sys
import matplotlib.pyplot as plt
import pandas as pd

from tools.logreg_utils import load, standardize


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            dataset.drop(columns="Index", inplace=True)
            df_stand = standardize(dataset)
            grouped = df_stand.groupby("Hogwarts House")
            colors = ["b", "g", "r", "y"]
            _, axes = plt.subplots(nrows=3, ncols=5, figsize=(30, 20))
            blop = 0
            x = 0
            for col in dataset.columns:
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    y = blop % 5
                    for i, (_, group) in enumerate(grouped):
                        axes[x, y].hist(group[col], alpha=0.5, color=colors[i])
                        axes[x, y].set_title(col)
                    blop += 1
                    if blop % 5 == 0:
                        x += 1
            plt.savefig("Histogram.png")


if __name__ == "__main__":
    main()
