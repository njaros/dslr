"""This script displays a scatter plot answering the question :
What are the two features that are similar ?"""

import sys
import matplotlib.pyplot as plt

from utils import load, standardize


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            dataset.drop(columns="Index", inplace=True)
            df = dataset.select_dtypes(["number"])
            df_stand = standardize(df)
            siz = len(df_stand.columns)
            _, axes = plt.subplots(nrows=6, ncols=13, figsize=(60, 30))
            blop = 0
            x = 0
            for i in range(siz):
                for j in range(i + 1, siz):
                    y = blop % 13
                    col1 = df.iloc[:, i]
                    col2 = df.iloc[:, j]
                    axes[x, y].scatter(
                        col1, col2, color="b", s=10, alpha=0.4, edgecolors="white"
                    )
                    axes[x, y].set_title(f"{col1.name} vs {col2.name}")
                    blop += 1
                    if blop % 13 == 0:
                        x += 1
            plt.savefig("ScatterPlot.png")


if __name__ == "__main__":
    main()
