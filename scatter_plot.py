"""This script displays a scatter plot answering the question :
What are the two features that are similar ?"""

import sys
import matplotlib.pyplot as plt

from utils import load


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            dataset.drop(columns="Index", inplace=True)
            df = dataset.select_dtypes(['number'])
            siz = len(df.columns)
            for i in range(siz):
                for j in range(i + 1, siz):
                    col1 = df.iloc[:, i]
                    col2 = df.iloc[:, j]
                    plt.scatter(col1, col2, color="b", s=10, alpha=0.4, edgecolors="white")
                    plt.title(f"{col1.name} vs {col2.name}")
                    plt.show()


if __name__ == "__main__":
    main()
