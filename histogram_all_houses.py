"""This script displays a histogram answering the question :
Which Hogwarts course has a homogeneous score distribution
between all four houses?"""

import sys
import matplotlib.pyplot as plt

from tools.logreg_utils import load, standardize


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            df = dataset.select_dtypes(["number"])
            df.drop(columns=["Index"], inplace=True)
            df_stand = standardize(df)
            _, axes = plt.subplots(nrows=3, ncols=5, figsize=(30, 20))
            blop = 0
            x = 0
            for col in df_stand.columns:
                y = blop % 5
                axes[x, y].hist(df_stand[col])
                axes[x, y].set_title(col)
                blop += 1
                if blop % 5 == 0:
                    x += 1
        plt.savefig("Histogram_all_houses.png")


if __name__ == "__main__":
    main()
