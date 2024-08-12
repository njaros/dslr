"""This script displays a pair plot or
scatter plot matrix"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns

from tools.logreg_utils import load


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            dataset.drop(
                columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"],
                inplace=True,
            )
            palettedict = {
                "Ravenclaw": "blue",
                "Slytherin": "green",
                "Gryffindor": "red",
                "Hufflepuff": "orange",
            }
            sns.pairplot(dataset, hue="Hogwarts House", palette=palettedict)
            plt.savefig("PairPlot.png")


if __name__ == "__main__":
    main()
