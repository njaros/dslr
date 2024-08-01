import sys
import matplotlib.pyplot as plt
import pandas as pd

from load_csv import load


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        dataset = load(sys.argv[1])
        if dataset is not None:
            dataset.drop(columns="Index", inplace=True)
            print(dataset.head(10))

            grouped = dataset.groupby("Hogwarts House")
            colors = ["b", "g", "r", "y"]
            _, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
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
            plt.show()


if __name__ == "__main__":
    main()
