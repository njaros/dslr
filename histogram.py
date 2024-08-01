import sys
import matplotlib.pyplot as plt

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
            ravenclaw_data = grouped.get_group("Ravenclaw")
            # print(ravenclaw_data)
            slytherin_data = grouped.get_group("Slytherin")
            gryffindor_data = grouped.get_group("Gryffindor")
            hufflepuff_data = grouped.get_group("Hufflepuff")
            _, axes = plt.subplots(nrows=2, ncols=7, figsize=(40, 4))
            ravenclaw_data.hist(
                ax=axes.ravel(),
                alpha=0.5,
                color="b",
            )
            slytherin_data.hist(
                ax=axes.ravel(),
                alpha=0.5,
                color="g",
            )
            gryffindor_data.hist(
                ax=axes.ravel(),
                alpha=0.5,
                color="r",
            )
            hufflepuff_data.hist(
                ax=axes.ravel(),
                alpha=0.5,
                color="y",
            )
            plt.legend(labels=("Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"))
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
