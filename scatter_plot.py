import pandas as pd
import sys
import matplotlib.pyplot as plt


def scatter(path):
    """
    path: string -> it must be a relative or absolute path to a csv file.

    rule: create scatter plots for each feature in the dataset.

    exceptions: raise exception if the file isn't readable
                or is to incorrect format, depending of pandas exceptions.
    """
    dataset = pd.read_csv(path)
    dataset.drop(
        columns=[
            "First Name",
            "Last Name",
            "Birthday",
            "Index",
            "Hogwarts House",
            "Best Hand",
        ],
        inplace=True,
    )
    fig, axs = plt.subplots(
        len(dataset.columns), len(dataset.columns), figsize=(20, 12)
    )
    fig.tight_layout()
    fig.suptitle("scatter plots -> looking for straight lines")
    referee = dataset.columns[1]
    x = 0
    for referee in dataset.columns:
        y = 0
        for col in dataset.columns:
            if col != referee:
                axs[x, y].scatter(
                    dataset[referee], dataset[col], c=[0.1 * x for x in range(1600)]
                )
                axs[x, y].set_xlabel(referee)
                axs[x, y].set_ylabel(col)
                y += 1
        x += 1
    plt.savefig("scatter.png")


if __name__ == "__main__":
    av = sys.argv
    ac = len(av)
    if ac == 1 or av[1] == "-h" or av[1] == "-help":
        print(
            """
            usage : python scatter_plot.py path_csv_file

            rule : this program will create scatter plots for each
                   feature in a dataset.
            """
        )
    elif ac != 2:
        print(
            """
            Only one argument required. try -help for details.
            """
        )
    else:
        try:
            scatter(av[1])
        except UnicodeDecodeError as e:
            print(f"{e.__class__.__name__}: {e.args[4]}")
        except Exception as e:
            print(f"{e.__class__.__name__}: {e.args}")
