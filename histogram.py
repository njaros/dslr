import pandas as pd
import matplotlib.pyplot as plt
import sys


def histogram(path):
    """
    path: string -> it must be a relative or absolute path to a csv file.

    rule: display histogram comparing each feature of the different values
          of the first column.

    exceptions: raise exception if the file isn't readable
                or is to incorrect format, depending of pandas exceptions.
    """
    dataset = pd.read_csv(path)
    dataset.drop(
        columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"],
        inplace=True,
    )
    houseSet = set()
    col_one = dataset.columns[0]
    for house in dataset[col_one]:
        houseSet.add(house)
    df_per_house: list[(str, pd.DataFrame)] = []
    for house in houseSet:
        df_per_house.append((house, dataset[dataset[col_one] == house]))
    fig, axs = plt.subplots(
        int((len(dataset.columns) - 1) / 4) + 1, 4, figsize=(20, 12)
    )
    fig.tight_layout()
    fig.suptitle("comparisons between each house")
    loop = 0
    colors = ["red", "blue", "green", "yellow", "brown", "purple", "black"]
    for col in dataset.columns[1:]:
        x = int(loop / 4)
        y = loop % 4
        axs[x, y].set_title(col)
        color_idx = 0
        for df in df_per_house:
            axs[x, y].hist(df[1][col], alpha=0.25, label=df[0], color=colors[color_idx])
            color_idx += 1
        loop += 1
        axs[x, y].legend()
    plt.savefig("histo.png")


if __name__ == "__main__":
    av = sys.argv
    ac = len(av)
    if ac == 1 or av[1] == "-h" or av[1] == "-help":
        print(
            """
            usage : python histogram.py path_csv_file

            rule : this program will returns an histogram comparing each
                   columns of a dataset between all possible values of the
                   first column.
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
            histogram(av[1])
        except UnicodeDecodeError as e:
            print(f"{e.__class__.__name__}: {e.args[4]}")
        except Exception as e:
            print(f"{e.__class__.__name__}: {e.args}")
