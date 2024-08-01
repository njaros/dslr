import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sb


def pair_plot(path):
    """
    path: string -> it must be a relative or absolute path to a csv file.

    rule: create all the pair_plots
          for each feature between all the Hogwarts House.

    exceptions: raise exception if the file isn't readable
                or is to incorrect format, depending of pandas exceptions.
    """
    dataset = pd.read_csv(path)
    dataset.drop(
        columns=[
            "Index",
            "First Name",
            "Last Name",
            "Birthday",
            "Best Hand",
            "Defense Against the Dark Arts",
        ],
        inplace=True,
    )
    sb.pairplot(dataset, hue="Hogwarts House")
    plt.savefig("pair_plot.png")


if __name__ == "__main__":
    av = sys.argv
    ac = len(av)
    if ac == 1 or av[1] == "-h" or av[1] == "-help":
        print(
            """
            usage : python histogram.py path_csv_file

            rule : this program will create all the pair_scatter_plots
                   for each feature between all the Hogwarts House
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
            pair_plot(av[1])
        except UnicodeDecodeError as e:
            print(f"{e.__class__.__name__}: {e.args[4]}")
        except Exception as e:
            print(f"{e.__class__.__name__}: {e.args}")
