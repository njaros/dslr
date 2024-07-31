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
