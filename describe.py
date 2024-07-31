import pandas as pd
import numpy as np
import sys


def describe(path):
    """
    path: string -> it must be a relative or absolute path to a csv file.

    rule: print statistic descriptions for each columns of the dataset

    exceptions: raise exception if the file isn't readable
                or is to incorrect format, depending of pandas exceptions.
    """
    dataset = pd.read_csv(path)
    dataset.replace(
        {
            "Hogwarts House": {
                "Ravenclaw": 0,
                "Slytherin": 1,
                "Gryffindor": 2,
                "Hufflepuff": 3,
            },
            "Best Hand": {"Left": 1, "Right": 0},
        },
        inplace=True,
    )
    dataset.drop(columns=["First Name", "Last Name", "Birthday", "Index"], inplace=True)
    describe_df = pd.DataFrame(
        index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    )
    for column in dataset.columns:
        vals = dataset[column]
        print(column, vals.mean())
        describe_df[column] = [
            vals.count(),
            vals.mean(),
            vals.std(),
            vals.min(),
            vals.quantile(0.25),
            vals.median(),
            vals.quantile(0.75),
            vals.max(),
        ]
    print(describe_df)


if __name__ == "__main__":
    av = sys.argv
    ac = len(av)
    if ac == 1 or av[1] == "-h" or av[1] == "-help":
        print(
            """
            usage : python describe.py path_csv_file

            rule : this program will returns a statistical description
                   of each columns of a dataset file formatted as a csv.
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
            describe(av[1])
        except UnicodeDecodeError as e:
            print(f"{e.__class__.__name__}: {e.args[4]}")
        except Exception as e:
            print(f"{e.__class__.__name__}: {e.args}")
