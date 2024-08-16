import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from logreg_train import get_ready, train
from logreg_predict import predict
from tools.logreg_utils import load, check_args
from tools.logreg_config import HELP_SCORE
import seaborn as sb
import matplotlib.pyplot as plt


def confusions_matrices(Y: pd.DataFrame, P: pd.DataFrame, houses: list[str]) -> dict:
    """confusions_matrices function: compare the predictions with
    the truth to fill the confusion matrices

    Parameters
    ----------
    Y: DataFrame representing the truth
    P: DataFrame representing the predictions
    houses: list of strings, each is a Hogwarts House and is
            an outcome to predict.
            Then a matrix has to be done for each element of that list

    Returns
    -------
    matrices: a dictionnary of matrices, each matrix is a dictionnary
              of Tp, Fn, Fp and Tn values.
              matrices: {
                "house1": {
                    "Tp": Tp_value,
                    "Fn": Fn_value,
                    "Fp": Fp_value,
                    "Tn": Tn_value
                },
                "house2": ....
              }
    """

    matrices = {}
    for house in houses:
        matrices[house] = {"Tp": 0, "Fn": 0, "Fp": 0, "Tn": 0}
    for p, y in zip(P["Hogwarts House"], Y["Hogwarts House"]):
        if p == y:
            for house in houses:
                if p != house:
                    matrices[house]["Tn"] += 1
                else:
                    matrices[house]["Tp"] += 1
        else:
            matrices[p]["Fp"] += 1
            matrices[y]["Fn"] += 1
            for house in houses:
                if (p != house) and (y != house):
                    matrices[house]["Tn"] += 1
    return matrices


def score_calculator(
    Tp: int, Fn: int, Fp: int, Tn: int
) -> tuple[list[list[int]], int, float, float, float]:
    """score_calculator function: calculate precision, recall and f1 score

    Parameters
    ----------
    Tp: int: true positive value
    Fn: int: false negative value
    Fp: int: false positive value
    Tp: int: true negative value

    Returns
    -------
    a tuple of values:
        matrix: a matrix composed by [
                                    [Tp, Fn],
                                    [Fp, Tn]]
        total: int: the total of the element the model had to find
        recall: float: the percentage of the total the model found
        precision: float: the percentage of success when model says
            he found an element
        f1_score: float: the harmonic mean of recall and precision
            as a percentage"""
    matrix = [[Tp, Fn], [Fp, Tn]]
    total = Tp + Fn
    if Tp + Fp != 0:
        precision = 100 * round(Tp / (Tp + Fp), 2)
    else:
        precision = 100.0
    if total != 0:
        recall = 100 * round(Tp / total, 2)
    else:
        recall = 100.0
    if precision + recall != 0:
        f1_score = round((2 * precision * recall) / (precision + recall), 2)
    else:
        f1_score = 0.0
    return matrix, total, recall, precision, f1_score


def matrices_interpretor(matrices: dict[str, dict[str, int]], houses: list[str]):
    """matrices_interpretor function: creates png files for
    confusion matrix of each house and write in standard output
    the model score.

    Parameters
    ----------
    matrices: dictionnaries of confusion matrix, each matrix is a dictionnary
        matrices: {
            "house1": {
                "Tp": Tp_value,
                "Fn": Fn_value,
                "Fp": Fp_value,
                "Tn": Tn_value
            },
            "house2": ....
          }
    houses: a list of each Hogwarts House"""
    accuracy = 0
    for house in houses:
        plt.clf()
        datas = matrices[house]
        accuracy += datas["Tp"]
        matrix, total, recall, precision, f1_score = score_calculator(
            datas["Tp"], datas["Fn"], datas["Fp"], datas["Tn"]
        )
        print(f"===={house} model score====")
        print(f"for {total=} {house} members, the model recovered {recall=}% of them")
        print(
            f"when model says he is a {house} member you can trust him at {precision=}%"
        )
        print(f"The model has a {f1_score=}% for {house}")
        print()
        sb.heatmap(matrix, annot=True, fmt=".0f")
        plt.title(f"confusion matrix for {house}")
        plt.savefig(f"scores/{house}_confusion_matrix.png")
    accuracy = 100 * round(
        accuracy / sum([matrices[house][k] for k in matrices[house]]), 2
    )
    print(f"Finally the model as an {accuracy=}%")


def score():
    """score function:
    - prepare the two datasets: one for train model, the other
    to evaluate the score of the model

    - train model

    - evaluate the score by creating confusion matrix for each Hogwarts House

    - display the score of the model in standard output
    """
    epochs, learning_rate = check_args(HELP_SCORE)
    dataset = load(sys.argv[1])
    pd.set_option("future.no_silent_downcasting", True)

    # Prepare the two datasets
    train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=2)

    # Train the model
    model = {}
    model["features"] = dict()
    model["thetas"] = dict()
    df, houses = get_ready(train_df, model["features"])
    train(df, houses, epochs, learning_rate, model["thetas"])

    # Evaluates the score of the trained model
    res_test = test_df[["Index", "Hogwarts House"]]
    predict_test = predict(test_df, model)
    houses = ["Slytherin", "Gryffindor", "Hufflepuff", "Ravenclaw"]
    matrices = confusions_matrices(res_test, predict_test, houses)

    # Display the results
    matrices_interpretor(matrices, houses)


if __name__ == "__main__":
    try:
        score()
    except UnicodeDecodeError as e:
        print(f"{e.__class__.__name__}: {e.args[4]}")
    except ValueError:
        print("There is a problem in your input parameters.")
    except AssertionError as msg:
        print(f"{msg.__class__.__name__}: {msg}")
    except Exception as e:
        print(f"{e.__class__.__name__}: {e.args}")
