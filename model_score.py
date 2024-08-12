import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from logreg_train import get_ready, train
from logreg_predict import predict
from tools.logreg_utils import load, check_args
from tools.logreg_config import HELP_SCORE
import seaborn as sb
import matplotlib.pyplot as plt


def fill_confusions_matrices(
    Y: pd.DataFrame, P: pd.DataFrame, matrices: dict, houses: list[str]
):
    """fill_confusions_matrices"""
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


def score():
    """score function

    - prepare the two datasets: one for train model, the other
    to evaluate the score of the model

    - train model

    - evaluate the score
    """
    epochs, learning_rate = check_args(HELP_SCORE)
    dataset = load(sys.argv[1])
    pd.set_option("future.no_silent_downcasting", True)
    model = {}
    model["features"] = dict()
    model["thetas"] = dict()
    train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=42)
    df, houses = get_ready(train_df, model["features"])
    train(df, houses, epochs, learning_rate, model["thetas"])
    # X, Y = prepare_df(train_df, model["features"])
    # train(X, Y, model["thetas"])
    res_test = test_df[["Index", "Hogwarts House"]]
    predict_test = predict(test_df, model)
    houses = ["Slytherin", "Gryffindor", "Hufflepuff", "Ravenclaw"]
    confusion_matrices = {}
    for house in houses:
        confusion_matrices[house] = {"Tp": 0, "Fn": 0, "Fp": 0, "Tn": 0}
    fill_confusions_matrices(res_test, predict_test, confusion_matrices, houses)
    accuracy = 0
    for house in houses:
        plt.clf()
        datas = confusion_matrices[house]
        Tp = datas["Tp"]
        Fn = datas["Fn"]
        Fp = datas["Fp"]
        Tn = datas["Tn"]
        accuracy += Tp
        matrix = [[Tp, Fn], [Fp, Tn]]
        total = Tp + Fn
        precision = 100 * round(Tp / (Tp + Fp), 2)
        recall = 100 * round(Tp / total, 2)
        f1_score = round((2 * precision * recall) / (precision + recall), 2)
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
    accuracy = 100 * round(accuracy / len(predict_test), 2)
    print(f"Finally the model as an {accuracy=}%")


if __name__ == "__main__":
    try:
        score()
    except UnicodeDecodeError as e:
        print(f"{e.__class__.__name__}: {e.args[4]}")
    except ValueError:
        print("There is a problem in your input parameters.")
    except AssertionError as msg:
        print(f"{msg.__class__.__name__}: {msg}")
        # except Exception as e:
        #     print(f"{e.__class__.__name__}: {e.args}")
