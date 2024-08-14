"""This module defines project-level constants."""

FEATURES_TO_REMOVE = [
    "Index",
    "Arithmancy",
    "Defense Against the Dark Arts",
    # "Divination",
    # "Muggle Studies",
    "Ancient Runes",
    # "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
    "Astronomy",
    "Herbology",
]

HELP_TRAIN = """
            usage : python logreg_train.py path_csv_file (epochs learning_rate).

            rule : this program will train a model to predict
                   the house membership of a Hogwarts student
                   and generates a file with weights for each house
                   named weights.json.
            """

HELP_SCORE = """
            usage: python model_score.py path_to_dataset.

            rule: this program will calculate the efficiency of a logistic
                   regression algorithm using a train dataset.

            how: program split into two part the dataset.
                 > one part for train the model using the algorithm
                 > the other part for calculate the efficiency of the model
            """

HELP_PREDICT = """
            usage : python logreg_predict.py path_csv_file.

            rule : this program will predict the house membership
                   for each line in a dataset containing a set of
                   school student in Hogwarts.
            """
