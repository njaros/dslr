"""This module defines project-level constants."""

FEATURES_TO_REMOVE = [
    "Index",
    "Arithmancy",
    "Defense Against the Dark Arts",
    # "Divination",
    # "Muggle Studies",
    "Ancient Runes",
    # "History of Magic",
    #     "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    #     "Charms",
    #     "Flying",
    #     "Astronomy",
    #     "Herbology",
]

# for the algorithm to choose :
#      1 = batch
#      2 = stochastic
#      3 = mini batch
CHOOSEN_ALGORITHM = 2

# If choosen algorithm is 3 (mini batch), you have to define
# how many batch to do
NUMBER_OF_BATCH = 50

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
