import sys
import matplotlib.pyplot as plt

from load_csv import load


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Incorrect input.")
    else:
        data = load(sys.argv[1])
        if data is not None:
            print(data.head(10))
            # print(data.describe())


if __name__ == "__main__":
    main()
