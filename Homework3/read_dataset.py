import csv
import numpy as np
import random
from matplotlib import pyplot as plt


def split_test_train(data):
    test_split = np.random.choice(len(data), size=int(len(data)*.2), replace=False)
    test = data[test_split]
    train_split = np.array([idx for idx in range(len(data)) if idx not in test_split])
    train = data[train_split]

    return test, train


def main():
    header = None
    data = []

    # UPDATE PATH - WILL BREAK IF YOU HAVE NOT SET THE CORRECT PATH
    iris_data_path = 'iris_data.csv'

    label_2_id = {'Iris-setosa': 1, 'Iris-versicolor': 0, 'Iris-virginica': 0}
    id_2_label = {1: 'Setosa', 0: 'NotSetosa'}

    with open(iris_data_path, 'r') as f:
        tmp = csv.reader(f)
        for idx, row in enumerate(tmp):
            if idx == 0:
                header = {row_idx: row_val for row_idx, row_val in enumerate(row)}
            elif len(row) == 0:
                continue
            else:
                row[4] = label_2_id[row[4]]
                data.append(row)
    data = np.array(data)

    # split up test and training set
    test, train = split_test_train(data)

    return train, test


if __name__ == '__main__':
    main()
