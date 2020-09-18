import csv
import numpy as np


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def main():
    header = None
    data = []
    iris_data_path = '/Users/brayd/Documents/GitHub/MachineLearningClass/Homework1/iris_data.csv'

    label_2_id = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    id_2_label = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

    with open(iris_data_path, 'r') as f:
        tmp = csv.reader(f)
        for idx, row in enumerate(tmp):
            if idx == 0:
                header = {row_idx: row_val for row_idx, row_val in enumerate(row)}
            elif len(row) == 0:
                continue
            else:
                data.append(row)
    data = np.array(data)
    print(data)
    print(header)


if __name__ == '__main__':
    main()
