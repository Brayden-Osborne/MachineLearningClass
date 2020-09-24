import csv
import numpy as np
import random
from matplotlib import pyplot as plt


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def prob(input_features, weights):
    scores = np.dot(input_features, weights)
    return sigmoid(scores)


def cost(onehot_label, prob):
    return -onehot_label * np.log(prob) + (1 - onehot_label) * np.log(1 - prob)


def split_test_train(data):
    test_split = np.random.choice(len(data), size=int(len(data)*.2), replace=False)
    test = data[test_split]
    train_split = np.array([idx for idx in range(len(data)) if idx not in test_split])
    train = data[train_split]

    return test, train


def calculate_gradient(features, probabilities, truth_one_hot):
    return np.dot(features.T, (probabilities - truth_one_hot)) / len(truth_one_hot)


def main():
    header = None
    data = []
    iris_data_path = '/Users/brayd/Documents/GitHub/MachineLearningClass/Homework1/iris_data.csv'

    label_2_id = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 1}
    id_2_label = {0: 'Setosa', 1: 'NotSetosa'}

    with open(iris_data_path, 'r') as f:
        tmp = csv.reader(f)
        for idx, row in enumerate(tmp):
            if idx == 0:
                header = {row_idx: row_val for row_idx, row_val in enumerate(row)}
            elif len(row) == 0:
                continue
            else:
                row[4] = id_2_label[label_2_id[row[4]]]
                data.append(row)
    data = np.array(data)
    print(data)
    print(header)
    test, train = split_test_train(data)

    one_hot_truth_train = [1 if sample[4] == 'Setosa' else 0 for sample in train]
    one_hot_truth_test = [1 if sample[4] == 'Setosa' else 0 for sample in test]
    features = train[:, 0:4].astype(float)
    weights = np.zeros(features.shape[1])

    num_iter = 100000
    lr = .01
    losses = []
    for idx in range(num_iter):
        weighted_features = np.dot(features, weights)
        probabilities = sigmoid(weighted_features)
        loss = cost(np.array(one_hot_truth_train), probabilities)
        gradient = calculate_gradient(features, probabilities, one_hot_truth_train)
        weights = weights - lr*gradient
        avg_loss = np.average(loss)
        print(avg_loss)
        if idx > 100:
            losses.append((idx, avg_loss))

    x = [val[0] for val in losses]
    y = [val[1] for val in losses]
    plt.plot(x, y)
    plt.show()
    test_features = test[:, 0:4].astype(float)
    test_probs = sigmoid(np.dot(test_features, weights))
    plt.plot(test_probs)
    threshold = .9
    test_preds = [1 if val > threshold else 0 for val in test_probs]
    x=1



if __name__ == '__main__':
    main()
