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

def flatten(probs):
    new_p = []
    for p in probs:
        if p >= .5:
            new_p.append(1)
        else:
            new_p.append(0)
    return new_p

def converts(actuals,name):
    print(actuals)
    #actuals = actuals[0]
    converted = []
    for a in actuals:
        if a == 'Setosa':
            converted.append(1)
        else:
            converted.append(0)
    return converted

def confusion_matrix(predicted,actual):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(predicted)):
        if actual[i] == 0 and predicted[i] == 0:
            TN += 1
        elif actual[i] == 1 and predicted[i] == 1:
            TP += 1
        elif actual[i] == 1 and predicted[i] == 0:
            FN += 1
        elif actual[i] == 0 and predicted[i] == 1:
            FP += 1
    print(TP,FN)
    print(FP,TN)


def main():
    header = None
    data = []

    # UPDATE PATH - WILL BREAK IF YOU HAVE NOT SET THE CORRECT PATH
    iris_data_path = '/Users/Matthew/Documents/GitHub/MachineLearningClass/Homework1/iris_data.csv'

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

    # split up test and training set
    test, train = split_test_train(data)
    one_hot_truth_train = [1 if sample[4] == 'Setosa' else 0 for sample in train]
    one_hot_truth_test = [1 if sample[4] == 'Setosa' else 0 for sample in test]

    features = train[:, 0:4].astype(float)
    weights = np.zeros(features.shape[1])

    num_iter = 10000
    lr = .01
    losses = []
    for idx in range(num_iter):
        # Determine the weights
        weighted_features = np.dot(features, weights)
        # Calculate probablity
        probabilities = sigmoid(weighted_features)
        # Calculate loss
        loss = cost(np.array(one_hot_truth_train), probabilities)
        # Determine gradient
        gradient = calculate_gradient(features, probabilities, one_hot_truth_train)
        # Update weights
        weights = weights - lr*gradient
        # Calculate average loss
        avg_loss = np.average(loss)
        print(avg_loss)
        if idx > 100:
            losses.append((idx, avg_loss))


    # Testing Weights
    test_features = test[:, 0:4].astype(float)
    test_probs = sigmoid(np.dot(test_features, weights))
    # Flatten to probablities to 1 if over or equal to .5 - to 0 if under .5
    flatten_probs = flatten(test_probs)
    convert_actual = converts(test[:, 4:5],'Setosa')
    # Determine and output condusion matrix
    print(convert_actual)
    print(flatten_probs)
    confusion_matrix(flatten_probs, convert_actual)

    # Setting up plot
    # Losses - [iterations(EPOCHS), Cost]
    x = [val[0] for val in losses]
    y = [val[1] for val in losses]

    plt.plot(x, y)
    plt.title("Setosa")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.show()




if __name__ == '__main__':
    main()
