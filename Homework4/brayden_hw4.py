from Homework4.createDataset import read_dataset
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class Model:
    def __init__(self, num_epochs=10, initial_lr=.00001, lr_decay_rate=None, t=None, d_low=None, d_high=None):
        self.weights = np.array([.5, .5, .5])
        self.num_epochs = num_epochs
        self.learning_rate = initial_lr
        self.lr_decay_rate = lr_decay_rate
        self.t = t
        self.d_low = d_low
        self.d_high = d_high

    def plot_surface(self):
        x1s = []
        x2s = []
        for x1val in range(-100, 100):
            for x2val in range(-100, 100):
                x1s.append(x1val)
                x2s.append(x2val)
        inputs = np.array([np.array([1, x1, x2]) for x1, x2 in zip(x1s, x2s)])
        preds = self.inference(inputs)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(x1s, x2s, preds)
        ax.set_xlabel('x1 Value')
        ax.set_ylabel('x2 Value')
        ax.set_zlabel('Network Output')
        ax.set_title('Network Decision Boundary')
        plt.show()

    def inference(self, x):
        ret = np.zeros(len(x))
        for idx, feat in enumerate(x):
            ret[idx] = np.matmul(np.transpose(self.weights), feat)
        return np.sign(ret)

    def get_error(self, preds, trues):
        error = 0
        for pred, true in zip(preds, trues):
            error += pow((pred - true), 2)
        return error / 2

    def update_batch(self, preds, trues, x):
        deltas = np.zeros(len(self.weights))
        for pred, true, feats in zip(preds, trues, x):
            for idx, weight in enumerate(self.weights):
                x_val = feats[idx]
                deltas[idx] += (true - np.sign(pred)) * x_val
        deltas = self.learning_rate * deltas
        self.weights = self.weights + deltas
        x=1

    def decay_lr(self, prev_error, new_error):
        if self.lr_decay_rate:
            self.learning_rate = self.learning_rate * self.lr_decay_rate
            return False
        if self.t and self.d_low and self.d_high:
            if new_error - prev_error > self.t:
                self.learning_rate = self.learning_rate * self.d_low
                return True
            elif new_error < prev_error:
                self.learning_rate = self.learning_rate * self.d_high
                return False
        return False

    def fit_batch(self, x, labels):
        old_error = 99999999999999999
        for epoch in range(self.num_epochs):
            pred = self.inference(x)
            new_error = self.get_error(pred, labels)
            print(new_error)
            ignore_batch = self.decay_lr(old_error, new_error)
            if not ignore_batch:
                self.update_batch(pred, labels, x)
                old_error = new_error
            acc = sum(np.array(np.sign(pred) == labels)) / len(labels)
            print(acc)
            # if epoch in [4, 9, 49, 99]:
            #     self.plot_surface()

    def fit_stochastic(self, x, labels):
        for epoch in range(self.num_epochs):
            for sample, label in zip(x, labels):
                npsamp = np.array([sample])
                nplab = np.array([label])
                pred = self.inference(npsamp)
                error = self.get_error(pred, nplab)
                self.update_batch(pred, nplab, npsamp)
            # TEST
            pred = self.inference(np.array(x))
            acc = sum(np.array(np.sign(pred) == labels)) / len(labels)
            print(acc)
            self.decay_lr()
            if epoch in [4, 9, 49, 99]:
                self.plot_surface()


def main():
    dataset_list = read_dataset()
    dataset_array = np.array(dataset_list)
    ones = np.expand_dims(np.ones(len(dataset_array)), axis=1)
    dataset_array = np.hstack((ones, dataset_array))
    features = dataset_array[:, 0:3]
    labels = dataset_array[:, 3]

    # Basic
    model = Model(num_epochs=100, initial_lr=.001)
    #
    # # Decay
    # model = Model(num_epochs=100, initial_lr=.001, lr_decay_rate=.95)

    # Adaptive
    # model = Model(num_epochs=100, initial_lr=.001, t=1.03, d_low=.95, d_high=1.05)

    model.fit_batch(features, labels)



if __name__ == '__main__':
    main()
