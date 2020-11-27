from createDataset import read_dataset
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

from mpl_toolkits import mplot3d


class Model:
    def __init__(self, num_epochs=10, initial_lr=.00001, lr_decay_rate=None, t=None, d_low=None, d_high=None):
        self.weights = np.array([.5, .5, .5])
        self.oldWeights = self.weights
        self.num_epochs = num_epochs
        self.learning_rate = initial_lr
        self.lr_decay_rate = lr_decay_rate
        self.t = t
        self.d_low = d_low
        self.d_high = d_high
        self.error = np.zeros(num_epochs)
        self.count = 0

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

    def plot_decision_surface(self):
        x1s = []
        x2s = []
        for x1val in range(-100, 100):
            for x2val in range(-100, 100):
                x1s.append(x1val)
                x2s.append(x2val)
        inputs = np.array([np.array([1, x1, x2]) for x1, x2 in zip(x1s, x2s)])

        outputs = np.array(list(map(lambda x: self.predict(x), inputs)))

        fig = go.Figure(data=[go.Surface(z=outputs.reshape((200,200)))])

        fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))

        fig.show()

        if save:
            fig.write_image(name + ".png")

    def plot_error(self, save, name=""):
        plt.plot(self.error)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        if save:
            plt.savefig( name + '.png')
        plt.show()

    def predict(self, x):
        return np.sign(np.matmul(np.transpose(self.weights), x))

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
            self.error[epoch] = new_error
            print(new_error)
            ignore_batch = self.decay_lr(old_error, new_error)
            if not ignore_batch:
                self.update_batch(pred, labels, x)
                old_error = new_error
            acc = sum(np.array(np.sign(pred) == labels)) / len(labels)
            print(acc)
            # if epoch in [4, 9, 49, 99]:
            #     self.plot_surface()

    def fit_batch_adaptive(self, x, labels):
        old_error = 99999999999999999
        for epoch in range(self.num_epochs):
            pred = self.inference(x)
            new_error = self.get_error(pred, labels)
            self.error[epoch] = new_error
            #print(new_error)
            ignore_batch = self.decay_lr(old_error, new_error)
            if not ignore_batch:
                print(self.learning_rate)
                self.oldWeights = self.weights
                self.update_batch(pred, labels, x)
                old_error = new_error
            else:
                print(self.learning_rate)
                self.weights = self.oldWeights
                self.update_batch(pred, labels, x)
            acc = sum(np.array(np.sign(pred) == labels)) / len(labels)
            # if acc > .995:
            #     self.count += 1
            #     print(self.count)
            #     break
            # else:
            #     self.count += 1
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
                print(error)
                self.update_batch(pred, nplab, npsamp)
            self.error[epoch] = error
            # TEST
            pred = self.inference(np.array(x))
            acc = sum(np.array(np.sign(pred) == labels)) / len(labels)
            # if acc > .995:
            #
            #     self.count += len(x)
            #     print(self.count)
            #     break
            # else:
            #     self.count += len(x)
            #print(acc)
            #self.decay_lr()
            #if epoch in [4, 9, 49, 99]:
            #    self.plot_surface()


def main():
    tic = time.perf_counter()
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
    #model = Model(num_epochs=100, initial_lr=.001, lr_decay_rate=.95)

    # Adaptive
    model = Model(num_epochs=200, initial_lr=.5, t=1.03, d_low=.9, d_high=1.2)

    model.fit_batch_adaptive(features, labels)
    #model.fit_stochastic(features, labels)
    # Plot Decision Surface
    #model.plot_decision_surface()
    toc = time.perf_counter()
    print(f"ran in {toc - tic:0.4f} seconds")
    # Plot and save Error
    model.plot_error(False)



if __name__ == '__main__':
    main()
