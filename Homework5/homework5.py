# Homework 5
# Brayden Osborne, Tyler Hilbert, Matthew Terry

import csv
import numpy as np
import matplotlib.pyplot as plt


def read_dataset():
    data = []
    labels = []
    with open('data_banknote_authentication.csv', 'r') as f:
        tmp = csv.reader(f)
        for idx, row in enumerate(tmp):
            if len(row) == 0:
                continue
            else:
                data.append([float(val) for val in row[0:4]])
                labels.append(int(row[4]))
    return data, labels


def split_dataset(dataset, labels):
    """
    Split our dataset into 80% train, 10% val, 10% test
    """
    train_split = np.random.choice(len(dataset), size=int(len(dataset) * .8), replace=False)
    train_ds = dataset[train_split]
    train_labels = labels[train_split]
    dataset = np.array([item for idx, item in enumerate(dataset) if idx not in train_split])
    labels = np.array([item for idx, item in enumerate(labels) if idx not in train_split])
    val_split = np.random.choice(len(dataset), size=int(len(dataset) * .5), replace=False)
    val_ds = dataset[val_split]
    val_labels = labels[val_split]
    test_split = np.array([idx for idx in range(len(dataset)) if idx not in val_split])
    test_ds = dataset[test_split]
    test_labels = labels[test_split]

    return train_ds, train_labels, val_ds, val_labels, test_ds, test_labels


class NeuralNetwork:
    def __init__(self, num_feats, hidden_layer_size, hidden_activation, final_activation, learning_rate):
        self.hidden_activation = get_activation_function(hidden_activation)
        self.hidden_activation_derivative = get_derivative_activation_function(hidden_activation)
        self.final_activation = get_activation_function(final_activation)
        self.final_activation_derivative = get_derivative_activation_function(final_activation)
        self.hidden_weights = np.random.randn(num_feats, hidden_layer_size) * .01
        self.hidden_bias = np.random.randn(1, hidden_layer_size) * .01
        self.final_weights = np.random.randn(1, hidden_layer_size) * .01
        self.final_bias = np.random.randn(1, 1) * .01
        self.learning_rate = learning_rate

    def train(self, data, labels, num_epochs=100):
        for epoch in range(num_epochs):
            h_o_pre, hidden_out, f_o_pre, final_out = self.forward(data)
            loss = self.calc_loss(y_true=labels, y_pred=final_out)
            self.backprop(hidden_out, final_out, data, labels, h_o_pre, f_o_pre)
            # print(loss)

    def backprop(self, hidden_out, final_out, in_data, labels, h_o_pre, f_o_pre):

        # FINAL
        d_error_d_out_f = final_out - labels
        d_out_d_net_f = self.final_activation_derivative(f_o_pre)
        d_net_d_weight_f = hidden_out
        d_error_d_weight_f = d_error_d_out_f * d_out_d_net_f * d_net_d_weight_f
        final_weight_update = np.sum(d_error_d_weight_f, axis=1) / len(d_error_d_weight_f)
        final_bias_change = (1 / len(labels)) * np.sum(d_error_d_out_f, axis=1, keepdims=True)

        # HIDDEN
        d_error_d_out_h = np.dot(self.final_weights.T, d_error_d_out_f)
        d_out_d_net_h = self.hidden_activation_derivative(h_o_pre)
        d_net_d_weight_h = np.copy(in_data)
        d_error_d_weight_h = np.dot(d_error_d_out_f * d_out_d_net_h, d_net_d_weight_h)
        hidden_weight_update = d_error_d_weight_h
        hidden_bias_change = np.sum(d_error_d_out_h, axis=1, keepdims=True)

        # UPDATE WEIGHTS AND BIASES
        self.final_weights = self.final_weights - self.learning_rate * final_weight_update
        self.hidden_weights = self.hidden_weights - self.learning_rate * hidden_weight_update.T
        self.final_bias = self.final_bias - self.learning_rate * final_bias_change.T
        self.hidden_bias = self.hidden_bias - self.learning_rate * hidden_bias_change.T

    def calc_loss(self, y_true, y_pred):
        """
        Calculate the MSE loss
        """
        error = 1/2 * pow(y_true - y_pred, 2)
        avg_loss = np.squeeze(np.sum(error) / len(y_true))
        return avg_loss

    def forward(self, input_data):
        x = np.copy(input_data)

        hidden_output = np.dot(self.hidden_weights.T, x.T) + self.hidden_bias.T
        activated_h_o = self.hidden_activation(np.copy(hidden_output))

        final_output = np.dot(self.final_weights, activated_h_o) + self.final_bias.T
        activated_f_o = self.final_activation(np.copy(final_output))
        return hidden_output, activated_h_o, final_output, activated_f_o

    def inference(self, data):
        _, _, _, y_hat = self.forward(data)
        plus_min = np.sign(y_hat - .5)
        labels = [0 if val == -1 else 1 for val in plus_min[0]]
        return np.array(labels)


def get_activation_function(name):
    if name == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-1 * x))

    elif name == 'tanh':
        return lambda x: (np.exp(x) - np.exp(-1 * x)) / (np.exp(x) + np.exp(-1 * x))

    else:
        raise NotImplementedError("UNKNOWN ACTIVATION FUNCTION")


def get_derivative_activation_function(name):
    if name == 'sigmoid':
        sig = lambda x: 1 / (1 + np.exp(-1 * x))
        return lambda x: sig(x) * (1 - sig(x))
    elif name == 'tanh':
        tanh = lambda x: (np.exp(x) - np.exp(-1 * x)) / (np.exp(x) + np.exp(-1 * x))
        return lambda x: 1 - tanh(x) ** 2
    else:
        raise NotImplementedError("UNKNOWN ACTIVATION FUNCTION")


def main():
    dataset, labels = read_dataset()
    dataset = np.vstack(dataset)
    labels = np.array(labels)
    train_ds, train_labels, val_ds, val_labels, test_ds, test_labels = split_dataset(dataset, labels)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    axes = [ax1, ax2]
    for ax in axes:
        ax.set(xlabel='Num Hidden Layers', ylabel='Accuracy')
        ax.set_ylim(0, 1)
        ax.set_xlim(1, 4)
    fig.suptitle('Classifier Accuracy as a function of Num Hidden Neurons')

    for idx, (hidden, final) in enumerate([['sigmoid', 'sigmoid'], ['tanh', 'tanh']]):
        accs = []
        best_model = None
        best_acc = 0
        best_str = None
        for num_hidden in [1, 2, 3, 4]:
            nn = NeuralNetwork(num_feats=4, hidden_layer_size=num_hidden, hidden_activation=hidden,
                               final_activation=final, learning_rate=.0001)
            nn.train(train_ds, train_labels, num_epochs=20000)
            val_pred_labels = nn.inference(val_ds)
            val_acc = sum(val_labels == val_pred_labels) / len(val_labels)
            accs.append(val_acc)
            if val_acc >= best_acc:
                best_model = nn
                best_acc = val_acc
                best_str = f'Hidden Activation: {hidden}  Final Activation: {final}  {num_hidden} hidden nodes'

        test_pred_labels = best_model.inference(test_ds)
        test_acc = sum(test_labels == test_pred_labels) / len(test_labels)

        print(best_str + f'\nTest Accuracy: {test_acc}')
        axes[idx].plot([1, 2, 3, 4], accs)
        axes[idx].set_title(f'Hidden: {hidden} Final: {final}')
    plt.show()
    fig.show()


if __name__ == "__main__":
    main()
