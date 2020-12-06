import csv
import numpy as np
import matplotlib as plt


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
    def __init__(self, num_feats=4, num_classes=2, hidden_layer_size=None,
                 hidden_activation="tanh", final_activation="sigmoid", learning_rate=.001):
        self.hidden_activation = get_activation_function(hidden_activation)
        self.hidden_activation_derivative = get_derivative_activation_function(hidden_activation)
        self.final_activation = get_activation_function(final_activation)
        self.final_activation_derivative = get_derivative_activation_function(final_activation)
        self.hidden_weights = np.random.randn(num_feats, hidden_layer_size) * .01
        self.hidden_bias = np.ones((1, hidden_layer_size))
        self.final_weights = np.random.randn(1, hidden_layer_size) * .01
        self.final_bias = np.ones((1, 1))
        self.learning_rate = learning_rate

    def train(self, data, labels, num_epochs=100):
        for epoch in range(num_epochs):
            hidden_out, final_out = self.forward(data)
            loss = self.calc_loss(y_true=labels, y_pred=final_out)
            self.backprop(hidden_out, final_out, data, labels)
            # print(loss)

    def backprop(self, hidden_out, final_out, in_data, labels):
        delta_final = final_out - labels
        final_weight_change = (1 / len(labels)) * np.dot(delta_final, hidden_out.T)
        final_bias_change = (1 / len(labels)) * np.sum(delta_final, axis=1, keepdims=True)

        delta_hidden = np.multiply(np.dot(self.final_weights.T, delta_final), 1 - np.power(hidden_out, 2))
        hidden_weight_change = (1 / len(labels)) * np.dot(delta_hidden, in_data)
        hidden_bias_change = np.sum(delta_hidden, axis=1, keepdims=True)

        self.final_weights = self.final_weights - self.learning_rate * final_weight_change
        self.hidden_weights = self.hidden_weights - self.learning_rate * hidden_weight_change.T
        self.final_bias = self.final_bias - self.learning_rate * final_bias_change.T
        self.hidden_bias = self.hidden_bias - self.learning_rate * hidden_bias_change.T

    def calc_loss(self, y_true, y_pred):
        loss_array = np.multiply(np.log(y_pred), y_true) + np.multiply((1 - y_true), np.log(1 - y_pred))
        avg_loss = np.squeeze(-np.sum(loss_array) / len(y_true))
        return avg_loss

    def forward(self, input_data):
        x = np.copy(input_data)
        hidden_output = np.dot(self.hidden_weights.T, x.T) + self.hidden_bias.T
        activated_h_o = self.hidden_activation(hidden_output)
        final_output = np.dot(self.final_weights, activated_h_o) + self.final_bias.T
        activated_f_o = self.final_activation(final_output)
        return activated_h_o, activated_f_o

    def inference(self, data):
        _, y_hat = self.forward(data)
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
    for hidden, final in [['sigmoid', 'sigmoid'], ['sigmoid', 'tanh'], ['tanh', 'sigmoid'], ['tanh', 'tanh']]:
        for num_hidden in [1, 2, 3, 4]:
            nn = NeuralNetwork(hidden_layer_size=num_hidden, hidden_activation=hidden, final_activation=final,
                               learning_rate=.001)
            nn.train(train_ds, train_labels, num_epochs=10000)
            val_pred_labels = nn.inference(val_ds)
            val_acc = sum(val_labels == val_pred_labels) / len(val_labels)
            print(f'Hidden Activation: {hidden}  Final Activation: {final}  {num_hidden} hidden nodes\n '
                  f'Validation Accuracy: {val_acc}')
        x = 1


if __name__ == "__main__":
    main()
