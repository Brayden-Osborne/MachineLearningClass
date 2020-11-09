import numpy as np
import math
import copy
import csv
from matplotlib import pyplot as plt


def split_test_train(data):
    test_split = np.random.choice(len(data), size=int(len(data)*.2), replace=False)
    test = data[test_split]
    train_split = np.array([idx for idx in range(len(data)) if idx not in test_split])
    train = data[train_split]

    return test, train


def read_dataset():
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

    return train.astype(float), test.astype(float)


class Likelihoods:
    def __init__(self, hist, bins):
        self.bins = bins
        hist = hist
        disc_vals = [(self.bins[idx] + self.bins[idx + 1]) / 2 for idx in range(len(self.bins) - 1)]
        num_samples = sum(hist)

        # We have to add 1 so that we don't have a zero probability for any feature
        prob_hist = [(hist_val + 1)/num_samples for hist_val in hist]

        self.prob_dict = {disc_val: prob_val for disc_val, prob_val in zip(disc_vals, prob_hist)}

    def discretize(self, x):
        for idx in range(len(self.bins) - 1):
            bottom = self.bins[idx]
            top = self.bins[idx + 1]
            if x < bottom:
                return (bottom + top) / 2
            if bottom <= x < top:
                return (bottom + top) / 2
        if x >= self.bins[-1]:
            return (self.bins[-2] + self.bins[-1]) / 2

    def get_prob(self, x):
        disc_x = self.discretize(x)
        return self.prob_dict[disc_x]


def get_prior_stats(dataset, num_bins):
    hists = []
    bins = []
    for feature_idx in range(4):
        features = dataset[:, feature_idx].astype(float)
        hist, bin_vals = np.histogram(features, bins=num_bins)
        hists.append(hist)
        bins.append(bin_vals)
    return hists, bins


class Node:
    def __init__(self, data, bins, disc_bin_vals, initial_entropy, len_dataset=None, feat=None, feat_val=None, leaf_label=None):
        self.data = data
        self.bins = bins
        self.disc_bin_vals = disc_bin_vals
        self.children = None
        self.feat = feat
        self.feat_val = feat_val
        self.leaf_label = leaf_label
        self.initial_entropy = initial_entropy

        # Need length of un-split dataset
        if len_dataset is not None:
            self.len_dataset = len_dataset
        else:
            self.len_dataset = len(self.data)

        # If no entropy then this node is a leaf
        if self.initial_entropy == 0 and self.leaf_label is None:
            self.leaf_label = self.get_most_common_class()

    def get_most_common_class(self):
        labels = self.data[:, 4]
        return np.argmax(np.bincount(labels.astype(int)))

    def find_split_attribute(self, available_feats):
        # Figure out which feature is going to give the most information gain and return its index
        information_gains = dict()
        for feature_idx in available_feats:
            running_weighted_entropy = 0
            for bin_val in self.bins[feature_idx]:
                # For each of the discreet feature values we want to get the entropy of their labels
                bin_labels = [sample[4] for sample in self.data if sample[feature_idx] == bin_val]
                feat_entropy = calculate_entropy(bin_labels.count(1), bin_labels.count(0))
                # We weight this by the ratio of samples to dataset size so that we prioritize the bigger groups
                running_weighted_entropy += (len(bin_labels) / self.len_dataset) * feat_entropy
            information_gains[feature_idx] = self.initial_entropy - running_weighted_entropy
        return max(information_gains)

    def grow_tree(self, available_feats=None):
        if available_feats is None:
            # We start with being able to use any of the features
            available_feats = [val for val in range(self.data.shape[1] - 1)]
        if len(available_feats) == 0:
            # If there aren't any available features we should just use the most common value in the node's train data
            self.leaf_label = self.get_most_common_class()
            return

        # Decide what we are going to use to make children
        split_feat_idx = self.find_split_attribute(available_feats)
        self.children = []
        # We want to make a node for each discreet value of the feature we are splitting
        for val in self.disc_bin_vals[split_feat_idx]:
            # Get the new dataset for this bin
            bin_indices = np.where(self.data[:, split_feat_idx] == val)
            bin_data = copy.deepcopy(self.data[bin_indices])

            # If there isn't data then the new node should just return the most common class of its parent
            if len(bin_data) == 0:
                new_node = Node(bin_data, self.bins, self.disc_bin_vals, initial_entropy=self.initial_entropy,
                                len_dataset=self.len_dataset, feat=split_feat_idx, feat_val=val,
                                leaf_label=self.get_most_common_class())
            # If there is data then we want a new node using that data
            else:
                new_node = Node(bin_data, self.bins, self.disc_bin_vals, initial_entropy=self.initial_entropy,
                                len_dataset=self.len_dataset, feat=split_feat_idx, feat_val=val)
            self.children.append(new_node)
        for child in self.children:
            # Remove this feature from this path
            new_feats = [val for val in available_feats if val != split_feat_idx]
            if child.leaf_label is not None:
                # If we set the label above we don't need to grow the tree anymore on this path
                continue
            child.grow_tree(new_feats)

    def predict(self, x):
        if self.leaf_label is not None:
            # We found a label
            return self.leaf_label
        else:
            for child in self.children:
                # If x's feature value is equal to the child's feature value we want to use that node
                if discretize(x[child.feat], self.bins[child.feat]) == child.feat_val:
                    return child.predict(x)


def calculate_entropy(num_class_1, num_class_2):
    if num_class_1 == 0 or num_class_2 == 0:
        # IF IT IS ONE CLASS THEN THERE IS NO ENTROPY
        return 0
    prob_1 = num_class_1 / (num_class_1 + num_class_2)
    prob_2 = num_class_2 / (num_class_1 + num_class_2)
    entropy = -1*prob_1*math.log2(prob_1) - prob_2*math.log2(prob_2)
    return entropy


def discretize(x, bins):
    # FIGURE OUT WHICH BIN X IS IN THEN RETURN THE CENTER OF THE BIN
    for idx in range(len(bins) - 1):
        bottom = bins[idx]
        top = bins[idx + 1]
        if x < bottom:
            return (bottom + top) / 2
        if bottom <= x < top:
            return (bottom + top) / 2
    if x >= bins[-1]:
        return (bins[-2] + bins[-1]) / 2


def discretize_dataset(train, num_bins):
    # BIN THE DATA USING HISTOGRAM
    train_bins = []
    for feature_idx in range(4):
        features = train[:, feature_idx]
        hist, bin_vals = np.histogram(features, bins=num_bins)
        train_bins.append(bin_vals)

    # CALCULATE THE DISCRETIZE VALUES BASED ON THE BINS
    bins_list, disc_val_list = [], []
    for feat_idx, feat_bins in enumerate(train_bins):
        disc_vals = [(feat_bins[idx] + feat_bins[idx + 1]) / 2 for idx in range(len(feat_bins) - 1)]
        bins_list.append(feat_bins)
        disc_val_list.append(disc_vals)
        for sample in train:
            sample[feat_idx] = discretize(sample[feat_idx], feat_bins)
    return bins_list, disc_val_list, train


def bayes(num_bins):
    train, test = read_dataset()
    # Split the Data
    setosa_samples = np.array([sample for sample in train if sample[4] == 1])
    not_setosa_samples = np.array([sample for sample in train if sample[4] == 0])

    # Get Statistics
    set_hist, set_bins = get_prior_stats(setosa_samples, num_bins)
    not_set_hist, not_set_bins = get_prior_stats(not_setosa_samples, num_bins)

    # Get Class Likelihoods
    set_likelihoods = [Likelihoods(feat_hist, feat_bins)
                       for feat_hist, feat_bins in zip(set_hist, set_bins)]
    not_set_likelihoods = [Likelihoods(feat_hist, feat_bins)
                           for feat_hist, feat_bins in zip(not_set_hist, not_set_bins)]

    # Get Priors
    set_prior = len(setosa_samples) / len(train)
    not_set_prior = len(not_setosa_samples) / len(train)

    correct = 0
    truth = np.array([])
    pred = np.array([])
    for sample in test:
        set_feat_probs = [set_likelihoods[idx].get_prob(feat)
                          for idx, feat in enumerate(sample[0:3])]
        not_set_feat_probs = [not_set_likelihoods[idx].get_prob(feat)
                              for idx, feat in enumerate(sample[0:3])]

        raw_set_posterior = np.prod(set_feat_probs) / set_prior
        raw_not_set_posterior = np.prod(not_set_feat_probs) / not_set_prior

        set_posterior = raw_set_posterior / (raw_set_posterior + raw_not_set_posterior)
        not_set_posterior = raw_not_set_posterior / (raw_set_posterior + raw_not_set_posterior)

        # GET LABELS
        label = 1 if set_posterior > not_set_posterior else 0
        pred = np.append(pred, label)
        truth = np.append(truth, sample[4])
    return truth, pred


def id3(num_bins):
    # GET AND DISCRETIZE RANDOM TRAIN SPLIT
    train, test = read_dataset()
    bins_list, disc_val_list, train = discretize_dataset(train, num_bins)

    # GET INITIAL ENTROPY
    num_setosa_samples = len(np.array([sample for sample in train if sample[4] == 1]))
    num_not_setosa_samples = len(np.array([sample for sample in train if sample[4] == 0]))
    initial_entropy = calculate_entropy(num_setosa_samples, num_not_setosa_samples)

    # MAKE ROOT AND GROW TREE
    root = Node(data=train, bins=bins_list, disc_bin_vals=disc_val_list, initial_entropy=initial_entropy)
    root.grow_tree()

    # GET RESULTS
    truth = np.array([])
    pred = np.array([])
    for sample in test:
        pred_label = root.predict(sample)
        pred = np.append(pred, pred_label)
        truth = np.append(truth, sample[4])
    return truth, pred


def get_stats(record):
    num_tp, num_tn, num_fp, num_fn = 0, 0, 0, 0


def plot_accuracies(id3_record, bayes_record):
    print("ID3")
    for classifier_name, record in zip(['Bayes', 'ID3'], [bayes_record, id3_record]):
        mins, maxes, means = [], [], []
        print(classifier_name)
        for bin_idx, num_bins in enumerate([5, 10, 15, 20]):
            accs = [sum(np.array(sample[0] == sample[1])) / len(sample[0]) for sample in record[num_bins]]
            mins.append(min(accs))
            maxes.append(max(accs))
            means.append(np.mean(accs))
            print(f"{num_bins} Bins:"
                  f"Min Acc:{mins[-1]:.2f} "
                  f"Mean Acc:{means[-1]:.2f} "
                  f"Max Acc:{maxes[-1]:.2f}")
        plt.plot([5, 10, 15, 20], mins, label=f'{classifier_name} Min Acc')
        plt.plot([5, 10, 15, 20], maxes, label=f'{classifier_name}ID3 Max Acc')
        plt.plot([5, 10, 15, 20], means, label=f'{classifier_name}ID3 Mean Acc')
    plt.legend()
    plt.xlabel('Num Bins')
    plt.ylabel('Accuracy')
    plt.show()


    x=1


def main():
    # 10 RANDOM SAMPLES FOR 4 DIFFERENT BINS
    i_bin_record = {key: [] for key in [5, 10, 15, 20]}
    b_bin_record = {key: [] for key in [5, 10, 15, 20]}
    for random_sample in range(10):
        for num_bins in [5, 10, 15, 20]:
            i_true, i_pred = id3(num_bins)
            b_true, b_pred = bayes(num_bins)
            i_bin_record[num_bins].append((i_true, i_pred))
            b_bin_record[num_bins].append((b_true, b_pred))
    # plot_accuracies(i_bin_record, b_bin_record)
    i_stats = get_stats(b_bin_record[10][0])


if __name__ == '__main__':
    main()
