from Homework3.read_dataset import main as read_dataset
import numpy as np
import math
import copy


class Node:
    def __init__(self, data, likelihoods, len_dataset=None, feat=None, feat_val=None, label=None):
        self.data = data
        self.likelihoods = likelihoods
        self.children = None

        if len_dataset is not None:
            self.len_dataset = len_dataset
        else:
            self.len_dataset = len(self.data)

        num_setosa_samples = len(np.array([sample for sample in data if sample[4] == 1]))
        num_not_setosa_samples = len(np.array([sample for sample in data if sample[4] == 0]))
        self.initial_entropy = calculate_entropy(num_setosa_samples, num_not_setosa_samples)
        self.label = label
        if self.initial_entropy == 0:
            if num_setosa_samples > 0:
                self.label = 1
            elif num_not_setosa_samples > 0:
                self.label = 0

        self.feat = feat
        self.feat_val = feat_val

    def find_split_attribute(self, available_feats):
        information_gains = dict()
        for feature_idx in available_feats:
            feat_vals = np.array([discretize(val, self.likelihoods[feature_idx].bins) for val in self.data[:, feature_idx]])
            feat_labels = self.data[:, 4]
            running_weighted_entropy = 0
            for bin_val in self.likelihoods[feature_idx].bins:
                subset = [label for val, label in zip(feat_vals, feat_labels) if val == bin_val]
                feat_entropy = calculate_entropy(subset.count(1), subset.count(0))
                weight = len(subset) / self.len_dataset
                running_weighted_entropy += weight * feat_entropy
            feat_inf_gain = self.initial_entropy - running_weighted_entropy
            information_gains[feature_idx] = feat_inf_gain
        return max(information_gains)

    def get_most_common_class(self):
        labels = self.data[:, 4]
        return np.argmax(np.bincount(labels.astype(int)))

    def grow_tree(self, available_feats=None):
        if available_feats is None:
            available_feats = [val for val in range(self.data.shape[1] - 1)]

        if self.label is not None:
            return
        elif len(available_feats) == 0:
            self.label = 1
            return
        feat_idx = self.find_split_attribute(available_feats)
        new_nodes = []
        bins = list(self.likelihoods[feat_idx].count_dict.keys())
        feat_vals = np.array([discretize(val, self.likelihoods[feat_idx].bins) for val in self.data[:, feat_idx]])
        for val in bins:
            bin_indices = np.where(feat_vals == val)
            bin_data = copy.deepcopy(self.data[bin_indices])
            if len(bin_data) == 0:
                new_node = Node(bin_data, self.likelihoods, len_dataset=self.len_dataset, feat=feat_idx, feat_val=val,
                                label=self.get_most_common_class())
            else:
                new_node = Node(bin_data, self.likelihoods, len_dataset=self.len_dataset, feat=feat_idx, feat_val=val)
            new_nodes.append(new_node)
        self.children = new_nodes
        for idx in range(len(self.children)):
            if self.children[idx].label is not None:
                continue
            self.children[idx].grow_tree([val for val in available_feats if val != feat_idx])

    def predict(self, x):
        if self.label is not None:
            return self.label
        else:
            for child in self.children:
                feat_idx = child.feat
                sample_feat_val = discretize(x[feat_idx], self.likelihoods[feat_idx].bins)
                if sample_feat_val == child.feat_val:
                    return child.predict(x)
        x = 1


def calculate_entropy(num_class_1, num_class_2):
    if num_class_1 == 0 or num_class_2 == 0:
        return 0
    set_prob = num_class_1 / (num_class_1 + num_class_2)
    not_set_prob = num_class_2 / (num_class_1 + num_class_2)
    entropy = 0
    for prob in [set_prob, not_set_prob]:
        entropy -= prob * math.log2(prob)
    return entropy


class Likelihoods:
    def __init__(self, hist, bins):
        self.bins = bins
        hist = hist
        disc_vals = [(self.bins[idx] + self.bins[idx + 1]) / 2 for idx in range(len(self.bins) - 1)]
        self.count_dict = {disc_val: hist_val for disc_val, hist_val in zip(disc_vals, hist)}


def discretize(x, bins):
    for idx in range(len(bins) - 1):
        bottom = bins[idx]
        top = bins[idx + 1]
        if x < bottom:
            return (bottom + top) / 2
        if bottom <= x < top:
            return (bottom + top) / 2
    if x >= bins[-1]:
        return (bins[-2] + bins[-1]) / 2


def get_prior_stats(dataset, num_bins):
    hists = []
    bins = []
    for feature_idx in range(4):
        features = dataset[:, feature_idx].astype(float)
        hist, bin_vals = np.histogram(features, bins=num_bins)
        hists.append(hist)
        bins.append(bin_vals)
    return hists, bins


def get_entropy(likelihoods, attribute_idx):
    # Calculate entropy
    probs = np.array(list(likelihoods[attribute_idx].prob_dict.values()))
    log_probs = np.array([math.log2(prob) for prob in probs])
    entropy = -np.dot(probs, log_probs)
    return entropy


# def discretize_dataset(train, test, num_bins):
#     train_hist, train_bins = get_prior_stats(train, num_bins)
#     likelihoods = []
#     for feat_hist, feat_bins in zip(train_hist, train_bins):
#         likelihoods.append(Likelihoods(feat_hist, feat_bins))
#     x=1


def main():
    bin_size_record = {key: [] for key in [5, 10, 15, 20]}
    for random_sample in range(10):
        for num_bins in [5, 10, 15, 20]:
            train, test = read_dataset()
            train = train.astype(float)

            train_hist, train_bins = get_prior_stats(train, num_bins)
            likelihoods = [Likelihoods(feat_hist, feat_bins)
                           for feat_hist, feat_bins in zip(train_hist, train_bins)]

            root = Node(train, likelihoods)
            root.grow_tree()

            test = test.astype(float)
            correct_count = 0
            for sample in test:
                pred_label = root.predict(sample)
                if pred_label == sample[4]:
                    correct_count += 1
            accuracy = correct_count / len(test)
            bin_size_record[num_bins].append(accuracy)

    for num_bins in [5, 10, 15, 20]:
        min_acc = min(bin_size_record[num_bins])
        max_acc = max(bin_size_record[num_bins])
        mean_acc = np.mean(bin_size_record[num_bins])
        print(f"{num_bins} Bins: \nMin Acc:{min_acc:.2f} Mean Acc:{mean_acc:.2f} Max Acc:{max_acc:.2f}\n")


if __name__ == '__main__':
    main()
