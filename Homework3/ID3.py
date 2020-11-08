from Homework3.read_dataset import main as read_dataset
import numpy as np
import math
import copy


class Node:
    def __init__(self, data, bins, disc_bin_vals, len_dataset=None, feat=None, feat_val=None, label=None):
        self.data = data
        self.bins = bins
        self.disc_bin_vals = disc_bin_vals
        self.children = None
        self.feat = feat
        self.feat_val = feat_val

        # Need length of un-split dataset
        if len_dataset is not None:
            self.len_dataset = len_dataset
        else:
            self.len_dataset = len(self.data)

        # Get initial entropy
        num_setosa_samples = len(np.array([sample for sample in data if sample[4] == 1]))
        num_not_setosa_samples = len(np.array([sample for sample in data if sample[4] == 0]))
        self.initial_entropy = calculate_entropy(num_setosa_samples, num_not_setosa_samples)

        # If no entropy then this node is a leaf
        self.label = label
        if self.initial_entropy == 0 and self.label is None:
            self.label = self.get_most_common_class()

    def get_most_common_class(self):
        labels = self.data[:, 4]
        return np.argmax(np.bincount(labels.astype(int)))

    def find_split_attribute(self, available_feats):
        information_gains = dict()
        for feature_idx in available_feats:
            running_weighted_entropy = 0
            for bin_val in self.bins[feature_idx]:
                bin_labels = [sample[4] for sample in self.data if sample[feature_idx] == bin_val]
                feat_entropy = calculate_entropy(bin_labels.count(1), bin_labels.count(0))
                running_weighted_entropy += (len(bin_labels) / self.len_dataset) * feat_entropy
            information_gains[feature_idx] = self.initial_entropy - running_weighted_entropy
        return max(information_gains)

    def grow_tree(self, available_feats=None):
        if available_feats is None:
            available_feats = [val for val in range(self.data.shape[1] - 1)]
        if self.label is not None:
            return

        split_feat_idx = self.find_split_attribute(available_feats)
        self.children = []
        for val in self.disc_bin_vals[split_feat_idx]:
            bin_indices = np.where(self.data[:, split_feat_idx] == val)
            bin_data = copy.deepcopy(self.data[bin_indices])
            if len(bin_data) == 0:
                new_node = Node(bin_data, self.bins, self.disc_bin_vals, len_dataset=self.len_dataset,
                                feat=split_feat_idx, feat_val=val, label=self.get_most_common_class())
            else:
                new_node = Node(bin_data, self.bins, self.disc_bin_vals, len_dataset=self.len_dataset,
                                feat=split_feat_idx, feat_val=val)
            self.children.append(new_node)
        for idx in range(len(self.children)):
            if self.children[idx].label is not None:
                continue
            self.children[idx].grow_tree([val for val in available_feats if val != split_feat_idx])

    def predict(self, x):
        if self.label is not None:
            return self.label
        else:
            for child in self.children:
                feat_idx = child.feat
                sample_feat_val = discretize(x[feat_idx], self.bins[feat_idx])
                if sample_feat_val == child.feat_val:
                    return child.predict(x)


def calculate_entropy(num_class_1, num_class_2):
    if num_class_1 == 0 or num_class_2 == 0:
        return 0
    set_prob = num_class_1 / (num_class_1 + num_class_2)
    not_set_prob = num_class_2 / (num_class_1 + num_class_2)
    entropy = 0
    for prob in [set_prob, not_set_prob]:
        entropy -= prob * math.log2(prob)
    return entropy


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


def discretize_dataset(train, num_bins):
    train_hist, train_bins = get_prior_stats(train, num_bins)
    bins_list = []
    disc_val_list = []
    for feat_idx, (feat_hist, feat_bins) in enumerate(zip(train_hist, train_bins)):
        disc_vals = [(feat_bins[idx] + feat_bins[idx + 1]) / 2 for idx in range(len(feat_bins) - 1)]
        bins_list.append(feat_bins)
        disc_val_list.append(disc_vals)
        for sample in train:
            sample[feat_idx] = discretize(sample[feat_idx], feat_bins)
    return bins_list, disc_val_list, train


def main():
    bin_size_record = {key: [] for key in [5, 10, 15, 20]}
    for random_sample in range(10):
        for num_bins in [5, 10, 15, 20]:
            train, test = read_dataset()

            bins_list, disc_val_list, train = discretize_dataset(train, num_bins)
            root = Node(train, bins_list, disc_val_list)
            root.grow_tree()

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
