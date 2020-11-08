from Homework3.read_dataset import main as read_dataset
import numpy as np
import math
import copy


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


def main():
    # 10 RANDOM SAMPLES FOR 4 DIFFERENT BINS
    bin_size_record = {key: [] for key in [5, 10, 15, 20]}
    for random_sample in range(10):
        for num_bins in [5, 10, 15, 20]:
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

            # CALCULATE ACCURACY ON TEST SET
            correct_count = 0
            for sample in test:
                pred_label = root.predict(sample)
                if pred_label == sample[4]:
                    correct_count += 1
            accuracy = correct_count / len(test)
            bin_size_record[num_bins].append(accuracy)

    # PRINT TESTING METRICS
    for num_bins in [5, 10, 15, 20]:
        min_acc = min(bin_size_record[num_bins])
        max_acc = max(bin_size_record[num_bins])
        mean_acc = np.mean(bin_size_record[num_bins])
        print(f"{num_bins} Bins: \nMin Acc:{min_acc:.2f} Mean Acc:{mean_acc:.2f} Max Acc:{max_acc:.2f}\n")


if __name__ == '__main__':
    main()
