from Homework3.read_dataset import main as read_dataset
import numpy as np
import math


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


def get_entropy(likelihoods, attribute_idx):
        # Calculate entropy
        probs = np.array(list(likelihoods[attribute_idx].prob_dict.values()))
        log_probs = np.array([math.log2(prob) for prob in probs])
        entropy = -np.dot(probs, log_probs)
        return entropy

def main():
    bin_size_record = {key: [] for key in [5, 10, 15, 20]}
    for random_sample in range(10):
        for num_bins in [5, 10, 15, 20]:
            train, test = read_dataset()
            # Split the Data
            setosa_samples = np.array([sample for sample in train if sample[4] == '1'])
            not_setosa_samples = np.array([sample for sample in train if sample[4] == '0'])

            # train_hist, train_bins = get_prior_stats(train, num_bins)
            # likelihoods = [Likelihoods(feat_hist, feat_bins)
            #                for feat_hist, feat_bins in zip(train_hist, train_bins)]
            # Get Statistics
            set_hist, set_bins = get_prior_stats(setosa_samples, num_bins)
            not_set_hist, not_set_bins = get_prior_stats(not_setosa_samples, num_bins)

            # Get Class Likelihoods
            set_likelihoods = [Likelihoods(feat_hist, feat_bins)
                               for feat_hist, feat_bins in zip(set_hist, set_bins)]
            not_set_likelihoods = [Likelihoods(feat_hist, feat_bins)
                                   for feat_hist, feat_bins in zip(not_set_hist, not_set_bins)]

            # Get Dataset Entropy
            set_prior = len(setosa_samples) / len(train)
            not_set_prior = len(not_setosa_samples) / len(train)
            entropy = 0
            for prob in [set_prior, not_set_prior]:
                entropy -= prob * math.log2(prob)

            train_entropies = {key: None for key in range(4)}
            for feature_idx in range(4):
                entropy = 0
                for likelihood in [set_likelihoods, not_set_likelihoods]:
                    entropy += get_entropy(likelihood, feature_idx)
                train_entropies[feature_idx] = entropy
            x=1

if __name__ == '__main__':
    main()
