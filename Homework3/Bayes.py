from Homework3.read_dataset import main as read_dataset
import numpy as np


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


def main():
    bin_size_record = {key: [] for key in [5, 10, 15, 20]}
    for random_sample in range(10):
        for num_bins in [5, 10, 15, 20]:
            train, test = read_dataset()
            # Split the Data
            setosa_samples = np.array([sample for sample in train if sample[4] == '1'])
            not_setosa_samples = np.array([sample for sample in train if sample[4] == '0'])

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
            for sample in test:
                set_feat_probs = [set_likelihoods[idx].get_prob(float(feat))
                                  for idx, feat in enumerate(sample[0:3])]
                not_set_feat_probs = [not_set_likelihoods[idx].get_prob(float(feat))
                                      for idx, feat in enumerate(sample[0:3])]

                raw_set_posterior = np.prod(set_feat_probs) / set_prior
                raw_not_set_posterior = np.prod(not_set_feat_probs) / not_set_prior

                set_posterior = raw_set_posterior / (raw_set_posterior + raw_not_set_posterior)
                not_set_posterior = raw_not_set_posterior / (raw_set_posterior + raw_not_set_posterior)

                label = 1 if set_posterior > not_set_posterior else 0
                if label == int(sample[4]):
                    correct += 1

            accuracy = correct / len(test)
            bin_size_record[num_bins].append(accuracy)

    for num_bins in [5, 10, 15, 20]:
        min_acc = min(bin_size_record[num_bins])
        max_acc = max(bin_size_record[num_bins])
        mean_acc = np.mean(bin_size_record[num_bins])
        print(f"{num_bins} Bins: \nMin Acc:{min_acc:.2f} Mean Acc:{mean_acc:.2f} Max Acc:{max_acc:.2f}\n")


if __name__ == '__main__':
    main()
