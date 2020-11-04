from Homework3.read_dataset import main as read_dataset
import numpy as np
import math


def get_prior_stats(dataset):
    means = np.zeros(dataset.shape[1] - 1)
    stddevs = np.zeros(dataset.shape[1] - 1)
    for feature_idx in range(4):
        features = dataset[:, feature_idx].astype(float)
        means[feature_idx] = np.mean(features)
        stddevs[feature_idx] = np.std(features)
    return means, stddevs


def get_gaussian(x, mean, std):
    return (1/(math.sqrt(2*np.pi)*std)) * math.exp(-(pow(x - mean, 2) / (2 * pow(std,2))))


def main():
    train, test = read_dataset()

    # Split the Data
    setosa_samples = np.array([sample for sample in train if sample[4] == '1'])
    not_setosa_samples = np.array([sample for sample in train if sample[4] == '0'])

    # Get Statistics
    set_mean, set_std = get_prior_stats(setosa_samples)
    not_set_mean, not_set_std = get_prior_stats(not_setosa_samples)

    # Get Priors
    set_prior = len(setosa_samples) / len(train)
    not_set_prior = len(not_setosa_samples) / len(train)

    correct = 0
    for sample in test:
        set_feat_probs = [get_gaussian(float(feat), set_mean[idx], set_std[idx]) for idx, feat in enumerate(sample[0:3])]
        not_set_feat_probs = [get_gaussian(float(feat), not_set_mean[idx], not_set_std[idx]) for idx, feat in enumerate(sample[0:3])]

        raw_set_posterior = np.prod(set_feat_probs) / set_prior
        raw_not_set_posterior = np.prod(not_set_feat_probs) / not_set_prior

        set_posterior = raw_set_posterior / (raw_set_posterior + raw_not_set_posterior)
        not_set_posterior = raw_not_set_posterior / (raw_set_posterior + raw_not_set_posterior)

        label = 1 if set_posterior > not_set_posterior else 0
        if label == int(sample[4]):
            correct += 1

    accuracy = correct / len(test)
    print(accuracy)


if __name__ == '__main__':
    main()