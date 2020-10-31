import numpy as np
import matplotlib.pyplot as plt
from random import randrange


class SVM:
    def __init__(self, x, y, c, tolerance, max_pass=1000):
        self.x = x
        self.y = y
        self.length, self.x_dim = np.shape(self.x)
        self.c = c
        self.tolerance = tolerance
        self.epsilon = 1e-3
        self.b = 0
        self.w = np.zeros(self.x_dim)
        self.max_pass = max_pass

        # l = len(x)
        # self.alphas = np.random.rand(self.length)
        # while abs(np.dot(self.alphas, y)) > 0.00000001:
        #     self.alphas += np.dot(self.alphas, y) / self.length

        self.alphas = np.zeros(self.length)

    @staticmethod
    def kernel(x1, x2):
        return np.dot(x1, x2)

    def classify(self, i):
        return float(np.dot(self.w.T, self.x[i])) - self.b

    def get_error(self, i1):
        return self.classify(i1) - self.y[i1]

    def update(self, i1, i2):
        if i1 == i2:
            return False
        a1 = self.alphas[i1]
        a2 = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        x1 = self.x[i1]
        x2 = self.x[i2]
        e1 = self.get_error(i1)
        e2 = self.get_error(i2)

        if y1 != y2:
            low_bound = max(0, a2 - a1)
            upper_bound = min(self.c, self.c + a2 - a1)
        else:
            low_bound = max(0, a2 + a1 - self.c)
            upper_bound = min(self.c, a2 + a1)
        if low_bound == upper_bound:
            return False
        k11 = self.kernel(x1, x1)
        k12 = self.kernel(x1, x2)
        k22 = self.kernel(x2, x2)

        eta = k11 + k22 - 2 * k12

        a2_new = a2 + y2 * (e1 - e2) / eta
        a2_new = low_bound if a2_new < low_bound else a2_new
        a2_new = upper_bound if a2_new > upper_bound else a2_new

        if abs(a2_new - a2) < self.epsilon * (a2_new + a2 + self.epsilon):
            return False
        a1_new = a1 + y1 * y2 * (a2 - a2_new)

        # Get New b
        b1 = e1 + y1 * (a1_new - a1) * k11 + y2 * (a2_new - a2) * k12 + self.b
        b2 = e2 + y1 * (a1_new - a1) * k12 + y2 * (a2_new - a2) * k22 + self.b
        if (0 < a1_new) and (self.c > a1_new):
            new_b = b1
        elif (0 < a2_new) and (self.c > a2_new):
            new_b = b2
        else:
            new_b = (b1 + b2) / 2.0

        delta_b = new_b - self.b
        self.b = new_b

        self.w = self.w + y1 * (a1_new - a1) * x1 + y2 * (a2_new - a2) * x2

        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new
        return True

    def second_heuristic(self, non_bound_indices, e2):
        i1 = -1
        if len(non_bound_indices) > 1:
            max_error = 0
            for j in non_bound_indices:
                e1 = self.get_error(j) - self.y[j]
                step = abs(e1 - e2)  # approximation
                if step > max_error:
                    max_error = step
                    i1 = j
        return i1

    def examine_example(self, i2):
        y2 = self.y[i2]
        a2 = self.alphas[i2]
        e2 = self.get_error(i2)
        r2 = e2 * y2
        if not ((r2 < -self.tolerance and a2 < self.c) or (r2 > self.tolerance and a2 > 0)):
            # The KKT conditions are met, SMO looks at another example.
            return 0

        # Second heuristic A: choose the Lagrange multiplier which
        # maximizes the absolute error.
        non_bound_idx = list(self.get_non_bound_indexes())
        i1 = self.second_heuristic(non_bound_idx, e2)
        if i1 >= 0 and self.update(i1, i2):
            return 1

        # Second heuristic B: Look for examples making positive
        # progress by looping over all non-zero and non-C alpha,
        # starting at a random point.
        if len(non_bound_idx) > 0:
            rand_i = randrange(len(non_bound_idx))
            for i1 in non_bound_idx[rand_i:] + non_bound_idx[:rand_i]:
                if self.update(i1, i2):
                    return 1
        # Second heuristic C: Look for examples making positive progress
        # by looping over all possible examples, starting at a random
        # point.
        rand_i = randrange(self.length)
        all_indices = list(range(self.length))
        for i1 in all_indices[rand_i:] + all_indices[:rand_i]:
            if self.update(i1, i2):
                return 1
        # Extremely degenerate circumstances, SMO skips the first example.
        return 0

    def error(self, i2):
        return self.classify(i2) - self.y[i2]

    def get_non_bound_indexes(self):
        return np.where(np.logical_and(self.alphas > 0, self.alphas < self.c))[0]

    # First heuristic: loop over examples where alpha is not 0 and not C
    # they are the most likely to violate the KKT conditions
    # (the non-bound subset).
    def first_heuristic(self):
        num_changed = 0
        non_bound_idx = self.get_non_bound_indexes()

        for i in non_bound_idx:
            num_changed += self.examine_example(i)
        return num_changed

    # def get_kkt(self):
    #     kkt = np.array([])
    #     for i in range(self.length):
    #         kkt = np.append(kkt, self.alphas[i] * (self.y[i] * (np.dot(self.w.T, self.x[i]) + self.b) - 1))
    #     return kkt

    def main_routine(self):
        num_passes = 0
        examine_all = True

        while num_passes < self.max_pass:
            num_passes += 1
            for i in range(self.length):
                self.examine_example(i)
            print(num_passes)

                # examine_all = False
        self.plot()

    def plot(self):
        w = self.w
        b = self.b
        x = self.x
        y = self.y
        class_1_x = [x[0] for x, y in zip(x, y) if y == 1]
        class_neg1_x = [x[0] for x, y in zip(x, y) if y == -1]
        class_1_y = [x[1] for x, y in zip(x, y) if y == 1]
        class_neg1_y = [x[1] for x, y in zip(x, y) if y == -1]
        plt.plot(class_1_x, class_1_y, 'ro')
        plt.plot(class_neg1_x, class_neg1_y, 'bo')

        a = -w[0] / w[1]
        xx = np.linspace(0, 315)
        yy = a * xx - b / w[1]
        plt.plot(xx, yy, '--')
        plt.show()


def main():
    with open('data.txt', 'r') as f:
        data = f.read()
    x = []
    y = np.array([])
    for raw_row in data.split('\n'):
        row = raw_row.split(" ")
        pair = np.array([float(row[0]), float(row[1])])
        x.append(pair)
        y = np.append(y, int(float(row[2])))
    x = np.stack(x, axis=0)
    smo = SVM(x, y, c=10, tolerance=0.001)
    smo.main_routine()


if __name__ == '__main__':
    main()
