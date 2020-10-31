import numpy as np
import matplotlib.pyplot as plt
from random import randrange


class SVM:
    def __init__(self, x, y, c=1, tolerance=.001, max_pass=100):
        self.x = x
        self.y = y
        self.length, self.x_dim = np.shape(self.x)
        self.c = c
        self.tolerance = tolerance
        self.epsilon = 1e-3
        self.b = 0
        self.w = np.zeros(self.x_dim)
        self.max_pass = max_pass
        self.alphas = np.zeros(self.length)

    def classify(self, i):
        return float(np.dot(self.w.T, self.x[i])) - self.b

    def get_error(self, i):
        return self.classify(i) - self.y[i]

    def update(self, i2, i1):
        if i2 == i1:
            return False
        a2 = self.alphas[i2]
        a1 = self.alphas[i1]
        y2 = self.y[i2]
        y1 = self.y[i1]
        x2 = self.x[i2]
        x1 = self.x[i1]
        e1 = self.get_error(i2)
        e2 = self.get_error(i1)

        if y2 != y1:
            low_bound = max(0, a1 - a2)
            upper_bound = min(self.c, self.c + a1 - a2)
        else:
            low_bound = max(0, a1 + a2 - self.c)
            upper_bound = min(self.c, a1 + a2)
        if low_bound == upper_bound:
            return False
        k11 = np.dot(x2, x2)
        k12 = np.dot(x2, x1)
        k22 = np.dot(x1, x1)

        eta = k11 + k22 - 2 * k12

        a2_new = a1 + y1 * (e1 - e2) / eta
        a2_new = low_bound if a2_new < low_bound else a2_new
        a2_new = upper_bound if a2_new > upper_bound else a2_new

        if abs(a2_new - a1) < self.epsilon * (a2_new + a1 + self.epsilon):
            return False
        a1_new = a2 + y2 * y1 * (a1 - a2_new)

        # Get New b
        b1 = e1 + y2 * (a1_new - a2) * k11 + y1 * (a2_new - a1) * k12 + self.b
        b2 = e2 + y2 * (a1_new - a2) * k12 + y1 * (a2_new - a1) * k22 + self.b
        if (0 < a1_new) and (self.c > a1_new):
            new_b = b1
        elif (0 < a2_new) and (self.c > a2_new):
            new_b = b2
        else:
            new_b = (b1 + b2) / 2.0

        self.b = new_b

        self.w = self.w + y2 * (a1_new - a2) * x2 + y1 * (a2_new - a1) * x1

        self.alphas[i2] = a1_new
        self.alphas[i1] = a2_new
        return True

    def get_i2(self, non_bound_indices, e1):
        i2 = -1
        if len(non_bound_indices) > 1:
            max_error = 0
            for j in non_bound_indices:
                e2 = self.get_error(j) - self.y[j]
                step = abs(e2 - e1)
                if step > max_error:
                    max_error = step
                    i2 = j
        return i2

    def pick_x2_and_update(self, i1):
        y1 = self.y[i1]
        a1 = self.alphas[i1]
        e1 = self.get_error(i1)
        r1 = e1 * y1
        if not ((r1 < -self.tolerance and a1 < self.c) or (r1 > self.tolerance and a1 > 0)):
            return 0

        non_bound_idx = list(self.get_non_bound_indexes())
        i2 = self.get_i2(non_bound_idx, e1)
        if i2 >= 0 and self.update(i2, i1):
            return 1

        rand_i = randrange(self.length)
        all_indices = list(range(self.length))
        for i2 in all_indices[rand_i:] + all_indices[:rand_i]:
            if self.update(i2, i1):
                return 1
        return 0

    def error(self, i2):
        return self.classify(i2) - self.y[i2]

    def get_non_bound_indexes(self):
        return np.where(np.logical_and(self.alphas > 0, self.alphas < self.c))[0]

    def fit(self):
        num_passes = 0
        while num_passes < self.max_pass:
            num_passes += 1
            for i1 in range(self.length):
                self.pick_x2_and_update(i1)
            print(num_passes)
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
    with open('data2.txt', 'r') as f:
        data = f.read()
    x = []
    y = np.array([])
    for raw_row in data.split('\n'):
        row = raw_row.split(" ")
        pair = np.array([float(row[0]), float(row[1])])
        x.append(pair)
        y = np.append(y, int(float(row[2])))
    x = np.stack(x, axis=0)
    smo = SVM(x, y, tolerance=0.001)
    smo.fit()


if __name__ == '__main__':
    main()
