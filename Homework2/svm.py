import numpy as np
import matplotlib.pyplot as plt
import random

def read_data(data_path):
    with open(data_path, 'r') as f:
        data = f.read()
    x = []
    y = np.array([])
    for raw_row in data.split('\n'):
        row = raw_row.split(" ")
        pair = np.array([float(row[0]), float(row[1])])
        x.append(pair)
        y = np.append(y, int(float(row[2])))
    x = np.stack(x, axis=0)
    return (x, y)


class SvmClassifier:
    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]
        self.length = len(self.x)
        self.b = 0
        self.epsilon = .001
        self.c = 1
        self.alphas = np.array([0 for _ in range(self.length)])
        self.w = np.zeros(self.x[0].shape)
    def update_w(self):
        w = np.zeros(self.x[0].shape)
        for i in range(self.length):
            w += self.alphas[i] * self.y[i] * self.x[i]
        self.w = w

    def get_kkt(self):
        w = self.update_w()
        kkt = np.array([])
        for i in range(self.length):
            kkt = np.append(kkt, self.alphas[i] * (self.y[i] * (np.matmul(w.T, self.x[i]) + self.b) - 1))
        return kkt

    def get_e(self, i1):
        e = np.array([])
        for i in range(self.length):
            ei = 0
            for j in range(self.length):
                kj1 = np.dot(self.x[j], self.x[i1])
                kji = np.dot(self.x[j], self.x[i])
                eij = self.alphas[j] * self.y[j] * (kj1 - kji) + self.y[i] - self.y[i1]
                ei += eij
            e = np.append(e, ei)
        return e

    def get_x1(self):
        kkt = self.get_kkt()
        i1 = np.argmax(kkt)
        x1 = self.x[i1]
        return i1, x1

    def get_x2(self, i1):
        e = self.get_e(i1)
        i2 = np.argmax(e)
        x2 = self.x[i2]
        return i2, x2

    @staticmethod
    def get_k(x1, x2):
        k11 = np.dot(x1, x1)
        k22 = np.dot(x2, x2)
        k12 = np.dot(x1, x2)
        return k11 + k22 - 2*k12

    def clip_a2(self, y1, y2, a1, a2, new_a2):
        if y1 == y2:
            low_bound = max(0, a2 + a1 - self.c)
            upper_bound = min(self.c, a2 + a1)
        else:
            low_bound = max(0, a2 - a1)
            upper_bound = min(self.c, self.c + a2 - a1)
        if new_a2 < low_bound:
            return low_bound
        elif new_a2 > upper_bound:
            return upper_bound
        else:
            return new_a2

    def update_a1_a2(self, i1, x1, i2, x2):
        e = self.get_e(i1)
        a1 = self.alphas[i1]
        a2 = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        k = self.get_k(x1, x2)

        # Update a2
        # new_a2 = a2 + (y2 * (e[i1] - e[i2])) / k
        new_a2 = a2 + (y2 * (e[i2])) / k
        new_a2 = self.clip_a2(y1, y2, a1, a2, new_a2)
        self.alphas[i2] = new_a2

        # Update a1
        new_a1 = a1 + y1*y2*(a2 - new_a2)
        if new_a1 > self.c:
            new_a1 = self.c
        self.alphas[i1] = new_a1
        return e[i1], e[i2], new_a1, new_a2, a1, a2

    # def squash_alpha(self):
    #     for i in range(self.length):
    #         if self.alphas[i] < self.epsilon:
    #             self.alphas[i] = 0

    def calc_b(self, e1, e2, a1_new, a1, a2_new, a2, i1, i2):
        y1, y2 = self.y[i1], self.y[i2]
        x1, x2 = self.x[i1], self.x[i2]
        b1 = e1 + y1*(a1_new - a1)*np.dot(x1, x1) + y2*(a2_new - a2)*np.dot(x1, x2) + self.b
        b2 = e2 + y1*(a1_new - a1)*np.dot(x1, x2) + y2*(a2_new - a2)*np.dot(x2, x2) + self.b
        self.b = (b1 + b2) / 2



    def calculate_b(self):
        for i in range(self.length):
            if self.alphas[i] > 0:
                self.b = self.y[i] - np.matmul(self.update_w().T, self.x[i])
                return

    @staticmethod
    def kernel(x1, x2):
        return np.dot(x1, x2)

    def plot(self):
        class_1_x = [x[0] for x, y in zip(self.x, self.y) if y == 1]
        class_neg1_x = [x[0] for x, y in zip(self.x, self.y) if y == -1]
        class_1_y = [x[1] for x, y in zip(self.x, self.y) if y == 1]
        class_neg1_y = [x[1] for x, y in zip(self.x, self.y) if y == -1]
        plt.plot(class_1_x, class_1_y, 'ro')
        plt.plot(class_neg1_x, class_neg1_y, 'bo')

        w = self.w
        a = -w[0] / w[1]
        xx = np.linspace(0, 315)
        yy = a * xx - self.b / w[1]
        plt.plot(xx, yy, '--')
        plt.show()

    def classify(self, i):
        return float(np.dot(self.w.T, self.x[i])) - self.b

    def get_error(self, i):
        return self.classify(i) - self.y[i]

    def check_kkt_fulfilled(self, e, y, a):
        return (e*y < -self.epsilon and a < self.c) or (e*y > self.epsilon and a > 0)

    def get_i2(self, i1):
        i2 = -1
        e1 = self.get_error(i1)
        max = 0
        for j in range(self.length):
            e2 = self.get_error(j) - self.y[j]
            step = abs(e2 - e1)  # approximation
            if step > max and j != i1:
                max = step
                i2 = j
        return i2

    def take_step(self, i1, i2):
        a1 = self.alphas[i1]
        x1 = self.x[i1]
        y1 = self.y[i1]
        e1 = self.get_error(i1)
        a2 = self.alphas[i2]
        x2 = self.x[i2]
        y2 = self.y[i2]
        e2 = self.get_error(i2)
        s = y1 * y2
        if y1 != y2:
            # Equation 13
            L = max(0, a2 - a1)
            H = min(self.c, self.c + a2 - a1)
        else:
            # Equation 14
            L = max(0, a2 + a1 - self.c)
            H = min(self.c, a2 + a1)
        if L == H:
            return False
        k11 = self.kernel(x1, x1)
        k12 = self.kernel(x1, x2)
        k22 = self.kernel(x2, x2)

        eta = k11 + k22 - 2 * k12
        if not eta > 0:
            return False

        a2_new = a2 + y2 * (e1 - e2) / eta
        if a2_new < L:
            a2_new = L
        elif a2_new > H:
            a2_new = H
        if abs(a2_new - a2) < self.epsilon * (a2_new + a2 + self.epsilon):
            return False
        a1_new = a1 + s * (a2 - a2_new)
        new_b = self.compute_b(e1, a1, a1_new, a2_new, k11, k12, k22, y1, y2, a2, e2)
        self.b = new_b
        self.w = self.w + y1 * (a1_new - a1) * x1 + y2 * (a2_new - a2) * x2
        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new
        return True

    def compute_b(self, E1, a1, a1_new, a2_new, k11, k12, k22, y1, y2, a2, e2):

        # Equation 20
        b1 = E1 + y1 * (a1_new - a1) * k11 + y2 * (a2_new - a2) * k12 + self.b
        # Equation 21
        b2 = e2 + y1 * (a1_new - a1) * k12 + y2 * (a2_new - a2) * k22 + self.b
        if (0 < a1_new) and (self.c > a1_new):
            new_b = b1
        elif (0 < a2_new) and (self.c > a2_new):
            new_b = b2
        else:
            new_b = (b1 + b2) / 2.0
        return new_b

    def get_non_bound_indexes(self):
        return np.where(np.logical_and(self.alphas > 0, self.alphas < self.c))[0]

    def first_heuristic(self):
        num_changed = 0
        non_bound_idx = self.get_non_bound_indexes()

        for i1 in non_bound_idx:
            i2 = self.get_i2(i1)
            num_changed += self.take_step(i1, i2)
        return num_changed

    def fit(self):
        num_changed = 0
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0

            if examine_all:
                for i1 in range(self.length):
                    i2 = self.get_i2(i1)
                    num_changed += self.take_step(i1, i2)
            else:
                num_changed += self.first_heuristic()

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

def main():

    data = read_data('data2.txt')
    svm = SvmClassifier(data)
    svm.fit()
    svm.plot()

if __name__ == '__main__':
    main()
