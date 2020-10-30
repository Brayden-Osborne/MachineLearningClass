import numpy as np
import csv
import matplotlib.pyplot as plt


def read_data(csv_path):
    f = open(csv_path, 'rt')

    reader = csv.reader(f)
    x = []
    y = np.array([])
    for row in reader:
        pair = np.array([float(row[1]), float(row[2])])
        x.append(pair)
        y = np.append(y, int(float(row[0])))
    x = np.stack(x, axis=0)
    return (x, y)


class SvmClassifier:
    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]
        self.length = len(self.x)
        self.b = 0
        self.epsilon = .0001
        self.c = 10000
        self.alphas = [.5 for _ in range(self.length)]

    def get_w(self):
        w = np.zeros(self.x[0].shape)
        for i in range(self.length):
            w += self.alphas[i] * self.y[i] * self.x[i]
        return w

    def get_kkt(self):
        w = self.get_w()
        kkt = np.array([])
        for i in range(self.length):
            kkt = np.append(kkt, self.alphas[i] * (self.y[i] * (np.dot(w, self.x[i]) + self.b) - 1))
        return kkt

    def get_e(self, i1):
        e = np.array([])
        for i in range(self.length):
            ei = 0
            for j in range(self.length):
                kj1 = np.dot(self.x[j], self.x[i1])
                kji = np.dot(self.x[j], self.x[i])
                # eij = self.alphas[j] * self.y[j] * kji - self.y[i]
                eij = self.alphas[j] * self.y[j] * (kj1 - kji) + self.y[i] - self.y[i1]
                ei += eij
            e = np.append(e, abs(ei))
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
        new_a2 = a2 + (y2 * (e[i1] - e[i2])) / k
        new_a2 = self.clip_a2(y1, y2, a1, a2, new_a2)
        self.alphas[i2] = new_a2

        # Update a1
        new_a1 = a1 + y1*y2*(a2 - new_a2)
        if new_a1 > self.c:
            new_a1 = self.c
        self.alphas[i1] = new_a1

    def squash_alpha(self):
        for i in range(self.length):
            if self.alphas[i] < self.epsilon:
                self.alphas[i] = 0

    def calculate_b(self):
        for i in range(self.length):
            if self.alphas[i] > 0:
                self.b = self.y[i] - np.matmul(self.get_w(), self.x[i])
                return

    def plot(self):
        class_1_x = [x[0] for x, y in zip(self.x, self.y) if y == 1]
        class_neg1_x = [x[0] for x, y in zip(self.x, self.y) if y == -1]
        class_1_y = [x[1] for x, y in zip(self.x, self.y) if y == 1]
        class_neg1_y = [x[1] for x, y in zip(self.x, self.y) if y == -1]
        plt.plot(class_1_x, class_1_y, 'ro')
        plt.plot(class_neg1_x, class_neg1_y, 'bo')

        x = -self.b / self.get_w()
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = self.b + np.transpose(self.get_w()) * x_vals
        plt.plot(x_vals, y_vals, '--')
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.show()

def main():

    data = read_data("data.csv")
    svm = SvmClassifier(data)
    while True:
        # svm.plot()
        i1, x1 = svm.get_x1()
        i2, x2 = svm.get_x2(i1)
        svm.update_a1_a2(i1, x1, i2, x2)
        svm.squash_alpha()
        svm.calculate_b()



if __name__ == '__main__':
    main()
