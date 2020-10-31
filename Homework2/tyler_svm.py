import numpy as np
import random as rnd
import csv
import matplotlib.pyplot as plt


class SVM():
    def __init__(self, maxIter=100, C=1.0, epsilon=0.001):
        self.maxIter = maxIter
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        count = 0
        while True:
            count += 1
            alphaPrev = np.copy(alpha)
            for j in range(0, n):
                i = self.genRandomInt(0, n-1, j)
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                k_ij = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2 * self.kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alphaPrimeJ, alphaPrimeI = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alphaPrimeJ, alphaPrimeI, y_j, y_i)

                self.w = self.calcWeights(alpha, y, X)
                self.b = self.calcB(X, y, self.w)

                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                alpha[j] = alphaPrimeJ + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alphaPrimeI + y_i * y_j * (alphaPrimeJ - alpha[j])

            diff = np.linalg.norm(alpha - alphaPrev)
            if diff < self.epsilon:
                break

            if count >= self.maxIter:
                print("Max number of iterations reached")
                return

        self.plot(X, y)
        self.b = self.calcB(X, y, self.w)
        self.w = self.calcWeights(alpha, y, X)

        alpha_idx = np.where(alpha > 0)[0]
        supportVectors = X[alpha_idx, :]

        return supportVectors, count

    def predict(self, X):
        return self.h(X, self.w, self.b)

    def calcB(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calcWeights(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))

    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alphaPrimeJ, alphaPrimeI, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alphaPrimeJ - alphaPrimeI), min(C, C - alphaPrimeI + alphaPrimeJ))
        else:
            return (max(0, alphaPrimeI + alphaPrimeJ - C), min(C, alphaPrimeI + alphaPrimeJ))

    def genRandomInt(self, a,b,z):
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = rnd.randint(a, b)
            cnt += 11
        return i

    def kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def plot(self, X, Y):
        class_1_x = [x[0] for x, y in zip(X, Y) if y == 1]
        class_neg1_x = [x[0] for x, y in zip(X, Y) if y == -1]
        class_1_y = [x[1] for x, y in zip(X, Y) if y == 1]
        class_neg1_y = [x[1] for x, y in zip(X, Y) if y == -1]
        plt.plot(class_1_x, class_1_y, 'ro')
        plt.plot(class_neg1_x, class_neg1_y, 'bo')

        # axes = plt.gca()
        # x_vals = np.array(axes.get_xlim())
        # y_vals = self.b + np.dot(self.w.T, x_vals.T)
        # plt.plot(x_vals, y_vals, '--')
        a = -self.w[0] / self.w[1]
        xx = np.linspace(0, 315)
        yy = a * xx - self.b / self.w[1]
        plt.plot(xx, yy, '--')
        plt.show()


# data = """
# 243 3 -1
# 116 165 1
# 198 127 -1
# 184 234 1
# 165 231 1
# 160 46 -1
# 70 169 1
# 300 94 -1
# 95 62 -1
# 61 186 1
# 79 140 1
# 245 185 -1
# 264 2 -1
# 192 146 -1
# 274 104 -1
# 170 37 -1
# 172 144 -1
# 222 131 -1
# 35 204 1
# 105 199 1
# 146 40 -1
# 35 68 1
# 221 41 -1
# 39 16 -1
# 18 77 1
# 32 134 1
# 21 61 1
# 109 266 1
# 134 172 1
# 100 183 1
# 45 158 1
# 263 193 -1
# 173 88 -1
# 299 98 -1
# 81 240 1
# 69 235 1
# 152 189 1
# 176 4 -1
# 252 37 -1
# 120 228 1
# 241 133 -1
# 86 214 1
# 43 288 1
# 221 77 -1
# 210 117 -1
# 181 75 -1
# 11 71 1
# 280 157 -1
# 293 115 -1
# 154 85 -1
# 44 160 1
# 49 192 1
# 35 61 1
# 161 32 -1
# 247 107 -1
# 254 201 -1
# 91 142 1
# 261 57 -1
# 152 9 -1
# 41 116 1
# 279 166 -1
# 3 283 1
# 61 168 1
# 53 31 -1
# 293 67 -1
# 185 229 1
# 146 17 -1
# 165 195 1
# 210 51 -1
# 39 155 1
# 108 85 -1
# 86 195 1
# 266 117 -1
# 49 124 1
# 124 20 -1
# 211 114 -1
# 118 159 1
# 250 137 -1
# 258 178 -1
# 210 291 1
# 95 29 -1
# 255 18 -1
# 75 209 1
# 282 254 -1
# 6 180 1
# 42 237 1
# 50 109 1
# 1 270 1
# 151 39 -1
# 263 173 -1
# 86 217 1
# 82 216 1
# 143 214 1
# 30 237 1
# 205 148 -1
# 265 111 -1
# 277 236 -1
# 290 27 -1
# 260 126 -1
# 270 300 1
# 147 175 1
# 70 229 1
# 56 166 1
# 101 273 1
# 157 273 1
# 237 136 -1
# 181 15 -1
# 20 225 1
# 204 158 -1
# 16 69 1
# 78 240 1
# 297 98 -1
# 75 300 1
# 123 262 1
# 54 99 1
# 260 76 -1
# 104 279 1
# 89 132 1
# 67 155 1
# 275 32 -1
# 108 210 1
# 148 182 1
# 68 158 1
# 243 297 1
# 167 239 1
# 78 194 1
# 69 266 1
# 186 128 -1
# 27 291 1
# 95 198 1
# 83 180 1
# 142 94 -1
# 235 296 1
# 281 220 -1
# 275 241 -1
# 295 45 -1
# 79 50 -1
# 235 211 -1
# 293 2 -1
# 199 46 -1
# 28 286 1
# 235 298 1
# 162 109 -1
# 223 99 -1
# 297 186 -1
# 55 153 1
# 33 171 1
# 165 273 1
# 62 199 1
# 176 236 1
# 262 82 -1
# 194 55 -1
# 161 252 1
# 67 15 -1
# 91 40 -1
# 111 90 -1
# 249 170 -1
# 248 134 -1
# 173 278 1
# 3 300 1
# 130 181 1
# 195 244 1
# 16 245 1
# 215 19 -1
# 173 62 -1
# 30 105 1
# 170 64 -1
# 137 99 -1
# 218 83 -1
# 104 220 1
# 90 213 1
# 12 87 1
# 163 2 -1
# 66 93 1
# 292 103 -1
# 15 134 1
# 155 60 -1
# 2 111 1
# 47 96 1
# 113 288 1
# 148 255 1
# 270 177 -1
# 76 33 -1
# 179 59 -1
# 246 149 -1
# 161 279 1
# 174 97 -1
# 204 89 -1
# 290 181 -1
# 17 79 1
# 277 93 -1
# 177 270 1
# 59 276 1
# 184 156 -1
# 191 149 -1
# 81 265 1
# 137 185 1
# 231 44 -1
# 244 34 -1
# 130 290 1
# 286 148 -1
# 137 190 1
# 26 119 1
# 187 238 1
# 247 207 -1
# 108 281 1
# 144 97 -1
# 169 294 1
# 55 155 1
# 233 154 -1
# 171 36 -1
# 299 193 -1
# 298 100 -1
# 35 294 1
# 128 285 1
# 64 8 -1
# 268 110 -1
# 162 19 -1
# 76 280 1
# 268 101 -1
# 78 100 1
# 255 145 -1
# 70 210 1
# 217 105 -1
# 47 75 1
# 22 257 1
# 174 265 1
# 104 261 1
# 281 53 -1
# 13 67 1
# 267 132 -1
# 100 14 -1
# 296 188 -1
# 76 28 -1
# 11 173 1
# 261 198 -1
# 182 156 -1
# 145 21 -1
# 119 263 1
# 289 99 -1
# 10 286 1
# 114 273 1
# 211 288 1
# 161 63 -1
# 108 151 1
# 214 161 -1
# 192 13 -1
# 38 230 1
# 233 141 -1
# 162 204 1
# 36 159 1
# 184 287 1
# 147 122 -1
# 114 242 1
# 4 300 1
# 258 188 -1
# 268 76 -1
# 227 262 1
# 240 17 -1
# 83 279 1
# 282 136 -1
# 299 235 -1
# 286 79 -1
# 150 297 1
# 6 126 1
# 112 281 1
# """

with open('data.txt', 'r') as f:
    data = f.read()

xVals = []
yVals = []
for line in data.split('\n'):
    d = line.split(" ")
    if len(d) < 3:
        continue
    xVals.append( np.array( [float(d[0]), float(d[1])] ) )
    yVals.append(float(d[2]))

X = np.array(xVals)
y = np.array(yVals)

model = SVM()
support_vectors, iterations = model.fit(X, y)
yHat = model.predict(X)
correct = 0
for x, yO, yP in zip(X, y, yHat):
    print (x, ": ", yO, ", SMO prediction: ", yP)
    if yO == yP:
        correct += 1
print ("Accuracy: ", correct/len(y))
