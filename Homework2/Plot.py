import csv
import matplotlib.pyplot as plt
import numpy as np

f = open("data.csv", 'rt')

reader = csv.reader(f)
dataX1 = []
dataY1 = []
dataXN = []
dataYN = []
for row in reader:
    if '-1' in row[0]:
        dataXN.append(float(row[1]))
        dataYN.append(float(row[2]))
    else:
        dataX1.append(float(row[1]))
        dataY1.append(float(row[2]))


plt.plot(dataX1, dataY1, 'ro')
plt.plot(dataXN, dataYN, 'bo')
plt.show()
