import csv
import matplotlib.pyplot as plt
import numpy as np

'''
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
#plt.show()
'''


# Read in iris dataset
xVals = []
yVals = []
with open('iris_data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row == []:
            continue
        if row[-1] == 'Iris-setosa':
            yVals.append(1)
        else:
            yVals.append(-1)
        xVals.append(float(row[0]))

x = np.array(xVals)
y = np.array(yVals)
print (x)
print (y)

# Create alpha array
l = len(x)
alpha = np.random.rand(l)
#print (np.dot(alpha, y))
while abs(np.dot(alpha,y)) > 0.00000001:
    alpha += np.dot(alpha,y) / l
    #print (np.dot(alpha, y))
print ("y dot alpha\n", np.dot(alpha,y))
print ("alpha\n", alpha)
