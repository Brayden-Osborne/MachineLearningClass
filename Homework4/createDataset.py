import csv
import random
with open('networkDataSet.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['x1','x2','output'])
    for i in range(1000):
        x1 = random.uniform(-100,100)
        x2 = random.uniform(-100,100)
        y = x1 + 2*x2 -2 
        if y > 0 :
            writer.writerow([str(x1),str(x2),'1'])
        else:
            writer.writerow([str(x1),str(x2),'0'])
