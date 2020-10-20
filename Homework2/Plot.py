import csv
import sys

f = open("data.csv", 'rt')

reader = csv.reader(f)
data = []
for row in reader:
    data.append( (row[1], row[2]) )

print (data)
