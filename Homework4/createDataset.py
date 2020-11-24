import csv
import random

def read_dataset():
    data = []
    with open('networkDataSet.csv', 'r') as f:
        tmp = csv.reader(f)
        for idx, row in enumerate(tmp):
            if idx == 0:
                header = {row_idx: row_val for row_idx, row_val in enumerate(row)}
            elif len(row) == 0:
                continue
            else:
                data.append([float(val) for val in row])
    return data


def main():
    with open('networkDataSet.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['x1', 'x2', 'output'])
        for i in range(1000):
            x1 = random.uniform(-100, 100)
            x2 = random.uniform(-100, 100)
            y = x1 + 2 * x2 - 2
            if y > 0:
                writer.writerow([str(x1), str(x2), '1'])
            else:
                writer.writerow([str(x1), str(x2), '-1'])


if __name__ == '__main__':
    main()
