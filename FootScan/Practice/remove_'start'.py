import csv
import numpy as np

csv_path = './FootScan/Practice/COPsample_.csv'

header = ['xpoint','ypoint']
index = [0,0]
start = ["start","start"]

with open(csv_path, 'w', newline ="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(3000):
        if i == 600:
            writer.writerow(start)
        else:
            writer.writerow(index)

with open(csv_path) as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    l_2 = []
    i = 0
    for row in l[1:]:
        if row[0] != "start":
            row = [int(v) for v in row]
            l_2.append(row)
        else:
            pass

print(f"l(header含む) = {np.shape(l)}")
print(f"l_2 = {np.shape(l_2)}")

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(l_2)