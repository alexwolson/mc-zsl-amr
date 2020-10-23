import csv, string, math
import numpy as np
np.set_printoptions(suppress=True,threshold=np.nan)
resultlist = []
materials = set()
with open("databases/database_onehot.csv", 'r') as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        if (not isinstance(line[2],list)):
            line[3] = [x.strip("' ") for x in line[3][1:-1].split(',')]
        for mat in line[3]:
            materials.add(mat)
    materials = list(materials)
totals = np.zeros((1,len(materials)))
with open("databases/database_onehot.csv", 'r') as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        oh = np.zeros((1,len(materials)))
        if (not isinstance(line[2],list)):
            line[3] = [x.strip("' ") for x in line[3][1:-1].split(',')]
        for mat in line[3]:
            oh[(0,materials.index(mat))] = 1
        line[3] = oh
        totals += oh
        resultlist.append(line)
print(totals)
print(materials)
for (t,m) in zip(np.ndarray.tolist(totals)[0], materials):
    print ("%s: %d" % (m,t))

with open("databases/database_onehot_type.csv", 'w') as out_file:
    writer = csv.writer(out_file)
    writer.writerows(resultlist)