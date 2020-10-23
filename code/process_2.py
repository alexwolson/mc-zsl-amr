import csv, string, math
from inflection import singularize
import numpy as np
from collections import Counter
np.set_printoptions(suppress=True,threshold=np.nan)
resultlist = []
materials = []
with open("databases/database.csv", 'r') as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        if (not isinstance(line[2],list)):
            line[2] = line[2].replace("and",",")
            line[2] = line[2].replace("with",",")
            line[2] = line[2].replace("on",",")
            line[2] = [singularize(x.strip("' ")).lower() for x in line[2].split(',')]
        for mat in line[2]:
            materials+=[mat]
    materialc = Counter(materials)

    materials = set()
    for (m,c) in materialc.items():
        #if c > 2:
        materials.add(m)
    materials = list(materials)
totals = np.zeros((1,len(materials)))
with open("databases/database.csv", 'r') as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        oh = np.zeros((1,len(materials)))
        if (not isinstance(line[2],list)):
            line[2] = line[2].replace("and",",")
            line[2] = line[2].replace("with",",")
            line[2] = line[2].replace("on",",")
            line[2] = [singularize(x.strip("' ")).lower() for x in line[2].split(',')]
        if (len(line[2]) > 0):
            for mat in line[2]:
                if mat in materials:
                    oh[(0,materials.index(mat))] = 1
            line[2] = oh
            totals += oh
            resultlist.append(line)
        else:
            print(mat)
print(totals)
print(materials)
for (t,m) in zip(np.ndarray.tolist(totals)[0], materials):
    print ("%s: %d" % (m,t))

with open("databases/database_onehot.csv", 'w') as out_file:
    writer = csv.writer(out_file)
    writer.writerows(resultlist)