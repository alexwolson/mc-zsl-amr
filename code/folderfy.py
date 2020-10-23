import csv, os

count = 0

with open("../database_processed.csv", 'r') as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        line[0] = "".join([c for c in line[0] if c not in "()"])
        if (not isinstance(line[3],list)):
            line[3] = "".join([x.strip("' ") for x in line[3][1:-1].split(',')])
        if line[3] == "":
            os.system("mv %s ./uncategorised" % (line[0] + ".jpg"))
        else:
            if (count == 5):
                folder = "validation"
                count = 0
            else:
                folder = "training"
                count = count + 1
            line[3] = "".join([c for c in line[3] if c.isalnum()])
            line[3] = "".join([c for c in line[3] if c not in "[]"])
            os.system("mkdir %s" % (folder))
            os.system("mkdir %s/%s" % (folder,line[3]))
            os.system("mv %s ./%s/%s" % (line[0] + ".jpg", folder, line[3]))