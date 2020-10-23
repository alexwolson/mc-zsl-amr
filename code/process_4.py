import csv, string, math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

np.set_printoptions(suppress=True,threshold=np.nan)
resultlist = []
materials = set()
stopWords = set(stopwords.words('english'))
imageNames = set()
with open("databases/database_onehot_type.csv", 'r') as in_file:
    reader = csv.reader(in_file)
    for line in tqdm(reader):
        if (line[0] in imageNames):
            continue
        imageNames.add(line[0])
        if (not isinstance(line[4],list)):
            line[4] = "".join([c for c in line[4] if c.isalnum() or c in ["-"," "]])
            line[4] = " ".join([x.strip("' ").lower() for x in line[4].split(',')])
        print(f"IN: {line[4]}")
        w = word_tokenize(line[4])
        out = []
        for word in w:
            if word not in stopWords:
                out.append(word)
        out = sorted(list(set(out)))
        line[4] = " ".join(out)
        print(f"OUT: {line[4]}")
        resultlist.append(line)
with open("databases/database_final.csv", 'w') as out_file:
    writer = csv.writer(out_file)
    writer.writerows(resultlist)