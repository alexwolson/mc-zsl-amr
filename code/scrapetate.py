import requests, json, shutil, csv, urllib
from tqdm import tqdm
from string import ascii_lowercase
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from inflection import singularize

resultlist = []
csvfile = open('collection/artwork_data.csv', 'r')
csvreader = csv.reader(csvfile)
stopWords = set(stopwords.words('english'))
number_of_lines = 69202
def strip_description(j):
    keywords = []
    if isinstance(j, list):
        for k in j:
            keywords += strip_description(k)
    else:
        if "children" in j.keys():
            for k in j["children"]:
                keywords += strip_description(k)
        if "name" in j.keys():
            js = j["name"].replace(","," ")
            js = js.replace("/"," ")
            js = js.split(" ")
            for k in js:
                if len(k) != 0 and ":" not in k and k not in stopWords:
                    keywords += [singularize(k)]
    return keywords

print("Starting...")
for row in tqdm(csvreader, total=number_of_lines):
    if row[0] == "id":
        continue
    try:
        objectID = str(row[1]).lower() + "-" + str(row[0]).lower()
        imagestream = requests.get(row[18], stream=True)
        with open(f"tateimages/{objectID}.jpg", 'wb') as out_file:
            shutil.copyfileobj(imagestream.raw, out_file)
        del imagestream
        materials = row[7]
        letterFolder = [c for c in str(row[1]).lower() if c in ascii_lowercase][0]
        tripleFolder = "".join([c for c in str(row[1]).lower() if c not in ascii_lowercase][0:3])
        f = open(f"collection/artworks/{letterFolder}/{tripleFolder}/{objectID}.json", 'rb')
        j = json.load(f)
        description = list(set(strip_description(j["subjects"])))
        resultlist.append([objectID, "1", materials, "1", description])
    except Exception as e:
        if "URL" not in str(e):
            print(e)
        continue

with open("database.csv", 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(resultlist)
