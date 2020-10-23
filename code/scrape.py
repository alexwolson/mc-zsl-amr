import requests, json, shutil, csv, urllib
from tqdm import tqdm
page = 0
key = ""
era = 0
filecount = 0
url = "https://www.rijksmuseum.nl/api/en/collection?key=" + key + "&format=json&ps=100&imgonly=true&f.dating.period="
resultlist = []
colors = set()
for era in tqdm(range(0,22)):
    try:
        page = 0
        r = requests.get(url + str(era) + "&p=" + str(page)).json()
        while(r['count'] > 100 * page):
            #print(url + str(era) + "&p=" + str(page))
            r = requests.get(url + str(era) + "&p=" + str(page)).json()
            page += 1
            try:
                for artmeta in tqdm(r['artObjects']):
                    art = requests.get(artmeta['links']['self'] + "?key=" + key + "&format=json").json()
                    try:
                        #print("Processing %s by %s" % (art['artObject']['title'], art['artObject']['principalMakers'][0]['name']))
                        for color in art['artObject']['normalized32Colors']:
                            colors.add(color)
                        objectID = art['artObject']['objectNumber']
                        imagestream = requests.get(art['artObject']['webImage']['url'], stream=True)
                        with open(objectID + ".jpg", 'wb') as out_file:
                            shutil.copyfileobj(imagestream.raw, out_file)
                        del imagestream
                        date = art['artObject']['dating']['presentingDate']
                        materials = art['artObject']['materials']
                        collection = art['artObject']['objectCollection']
                        description = art['artObject']['classification']['iconClassDescription']
                        resultlist.append([objectID, date, materials, collection, description])
                    except:
                        continue
            except:
                continue
            with open("database_era"+str(era)+"_page_"+str(page)+".csv", 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerows(resultlist)
            resultlist = []
    except:
        continue

with open("colors.csv",'wb') as out_file:
    writer = csv.writer(out_file)
    writer.writerows(list(colors))
