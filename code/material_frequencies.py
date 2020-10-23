import csv
import numpy as np

def parse_materialvec(x):
    x = x.strip("[]")
    x = "".join([c for c in x if c not in '\n'])
    x = x.split(" ")
    x = [c for c in x if not c == ""]
    x = [c.strip('.') for c in x]
    x = [int(c) for c in x]
    return np.array(x)

if __name__=="__main__":
    mat = np.zeros((144,144))
    key = ['uncategorised', 'kingwood (wood)', 'watercolor (paint)', 'silver thread', 'imitation tortoise shell', 'leaf (metal)', 'mahogany (wood)', 'quartz (mineral)', 'holly (wood)', 'wax', 'panel', 'boxwood (hardwood)', 'velvet (fabric weave)', 'steel (alloy)', 'copper alloy', 'chrysoprase', 'chintz', 'satin', 'lacquer (coating)', 'gilding (material)', 'lanyards', 'batiste', 'parchment (animal material)', 'porcelain (material)', 'walnut (hardwood)', 'gold leaf', 'leather', 'juniper (wood)', 'cherry (wood)', 'gold (metal)', 'cotton (textile)', 'linden', 'ruby (mineral)', 'plastic (organic material)', 'amethyst (mineral)', 'wood (plant material)', 'pipe clay', 'garnet (mineral)', 'hardboard', 'nylon', 'metal thread', 'teak (wood)', 'sandstone', 'jet (coal)', 'tin glaze', 'softwood', 'prepared paper', 'pencil', 'nautilus shell', 'dye', 'andesite', 'willow (wood)', 'ivory', 'brick (clay product)', 'textile materials', 'oil paint (paint)', 'serpentine (mineral)', 'turquoise (mineral)', 'aigrette', 'mother of pearl', 'stucco', 'brass (alloy)', 'tin (metal)', 'glass', 'terracotta (clay material)', 'sand', 'chamois (animal material)', 'ebony (wood)', 'marble (rock)', 'linen (material)', 'paste (glass)', 'stoneware', 'rope', 'alabaster (mineral)', 'graphite (mineral)', 'gouache (paint)', 'iron (metal)', 'paper', 'bone (material)', 'slate (rock)', 'purpleheart (wood)', 'citrine', 'fruitwood', 'bronze (metal)', 'cast iron', 'bone glue', 'aluminum (metal)', 'earthenware', 'spruce (wood)', 'silver leaf', 'extrusive rock', 'silver (metal)', 'brocade (textile)', 'silver stain', 'rosewood (wood)', 'freshwater pearl', 'tempera', 'satinwood (wood)', 'chalk', 'photographic paper', 'limestone', 'glue', 'elm (wood)', 'cellulose acetate', 'clay', 'olive (wood)', 'granite (rock)', 'silk', 'paint (coating)', 'celluloid (cellulosic)', 'encaustic paint', 'glaze', 'diamond (mineral)', 'soapstone (metamorphic rock)', 'copper (metal)', 'opal', 'jasper', 'oak (wood)', 'touchstone', 'beech (wood)', 'yarn', 'ink', 'pine (wood)', 'Bakelite (TM)', 'metal', 'ash (wood)', 'gypsum', 'jade (rock)', 'lapis lazuli (rock)', 'varnish', 'cloth', 'maple (wood)', 'rayon', 'cobalt (mineral)', 'felt (textile)', 'agate (chalcedony)', 'India ink (ink)', 'calf (leather)', 'poplar (wood)', 'cedar (wood)', 'suede', 'pear (wood)', 'deck paint', 'canvas']

    with open("databases/database_onehot_type.csv", 'r') as in_file:
        reader = csv.reader(in_file)
        for line in reader:
            x = parse_materialvec(line[2])
            for idx in np.nonzero(x)[0]:
                mat[idx] += x

    #mat = np.triu(mat)
    matnorm = mat/np.sum(mat,axis=0)

    np.fill_diagonal(matnorm, 0)
    #indices = np.nonzero(np.triu(matnorm))
    indices = np.nonzero(matnorm)
    bigvals = []

    for i in range(len(indices[0])):
            f = indices[0][i]
            m = indices[1][i]
            freq = matnorm[(f,m)]*2
            fname = key[f]
            mname = key[m]
            bigvals.append(((fname,mname),freq))

    sortkey = lambda x: x[1]
    bigvals = sorted(bigvals, key=sortkey, reverse=True)

    for ((fname,sname),f) in bigvals:
        if (f < 0.75):
            break
        print("%s has %s with frequency %f" % (sname,fname,f))