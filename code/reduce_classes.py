import csv, string, math

classes = {}
artworks_by_class = {}
selected_artwork_by_class = {}

with open("../databases/database.csv") as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        art_id = line[0]
        materials = line[2]
        materials = materials[2:]
        materials = materials[::-1]
        materials = materials[2:]
        materials = materials[::-1]
        materials = materials.replace('\'', "")
        elems = materials.split(',')
        for elem in elems:
            if len(elem) > 1:
                if elem[0] == ' ':
                    elem = elem[1:]
                if elem in classes:
                    classes[elem] = classes[elem] +1
                    artworks_by_class[elem].append(art_id)
                else:
                    classes[elem] = 1
                    artworks_by_class[elem] = [art_id]

    for key in artworks_by_class.keys():
        maximum = 147
        if len(artworks_by_class[key]) < maximum:
            maximum = len(artworks_by_class[key])

        if key not in selected_artwork_by_class:
            selected_artwork_by_class[key] = []

        index = 0
        for artwork in artworks_by_class[key]:
            if index < maximum:
                if artwork not in selected_artwork_by_class:
                    selected_artwork_by_class[key].append(artwork)
                    index = index + 1

        # for artwork in selected_artwork_by_class.keys():
        #     if len(selected_artwork_by_class[artwork]) > 147:
        #         print(artwork)
        #         print(len(selected_artwork_by_class[artwork]))

        selected_artwork = list(selected_artwork_by_class.values())
        flat_list = [item for sublist in selected_artwork for item in sublist]

        #print(flat_list)
        with open("selected_artwork.txt", 'w') as outfile:
            for i in flat_list:
                outfile.write(i+"\n")

        
        #print(elems)
