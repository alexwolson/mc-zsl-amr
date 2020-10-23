import csv

selected = []

"""
script to select max 147 images from each class
"""
with open("selected_artwork.txt", "r") as in_file:
	for line in in_file:
		selected.append(line[0:len(line)-2])

#print(selected)

with open("../databases/database_final.csv","r") as csv_infile, open("../databases/database_reduced.csv","w") as out_file:
    reader = csv.reader(csv_infile)
    for line in reader: 
    	if line[0] in selected:	
    		"""
    		sorry about the next few lines
    		"""
    		line[2] = "\"" + line[2] + "\""
    		for i in range(1, len(line)):
    			line[i-1] = line[i-1] + ','
    		line.append('\n')
    		out_file.writelines(line)
