from keras.preprocessing.image import load_img
import csv, string, math
resultlist = []
with open("databases/database.csv", 'r') as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        if (len("".join([c for c in line[4] if c.isalnum()])) == 0):
            continue
        line[0] = "".join([c for c in line[0] if c not in "()"])
        try:
            img = load_img("images/"+line[0]+".jpg")
        except Exception as e:
            print(e)
            continue
        date = ''.join([c for c in line[1] if c not in ['c','.',' ',] and c not in string.ascii_lowercase]).split('-')
        if (len(date) == 1):
            line[1] = date[0]
        elif (len(date) == 2):
            try:
                line[1] = math.floor(((int(date[1]) + int(date[0]))/2))
            except:
                print(f"{line[0]} is a weird one: {line[1]}")
                line[1] = int(input("Please enter the correct value manually: "))
        else:
            print(f"{line[0]} is a weird one: {line[1]}")
            line[1] = int(input("Please enter the correct value manually: "))
        resultlist.append(line)

with open("databases/database_processed.csv", 'w') as out_file:
    writer = csv.writer(out_file)
    writer.writerows(resultlist)
