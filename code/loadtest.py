from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras import backend as K
import pandas
from os import listdir
import numpy as np
import csv


num_classes = 144

dataframe = pandas.read_csv("databases/database_onehot.csv", header=None)
dataset = dataframe.values

csv_labels = []

with open("databases/database_onehot_type.csv", 'r') as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        x = line[2]
        x = x.strip("[]")
        x = "".join([c for c in x if c not in '\n'])
        x = x.split(" ")
        x = [c for c in x if not c == ""]
        x = [c.strip('.') for c in x]
        x = [int(c) for c in x]
        x = np.array(x)
        csv_labels.append(x)

#the labels - Y
csv_names = dataset[:,0]

training_data_names = listdir("./images/training")
test_data_names = listdir("./images/validation")

#strip the jpg names
#training_data_names = [x[:-4] for x in training_data_names]
#test_data_names = [x[:-4] for x in test_data_names]

x_train = []
x_test = []

y_train = []
y_test = []

for name in training_data_names:
    img = load_img("./images/training/"+name)
    x = img_to_array(img)
    #x = x.reshape((1,) + x.shape)
    #print(x)
    #add the image to the training set
    x_train.append(x)
    #x_train = np.append(x_train, [x])
    #add the corresponding label from the csv file
    itemindex = np.where(csv_names==name[:-4])[0][0]
    #print(itemindex)
    y_train.append(csv_labels[itemindex])
    #np.append(y_train, csv_labels[itemindex])
x_train = np.array(x_train)
y_train = np.array(y_train)


for name in test_data_names:
    img = load_img("./images/validation/"+name)
    x = img_to_array(img)
    #x = x.reshape((1,) + x.shape)

    #add the image to the training set
    x_test.append(x)
    #x_test = np.append(x_test, [x])
    #add the corresponding label from the csv file
    itemindex = np.where(csv_names==name[:-4])[0][0]
    y_test.append(csv_labels[itemindex])
    #np.append(y_test, csv_labels[itemindex])
x_test = np.array(x_test)
y_test = np.array(y_test)



#print(y_train)
# dimensions of our images.
img_width, img_height = 256, 256

train_data_dir = 'images/training'
validation_data_dir = 'images/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 100
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


multiclass_train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

multiclass_test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)

model.fit_generator(multiclass_train_generator,
                    steps_per_epoch=len(x_train) / batch_size, epochs=epochs)

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     verbose = 1,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)

#print(train_generator.filenames)
#print(y_test)
