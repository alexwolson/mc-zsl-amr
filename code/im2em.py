import csv, os, pandas
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from material_frequencies import parse_materialvec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Embedding
from keras.layers import Activation, Dropout, Flatten, Dense, Input, GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras import backend as K
from os import listdir

texts = []
tdict = {}
names = []
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 100
GLOVE_DIR = "glove"

with open("databases/database_final.csv", 'r') as in_file:
    reader = csv.reader(in_file)
    for line in reader:
        if (line[4] != ""):
            texts.append(line[4])
            tdict[line[0] + ".jpg"] = line[4]
            names.append(line[0] + ".jpg")

training_data_names = [n for n in listdir("./images/training") if n in names]
validation_data_names = [n for n in listdir("./images/validation") if n in names]

print("Got %d texts" % (len(texts)))

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labeldim = 144

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

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
    #print(itemindex)
    y_train.append(tdict[name])
    #np.append(y_train, csv_labels[itemindex])
x_train = np.array(x_train)
y_train = np.array(y_train)


for name in validation_data_names:
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


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(labeldim, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=100,
          validation_data=(x_val, y_val))
