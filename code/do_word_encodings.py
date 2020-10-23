import logging
import os, sys
print("Doing standard imports...")
logging.info("Doing standard imports...")
import csv, os, pickle, random
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print("Importing tensorflow...")
logging.info("Importing tensorflow...")
import tensorflow as tf
print("Importing keras...")
logging.info("Importing keras")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Activation, Dropout, Flatten, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from pretrain_images import pretrain_images
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import backend as K

NUMBER_OF_CLASSES = 160
NUMBER_OF_TATE_CLASSES = 364
MOST_COMMON = 26

def parse_materialvec(x):
    x = x.strip("[]")
    x = "".join([c for c in x if c not in '\n'])
    x = x.split(" ")
    x = [c for c in x if not c == ""]
    x = [c.strip('.') for c in x]
    x = [int(c) for c in x]
    return np.array(x)

def load_data(MAX_NUM_WORDS, EMBEDDING_DIM, GLOVE_DIR="glove", network="VGG16", tate=False, database="database_reduced.csv"):
    if(tate):
        NUMBER_OF_CLASSES = NUMBER_OF_TATE_CLASSES
    else:
        NUMBER_OF_CLASSES = 160
    #Attempt to load in the data from pickle files. If that doesn't work, we will generate it now.
    db_path = 'databases/' + database
    try:
        with open("image_ids.pickle", "rb") as in_file:
            imageIDs = pickle.load(in_file)

        with open("corresponding_class_embeddings.pickle", "rb") as in_file:
            corresponding_class_embeddings = pickle.load(in_file)

        with open("embedding_matrix.pickle", "rb") as in_file:
            embedding_matrix = pickle.load(in_file)

        with open("corresponding_similarities.pickle", "rb") as in_file:
            corresponding_similarities = pickle.load(in_file)

        with open("pretrained_images_"+network+".pickle","rb") as in_file:
            image_features = pickle.load(in_file)

        with open("class_labels.pickle", "rb") as in_file:
            class_labels = pickle.load(in_file)

        logging.info("Successfully loaded data from pickle")
    except Exception as e:
        logging.error("Pickle load failed!")
        logging.error(e)

        #Get the pre-trained features (100d vectors) for 40,000 words.
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
        for line in tqdm(f,total=400000):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        #Get all of the words used to describe each class
        class_word_sets = defaultdict(list)
        for i in range(NUMBER_OF_CLASSES):
            class_word_sets[i] =  []#have to do this because for some reason some of the classes don't show up in the next bit. This is hardcoded!
        if (tate):
            thefilename = "../databases/database_reduced.csv"
        else:
            thefilename = "databases/database_reduced.csv"
        with open(thefilename, 'r') as in_file:
            reader = csv.reader(in_file)
            for line in tqdm(reader,total=53870):
                if (line[4] != ""):
                    onehot_classes = parse_materialvec(line[2])
                    for i in np.nonzero(onehot_classes)[0]:
                        line[4] = line[4].replace("-","")
                        for word in line[4].split(" "):
                            class_word_sets[i].append(word)
        if (tate):
            thefilename = "databases/database_final.csv"
        else:
            thefilename = "tateenvironment/databases/database_final.csv"
        with open(thefilename,'r') as in_file:
            reader = csv.reader(in_file)
            for line in tqdm(reader,total=53870):
                if (line[4] != ""):
                    onehot_classes = parse_materialvec(line[2])
                    for i in np.nonzero(onehot_classes)[0]:
                        line[4] = line[4].replace("-","")
                        for word in line[4].split(" "):
                            class_word_sets[i].append(word)

        for i in range(len(class_word_sets)):
            class_word_sets[i] = [c[0] for c in Counter(class_word_sets[i]).most_common(MOST_COMMON)][:26]
            if(len(class_word_sets[i])>26):
                class_word_sets[i] = class_word_sets[i][:26]
        class_word_lists_filtered = []
        print(class_word_sets)
        #Filter the words by whether they are in the embeddings matrix, and convert to a list of lists
        for (k,cl) in class_word_sets.items():
            class_word_lists_filtered.append(" ".join([word for word in cl if word in embeddings_index.keys()]))
            print(f"{k}:{len([word for word in cl if word in embeddings_index.keys()])}")
        print(class_word_lists_filtered[NUMBER_OF_CLASSES-1])
        #Tokenize the class descriptions
        for sequence in class_word_lists_filtered:
            print(sequence)
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(class_word_lists_filtered)
        embedded_class_words = tokenizer.texts_to_sequences(class_word_lists_filtered)
        sequences_lengths = [len(x) for x in embedded_class_words]
        for sequence in embedded_class_words:
            print(len(sequence))
        MAX_SEQUENCE_LENGTH = np.max(sequences_lengths)
        print(f"Maximum sequence length is {MAX_SEQUENCE_LENGTH}")

        word_index = tokenizer.word_index
        logging.info(f'Found {len(word_index)} unique tokens.')

        padded_class_embeddings = pad_sequences(embedded_class_words, maxlen=MAX_SEQUENCE_LENGTH)

        corresponding_class_embeddings = []
        imageIDs = []
        class_labels = []
        corresponding_similarities = []
        #Now create the data. Each image gets added once for every class it appears in
        with open(db_path, 'r') as in_file:
            reader = csv.reader(in_file)
            for line in tqdm(reader):
                if (line[4] != ""):
                    onehot_classes = parse_materialvec(line[2])
                    for i in np.nonzero(onehot_classes)[0]:
                        imageIDs.append(line[0])
                        class_labels.append(i)
                        corresponding_class_embeddings.append(padded_class_embeddings[i])
                        corresponding_similarities.append(1)

                        #Add some classes not in the thing as counter-examples
                        used = []
                        for i in range(1):
                            j = random.randint(0,NUMBER_OF_CLASSES-1)
                            while (np.any(j == np.nonzero(onehot_classes)) or j in used):
                                j = random.randint(0,NUMBER_OF_CLASSES-1)
                            used.append(j)
                            class_labels.append(j)
                            imageIDs.append(line[0])
                            corresponding_class_embeddings.append(padded_class_embeddings[j])
                            corresponding_similarities.append(0)

        #Convert the data to numpy arrays
        corresponding_class_embeddings = np.asarray(corresponding_class_embeddings)
        imageIDs = np.asarray(imageIDs)
        corresponding_similarities = np.asarray(corresponding_similarities)

        #Shuffle it around
        indices = np.arange(corresponding_class_embeddings.shape[0])
        np.random.shuffle(indices)
        corresponding_class_embeddings = corresponding_class_embeddings[indices]
        imageIDs = imageIDs[indices]
        corresponding_similarities = corresponding_similarities[indices]
        corresponding_class_embeddings = corresponding_class_embeddings[:5000]
        imageIDs = imageIDs[:5000]
        corresponding_similarities = corresponding_similarities[:5000]

        #Generate the matrix of embeddings, for the embeeding layer
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in tqdm(word_index.items(), total=len(word_index)):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        try:
            #Train on image features
            with open(f"pretrained_images_{network}.pickle","rb") as in_file:
                image_features = pickle.load(in_file)
                print(image_features.shape)
                logging.info(image_features.shape)
        except:
            image_features = pretrain_images(imageIDs, network)

        with open(f"pretrained_images_{network}.pickle","wb") as out_file:
            pickle.dump(image_features, out_file)

        with open("image_ids.pickle", "wb") as out_file:
            pickle.dump(imageIDs, out_file)

        with open("corresponding_class_embeddings.pickle", "wb") as out_file:
            pickle.dump(corresponding_class_embeddings, out_file)

        with open("embedding_matrix.pickle", "wb") as out_file:
            pickle.dump(embedding_matrix, out_file)

        with open("corresponding_similarities.pickle", "wb") as out_file:
            pickle.dump(corresponding_similarities, out_file)

        with open("class_labels.pickle", "wb") as out_file:
            pickle.dump(class_labels, out_file)

        logging.info("Successfully generated data to pickle")

    return (image_features, corresponding_class_embeddings, embedding_matrix, corresponding_similarities, class_labels)

#def zsl_split_data(feature_data_matrix, image_scores, labelvectors, batch_size, VALIDATION_SPLIT_PERCENTAGE=20):
def zsl_split_data(ccembeddings, imfeatures, labels, class_labels, batch_size, VALIDATION_SPLIT_PERCENTAGE=0.2):
    print("==========================================================")
    print("SPLITTING DATA and ommitting "+str(VALIDATION_SPLIT_PERCENTAGE*100)+" percent of classes for Validation")
    print(f"Shape of class embeddings is {ccembeddings.shape}.\nShape of image features is {imfeatures.shape}.\nShape of labels is {labels.shape}.\nShape of class labels is {len(class_labels)}.")
    #get number of validation classes
    num_val_classes = int(np.floor(NUMBER_OF_CLASSES*VALIDATION_SPLIT_PERCENTAGE))

    #generate unique validation classes
    val_classes = random.sample(range(0, NUMBER_OF_CLASSES-1), num_val_classes)

    #setup training data arrays
    description_train = []
    image_train_scores = []
    output_vectors_train = []

    #setup validation data arrays
    description_validation = []
    image_validation_scores = []
    output_vectors_validation = []

    #loop over everything in th data set
    for i in tqdm(range(0, len(class_labels)-1)):
        #assume that a sample is part of the training data unless we add it to validation
        train_data_sample = True
        #loop over the validation classes
        for j in val_classes:
            #if the data instance has a label belonging to a validation class
            #add it to the validation set
            if j == class_labels[i]:
                description_validation.append(ccembeddings[i])
                image_validation_scores.append(imfeatures[i])
                output_vectors_validation.append(labels[i])
                train_data_sample = False
                #only one instance of a validation class is enough to condem the
                #data to a lifetime in the validation set
                break
        #if no validation classes are found add it to the training data
        if (train_data_sample):
            description_train.append(ccembeddings[i])
            image_train_scores.append(imfeatures[i])
            output_vectors_train.append(labels[i])

    #print info
    print(str(len(description_validation))+" Objects in Validation Set")
    print(str(len(description_train))+" Objects in Training Set")

    #check that the sizes of the validation data can be divided into batches correctly
    if len(description_validation) % batch_size != 0:
        print("WARNING: "+str(len(description_validation))+" Objects in Validation set in zsl_split_data() is not divisible by batch_size "+str(batch_size))
        #Get the remainder and remove the extrenuious data from validation set
        data_omision_size = len(description_validation) % batch_size
        print("Omitting "+str(data_omision_size)+" objects from Validation set")
        description_validation = description_validation[:-data_omision_size]
        image_validation_scores = image_validation_scores[:-data_omision_size]
        output_vectors_validation = output_vectors_validation[:-data_omision_size]

    #check that the sizes of the train data can be divided into batches correctly
    if len(description_train) % batch_size != 0:
        print("WARNING: "+str(len(description_train))+" Objects in Training set in zsl_split_data() is not divisible by batch_size "+str(batch_size))
        #Get the remainder and remove the extrenuious data from training set
        data_omision_size = len(description_train) % batch_size
        print("Omitting "+str(data_omision_size)+" objects from Training set")
        description_train = description_train[:-data_omision_size]
        image_train_scores = image_train_scores[:-data_omision_size]
        output_vectors_train = output_vectors_train[:-data_omision_size]

    #convert data to numpy arrays
    description_train = np.asarray(description_train)
    image_train_scores = np.asarray(image_train_scores)
    output_vectors_train = np.asarray(output_vectors_train)
    description_validation = np.asarray(description_validation)
    image_validation_scores = np.asarray(image_validation_scores)
    output_vectors_validation = np.asarray(output_vectors_validation)

    print("==========================================================")

    return (image_train_scores, description_train, output_vectors_train, image_validation_scores, description_validation, output_vectors_validation)

#sorts the data by classes and passes it to split_data
def split_data_on_classes(ccembeddings, imfeatures, labels, batch_size, VALIDATION_SPLIT=0, nb_validation_classes=0):
    print("==========================================================")
    print("SPLITTING DATA - ON CLASSES")

    #Zip data together so it can be sorted on labels
    temp_data_0 = zip(ccembeddings, imfeatures)
    temp_data_1 = sorted(zip(labels,temp_data_0))

    #sort the data
    imfeatures = [x for _,(_,x) in temp_data_1]
    ccembeddings = [x for _,(x,_) in temp_data_1]

    #NOTE: VALIDATE THE MAX CLASSES
    #TRAIN_CLASSES = MAX_CLASSES - VALIDATION_CLASSES
    num_train_classes = NUMBER_OF_CLASSES-1 - nb_validation_classes

    class_count = 0
    index = 0
    prev_class = ""

    #get the index of the first occurence of the first validation class
    while (class_count < num_train_classes):
        if labels[index] != prev_class:
            prev_class = labels[index]
            class_count = class_count + 1

        index = index + 1

    nb_validation_samples = len(labels) - index

    return split_data(ccembeddings, imfeatures, labels, batch_size, VALIDATION_SPLIT, nb_validation_samples)

def split_data(ccembeddings, imfeatures, labels, batch_size, VALIDATION_SPLIT=0, nb_validation_samples=0):

    print("==========================================================")
    print("SPLITTING DATA")

    print(ccembeddings.shape)
    print(imfeatures.shape)
    print(labels.shape)

    if (nb_validation_samples == 0):
        nb_validation_samples = int(VALIDATION_SPLIT * ccembeddings.shape[0])

    nb_train_samples = nb_validation_samples

    if nb_validation_samples % batch_size != 0:
        print("WARNING: nb_validation_samples "+str(nb_validation_samples)+" in split_data() is not divisible by batch_size "+str(batch_size))
        data_omision_size = nb_validation_samples % batch_size
        print("Changing nb_validation_samples from "+str(nb_validation_samples)+" to "+str(nb_validation_samples - data_omision_size)+" to prevent size mismatch error")
        nb_validation_samples = nb_validation_samples - data_omision_size
        print(str(len(ccembeddings)))

        nb_train_samples = nb_validation_samples + ((len(ccembeddings) - nb_validation_samples) % batch_size)
        print(str(nb_train_samples))

    description_train = ccembeddings[:-nb_train_samples]
    image_train_scores = imfeatures[:-nb_train_samples]

    if (len(image_train_scores) % batch_size != 0) :
        raise ValueError("TRAINING DATA("+str(len(image_train_scores))+") NOT DEVISABLE BY BATCH SIZE("+str(batch_size)+")!")

    output_vectors_train = labels[:-nb_train_samples]


    description_validation = ccembeddings[-nb_validation_samples:]
    image_validation_scores = imfeatures[-nb_validation_samples:]

    if (len(image_validation_scores) % batch_size != 0) :
        raise ValueError("VALIDATION DATA("+str(len(image_validation_scores))+") NOT DEVISABLE BY BATCH SIZE("+str(batch_size)+")!")

    output_vectors_validation = labels[-nb_validation_samples:]
    # print("==========================================================")
    # #get and load training images
    # print("Loading training images...")
    # logging.info("Loading training images...")
    # image_train = []
    # for name in tqdm(image_train_filenames):
    #     img = load_img("./images/"+name+".jpg")
    #     x = img_to_array(img)
    #     x = np.ndarray.flatten(x)
    #     image_train.append(x)
    # print("Converting image list to numpy array (this takes a while...")
    # logging.info("Converting image list to numpy array (this takes a while...")


    # image_train = np.array(image_train)

    # #get and load training images
    # print("Loading validation images...")
    # logging.info("Loading validation images...")
    # image_validation = []
    # for name in tqdm(image_validation_filenames):
    #     img = load_img("./images/"+name+".jpg")
    #     x = img_to_array(img)
    #     x = np.ndarray.flatten(x)
    #     image_validation.append(x)
    # image_validation = np.array(image_validation)
    return (image_train_scores, description_train, output_vectors_train, image_validation_scores, description_validation, output_vectors_validation)

def build_our_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    #print(description_train)
    # dimensions of our images.
    img_width, img_height = 256, 256

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
    model.add(Dense(MAX_SEQUENCE_LENGTH*EMBEDDING_DIM))
    model.add(Activation('sigmoid'))

    opt = SGD()

    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    return model

def train_model(model, image_train, description_train, label_vectors_train, image_validation, description_validation, label_vectors_validation, batch_size, epochs, description):
    """
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    """
    path = 'logs/' + description
    os.mkdir(path)
    path = './' + path
    tensorboard = TensorBoard(log_dir=path, histogram_freq=0,
                          write_graph=True, write_images=False)

    x_train = [image_train,description_train]
    #y_train = np.concatenate([label_vectors_train,label_vectors_train], axis=-1)
    y_train = label_vectors_train
   # multiclass_train_generator = train_datagen.flow(x_train, y_trairn, batch_size=batch_size)

    x_val = [image_validation, description_validation]
    #y_val = np.concatenate([label_vectors_validation, label_vectors_validation], axis=-1)
    y_val = label_vectors_validation
   # multiclass_test_generator = test_datagen.flow(x_val,y_val, batch_size=batch_size)

    keras_model = model.fit(
                    x = x_train,
                    y = y_train,
                    validation_data = (x_val, y_val),
                    batch_size = batch_size,
                    epochs = epochs,
                    callbacks=[tensorboard])

    return model

    """
    model.fit_generator(multiclass_train_generator,
                        validation_data=multiclass_test_generator,
                        validation_steps=len(image_validation) / batch_size,
                        steps_per_epoch=len(image_train) / batch_size, epochs=epochs)
    """

# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)

# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# preds = Dense(labeldim, activation='softmax')(x)

# model = Model(sequence_input, preds)
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])

# model.fit(description_train, description_train,
#           batch_size=128,
#           epochs=100,
#           validation_data=(image_validation, description_validation))
