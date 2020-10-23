from tqdm import tqdm
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.applications import vgg16, xception, resnet50
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input

def pretrain_images(imageIDs, network):
    unique_imageIDs = list(set(imageIDs))
    print(f"Pre-training {len(unique_imageIDs)} unique images")
    imagedict = {}
    target_dim_x = 0
    target_dim_y = 0
    #model to pretrain on
    if (network == "VGG16"):
        model = vgg16.VGG16()
        target_dim_x = 224
        target_dim_y = 224
    elif (network == "ResNet50"):
        model = resnet50.ResNet50()
        target_dim_x = 224
        target_dim_y = 224
    elif (network == "Xception"):
        model = xception.Xception()
        target_dim_x = 299
        target_dim_y = 299
    imagesubset = []
    iteration = 0
    for i,name in tqdm(enumerate(unique_imageIDs),total=len(unique_imageIDs)):
        img = load_img(f"./images/{name}.jpg", target_size=(target_dim_x, target_dim_y))
        x = img_to_array(img)
        imagesubset.append(x)
        imagedict[name] = i
        if(len(imagesubset) == 1000):
            x = np.array(imagesubset)
            x = preprocess_input(x)
            x = model.predict(x)
            with open(f"preprocessed_{network}_{str(iteration)}.pickle",'wb') as out_file:
                pickle.dump(x, out_file)
            iteration += 1
            imagesubset = []
            del(x)
    x = np.array(imagesubset)
    x = preprocess_input(x)
    x = model.predict(x)
    with open(f"preprocessed_{network}_{str(iteration)}.pickle",'wb') as out_file:
        pickle.dump(x, out_file)
    iteration += 1
    del(x)
    del(imagesubset)

    images = []
    for i in tqdm(range(iteration)):
        with open(f"preprocessed_{network}_{str(i)}.pickle",'rb') as in_file:
            x = pickle.load(in_file)
            for j in range(x.shape[0]):
                images.append(x[j])

    resultimages = []
    for name in tqdm(imageIDs):
        idx = imagedict[name]
        resultimages.append(images[idx])
    
    return np.array(resultimages)
