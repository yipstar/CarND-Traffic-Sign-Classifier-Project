import random
import numpy as np
from sklearn.utils import shuffle
import math

def train_test_split_by_track(X_dataset, y_dataset):

    flatten = lambda l: [item for sublist in l for item in sublist]

    groups = math.ceil(X_dataset.shape[0] / 30)

    X_validation_list = []
    y_validation_list = []
    X_train_list = []
    y_train_list = []

    tracks_by_class = {}

    for i in range(0, groups):
        #print(i)
        #row = X_dataset[i]
        x_track = X_dataset[i*30:i*30+30, :]
        y_track = y_dataset[i*30:i*30+30]

        # select one image from the track randomly and add it to the validation set, add remaining images in the track to the training set

        #print(x_track.shape)
        #print(row.shape)

        image_index = random.randint(0, 29)
        image = x_track[image_index, :]
        #print(image)

        label = y_track[image_index]

        X_validation_list.append(image)
        y_validation_list.append(label)

        rest_x = np.delete(x_track, image_index, 0)
        #print(rest_x.shape)

        rest_y = np.delete(y_track, image_index, 0)
        #print(rest_y.shape)

        X_train_list.append(rest_x)
        y_train_list.append(rest_y)

    # print(len(X_validation_list))
    # print(X_validation_list[0].shape)
    # print(len(y_validation_list))

    X_train_list = flatten(X_train_list)
    # print(len(X_train_list))

    y_train_list = flatten(y_train_list)
    # print(len(y_train_list))

    #print(X_train_list.shape)
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_validation = np.array(X_validation_list)
    y_validation = np.array(y_validation_list)

    # print(X_train.shape)
    # print(y_train.shape)

    # shuffle the examples
    # X_train, y_train = shuffle(X_train, y_train)
    # X_validation, y_validation = shuffle(X_validation, y_validation)

    return (X_train, X_validation, y_train, y_validation)
