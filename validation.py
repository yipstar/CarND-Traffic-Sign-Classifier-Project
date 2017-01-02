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
    validation_tracks = []
    train_tracks = []

    for i in range(0, groups):
        #print(i)
        #row = X_dataset[i]
        x_track = X_dataset[i*30:i*30+30, :]
        y_track = y_dataset[i*30:i*30+30]

        label = y_dataset[i*30]
        tracks_by_class.setdefault(label, []).append([x_track, y_track])

    # select one track from each class at random and add it to the validation set, add remaining tracks  to the training set

    #print(x_track.shape)
    #print(row.shape)

    for key, value in tracks_by_class.items():
        track_size = len(value)
        # print(track_size)
        random_track_index = random.randint(0, track_size - 1)
        random_track = value[random_track_index]
        # print("random track size")
        # print(random_track[0].shape)
        validation_tracks.append(random_track)

        # remove the randomly selected track leaving the rest of the classes
        # tracks
        value.pop(random_track_index)
        train_tracks.append(value)

    X_validation_list = list(map(lambda x: x[0].tolist(), validation_tracks))
    y_validation_list = list(map(lambda x: x[1].tolist(), validation_tracks))

    X_validation_list = flatten(X_validation_list)
    y_validation_list = flatten(y_validation_list)

    print("X_validation_list size: ", len(X_validation_list))
    print("y_validation_list size: ", len(y_validation_list))

    X_validation = np.array(X_validation_list)
    y_validation = np.array(y_validation_list)

    train_tracks = flatten(train_tracks)

    X_train_list = list(map(lambda x: x[0].tolist(), train_tracks))
    y_train_list = list(map(lambda x: x[1].tolist(), train_tracks))

    X_train_list = flatten(X_train_list)
    y_train_list = flatten(y_train_list)

    print("X_train_list size: ", len(X_train_list))
    print("y_train_list size: ", len(y_train_list))

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    # # shuffle the examples
    X_train, y_train = shuffle(X_train, y_train)
    X_validation, y_validation = shuffle(X_validation, y_validation)

    return (X_train, X_validation, y_train, y_validation)
