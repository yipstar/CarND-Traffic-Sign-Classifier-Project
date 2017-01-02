from PIL import Image
import numpy as np
import os, os.path

def get_test_data():

    X_test_images = []
    y_test_images = []

    for f in os.listdir("test_images"):
        parts = os.path.splitext(f)
        label = parts[0].split("_")[1]

        image = Image.open("test_images/{0}".format(f))
        data = np.asarray(image, dtype="int32" )

        y_test_images.append(label)
        X_test_images.append(data)

        print(y_test_images)
        print(X_test_images)

    return (X_test_images, y_test_images)
