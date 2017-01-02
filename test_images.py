import cv
import numpy as np

x = cv.LoadImageM('test_images/children_28.jpg')
im = np.asarray(x)

print(im.shape)
