# Load pickled data
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_dataset, y_dataset = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# TODO: Number of training examples
n_train = X_dataset.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_dataset.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_dataset))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here.

# X_normalized = np.ndarray(X_dataset.shape, dtype=np.float32)
# X_normalized[:, :, :, 0] = X_dataset[:, :, :, 0] / 256
# X_normalized[:, :, :, 1] = X_dataset[:, :, :, 1] / 256
# X_normalized[:, :, :, 2] = X_dataset[:, :, :, 2] / 256
# print(X_normalized[:, :, :, 0])

# X_normalized = X_dataset

# Try normalizing to grayscale
#gray = np.mean(rgb, -1)

# X_normalized[:, :, :, 0] = X_dataset[:, :, :, 0] / 256
from skimage import color

# Grayscale and normalize
# 0.2125 R + 0.7154 G + 0.0721 B
# 0.299 R + 0.587 G + 0.114 B

X_normalized = np.ndarray(X_dataset.shape, dtype=np.float32)
X_normalized[:, :, :, 0] = (((X_dataset[:, :, :, 0] * 0.299) + (X_dataset[:, :, :, 1] * 0.587) + (X_dataset[:, :, :, 2] * 0.299)) / 3) / 256
X_normalized[:, :, :, 1] = (((X_dataset[:, :, :, 0] * 0.299) + (X_dataset[:, :, :, 1] * 0.587) + (X_dataset[:, :, :, 2] * 0.587)) / 3) / 256
X_normalized[:, :, :, 2] = (((X_dataset[:, :, :, 0] * 0.299) + (X_dataset[:, :, :, 1] * 0.587) + (X_dataset[:, :, :, 2] * 0.114)) / 3) / 256

# print("normalized shape")
# print(X_normalized.shape)
# print(X_normalized[:, :, :, 0])

### Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

# Create a validation set

# Use 10% of training examples as validation
# X_train, X_valid, y_train, y_valid = train_test_split(X_normalized, y_dataset,
#                                                       test_size=0.10,
#                                                       random_state=42)

# Create validation set using 1 random image from each track
from validation import train_test_split_by_track

X_train, X_valid, y_train, y_valid = train_test_split_by_track(X_normalized, y_dataset)

print("Number of validation examples = ", X_valid.shape[0])
print(X_train.shape)

print(X_valid.shape)

import tensorflow as tf
from tensorflow.contrib.layers import flatten

# import lenet

from lenet import LeNet

# see the LeNet function above
EPOCHS = 10
BATCH_SIZE = 128

### Train your model here.
### Feel free to use as many code cells as needed.

print(X_train.shape)

x = tf.placeholder(tf.float32, (None, 32, 32, X_train.shape[3]))
y = tf.placeholder(tf.int32, (None))

one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

image_size = 32
image_channels = X_train.shape[3]
# print(image_channels)

# training pipeline
logits = LeNet(x, image_size, image_channels, n_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# evaluation function
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            #print(offset)
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess, 'lenet')
    print("Model saved")

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

