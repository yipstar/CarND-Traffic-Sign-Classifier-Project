#from run import evaluate

from test_images import get_test_data
import tensorflow as tf

X_my_test, y_my_test = get_test_data()

# evaluation function
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
# accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

image_size = 32
image_channels = 3
n_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))

from lenet import LeNet
logits = LeNet(x, image_size, image_channels, n_classes)

with tf.Session() as sess:
    loader = tf.train.import_meta_graph('lenet.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./'))

    predictions = sess.run(tf.argmax(logits, 1), feed_dict={x: X_my_test, y: y_my_test})

    print(predictions)

    # test_accuracy = evaluate(X_my_test, y_my_test)
    # print("Test Accuracy = {:.3f}".format(test_accuracy))
