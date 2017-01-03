from run import evaluate
from test_images import get_test_data
import tensorflow as tf

X_my_test, y_my_test = get_test_data()

with tf.Session() as sess:
    loader = tf.train.import_meta_graph('lenet.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./'))

    test_accuracy = evaluate(X_my_test, y_my_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
