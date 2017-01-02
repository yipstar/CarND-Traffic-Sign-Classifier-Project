from tensorflow.contrib.layers import flatten
import tensorflow as tf

# TODO: use mu and sigma when calling truncated_normal

def LeNet(x, image_size, image_channels, n_classes):
    # Hyperparameters
    mu = 0
    sigma = 0.01

    print(x)

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # [height, width, input_depth, output_depth]

    # out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    # out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

    # out_height = ceil(float(32 - 5 + 1) / float(1)) = 28
    # out_width = ceil(float(32 - 5 + 1) / float(1)) = 28

    conv1_W = tf.Variable(tf.truncated_normal([5, 5, image_channels, 6], mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))

    strides = 1
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, conv1_b)

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    print(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    k = 2
    conv1 = tf.nn.max_pool(conv1,
                           ksize=[1, k, k, 1],
                           strides=[1, k, k, 1],
                           padding='VALID')

    print(conv1)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))

    strides = 1
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, strides, strides, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, conv2_b)

    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    print(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    k = 2
    conv2 = tf.nn.max_pool(conv2,
                           ksize=[1, k, k, 1],
                           strides=[1, k, k, 1],
                           padding='SAME')

    print(conv2)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flattened_conv2 =  tf.contrib.layers.flatten(conv2)
    print(flattened_conv2)

    # TODO use this method
    # fc1_shape = (fc1.get_shape().as_list()[-1], 120)

    # NOTE: You might have set fc1_shape to (400, 120) manually. That would be fine since that's the correct output shape. However, at some point you might want to alter a previous layer, which might change the output shape of Pooling Layer 2. In that case, hard-coding the shape of Fully Connected Layer 1 to (400, 120) would be incorrect. So it's more robust to use the formula above to set the shape.


    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal([5*5*16, 120], mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.add(tf.matmul(flattened_conv2, fc1_W), fc1_b)

    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal([120, 84]))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1, fc2_W), fc2_b)

    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal([84, n_classes]))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.add(tf.matmul(fc2, fc3_W), fc3_b)

    return logits
