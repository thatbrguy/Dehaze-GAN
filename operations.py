import tensorflow as tf

def conv(input_, out_channels, stride): # deprecated

    with tf.variable_scope('Conv') as scope:
        input__channels = input_.get_shape()[-1]
        h = tf.get_variable(name = 'filter', shape = [4, 4, input__channels, out_channels], dtype = tf.float32, initializer = tf.keras.initializers.he_normal())
        padded_input = tf.pad(input_, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, h, [1, stride, stride, 1], padding="VALID")
        return conv

def batchnorm(input): # deprecated

    with tf.variable_scope("BatchNorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def linear(input_, output_size, name = 'linear'): # deprecated
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable("w_"+name, shape = [shape[1], output_size] , dtype = tf.float32,
                            initializer = tf.keras.initializers.he_normal())
        bias = tf.get_variable("b_"+name, [output_size], initializer=tf.constant_initializer(0.0))
        
        return tf.matmul(input_,w) + bias

def Conv(input_, kernel_size, stride, output_channels, padding = 'SAME', mode = None):

    with tf.variable_scope("Conv") as scope:

        input_channels = input_.get_shape()[-1]
        kernel_shape = [kernel_size, kernel_size, input_channels, output_channels]

        kernel = tf.get_variable("Filter", shape = kernel_shape, dtype = tf.float32, initializer = tf.keras.initializers.he_normal())
        
        # Patchwise Discriminator (PatchGAN) requires some modifications.
        if mode == 'discriminator':
            input_ = tf.pad(input_, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")

        return tf.nn.conv2d(input_, kernel, strides = [1, stride, stride, 1], padding = padding)

def TransposeConv(input_, output_channels, kernel_size = 4):

    with tf.variable_scope("TransposeConv") as scope:

        input_height, input_width, input_channels = [int(d) for d in input_.get_shape()[1:]]
        batch_size = tf.shape(input_)[0] 

        kernel_shape = [kernel_size, kernel_size, output_channels, input_channels]
        output_shape = tf.stack([batch_size, input_height*2, input_width*2, output_channels])

        kernel = tf.get_variable(name = "filter", shape = kernel_shape, dtype=tf.float32, initializer = tf.keras.initializers.he_normal())
        
        return tf.nn.conv2d_transpose(input_, kernel, output_shape, [1, 2, 2, 1], padding="SAME")

def MaxPool(input_):
    with tf.variable_scope("MaxPool"):
        return tf.nn.max_pool(input_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def AvgPool(input_, k = 2):
    with tf.variable_scope("AvgPool"):
        return tf.nn.avg_pool(input_, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def ReLU(input_):
    with tf.variable_scope("ReLU"):
        return tf.nn.relu(input_)

def LeakyReLU(input_, leak = 0.2):
    with tf.variable_scope("LeakyReLU"):
        return tf.maximum(input_, leak * input_)

def BatchNorm(input_, isTrain, name='BN', decay = 0.99):
    with tf.variable_scope(name) as scope:
        return tf.contrib.layers.batch_norm(input_, is_training = isTrain, decay = decay)

def DropOut(input_, isTrain, rate=0.2, name='drop') :
    with tf.variable_scope(name) as scope:
        return tf.layers.dropout(inputs=input_, rate=rate, training=isTrain)
