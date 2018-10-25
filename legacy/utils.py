import tensorflow as tf

def conv(ip, out_channels, stride):

	with tf.variable_scope('Conv') as scope:
		ip_channels = ip.get_shape()[-1]
		h = tf.get_variable(name = 'filter', shape = [4, 4, ip_channels, out_channels], dtype = tf.float32, initializer = tf.keras.initializers.he_normal())
		padded_input = tf.pad(ip, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
		conv = tf.nn.conv2d(padded_input, h, [1, stride, stride, 1], padding="VALID")
		return conv

def deconv(ip, out_channels, filters = 4):

	with tf.variable_scope("DeConv") as scope:
		in_height, in_width, in_channels = [int(d) for d in ip.get_shape()[1:]]
		batch_size = tf.shape(ip)[0] 
		output_shape = tf.stack([batch_size, in_height*2, in_width*2, out_channels])
		h = tf.get_variable(name = "filter", shape = [filters, filters, out_channels, in_channels], dtype=tf.float32, initializer = tf.keras.initializers.he_normal())
		conv = tf.nn.conv2d_transpose(ip, h, output_shape, [1, 2, 2, 1], padding="SAME")
		return conv

def lrelu(ip, leak = 0.2):
	with tf.name_scope("LeakyRelu"):
		return tf.maximum(ip, leak*ip)

def batchnorm(input):
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

def linear(ip, output_size, name = 'linear'):
    shape = ip.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable("w_"+name, shape = [shape[1], output_size] , dtype = tf.float32,
                            initializer = tf.keras.initializers.he_normal())
        bias = tf.get_variable("b_"+name, [output_size], initializer=tf.constant_initializer(0.0))
        
        return tf.matmul(ip,w) + bias

def Conv(ip, filter, stride, output_ch, padding = 'SAME'):
	input_ch = ip.get_shape()[3]
	h = tf.get_variable("Filter", shape = [filter, filter, input_ch, output_ch], dtype = tf.float32, initializer = tf.keras.initializers.he_normal())
	b = tf.get_variable("Bias", shape = [output_ch], dtype = tf.float32, initializer = tf.constant_initializer(0))
	return tf.nn.conv2d(ip, h, strides = [1, stride, stride, 1], padding = padding)

def MaxPool(ip):
	return tf.nn.max_pool(ip, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def AvgPool(ip, k=2):
	return tf.nn.avg_pool(ip, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def Relu(ip):
	return tf.nn.relu(ip)

def BatchNorm(ip, isTrain, decay = 0.99):
	return tf.contrib.layers.batch_norm(ip, is_training = isTrain, decay = decay)

def DropOut(x, rate = 0.2) :
    return tf.nn.dropout(x, keep_prob=(1-rate))