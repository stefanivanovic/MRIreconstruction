from helper import *

def add_con2d_weight_bias(w_shape, b_shape, order_no):
    with tf.variable_scope("one", reuse=tf.AUTO_REUSE):
        Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]


def ista_block(input_layers, input_data, layer_no, PhiTPhi, PhiTb):
    tau_value = tf.Variable(0.1, dtype=tf.float32)
    lambda_step = tf.constant(0.1, dtype=tf.float32)
    soft_thr = tf.Variable(0.1, dtype=tf.float32)
    conv_size = 32
    filter_size = 3

    x1_ista = tf.add(input_layers[-1] - tf.scalar_mul(lambda_step, tf.matmul(input_layers[-1], PhiTPhi)), tf.scalar_mul(lambda_step, PhiTb))  # X_k - lambda*A^TAX

    x2_ista = tf.reshape(x1_ista, shape=[-1, 28, 28, 1])

    [Weights0, bias0] = add_con2d_weight_bias([filter_size, filter_size, 1, conv_size], [conv_size], 0)

    [Weights1, bias1] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 1)
    [Weights11, bias11] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 11)

    [Weights2, bias2] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 2)
    [Weights22, bias22] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 22)

    [Weights3, bias3] = add_con2d_weight_bias([filter_size, filter_size, conv_size, 1], [1], 3)

    x3_ista = tf.nn.conv2d(x2_ista, Weights0, strides=[1, 1, 1, 1], padding='SAME')

    x4_ista = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x44_ista = tf.nn.conv2d(x4_ista, Weights11, strides=[1, 1, 1, 1], padding='SAME')

    x5_ista = tf.multiply(tf.sign(x44_ista), tf.nn.relu(tf.abs(x44_ista) - soft_thr))

    x6_ista = tf.nn.relu(tf.nn.conv2d(x5_ista, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x66_ista = tf.nn.conv2d(x6_ista, Weights22, strides=[1, 1, 1, 1], padding='SAME')

    x7_ista = tf.nn.conv2d(x66_ista, Weights3, strides=[1, 1, 1, 1], padding='SAME')

    x7_ista = x7_ista + x2_ista

    x8_ista = tf.reshape(x7_ista, shape=[-1, 784])

    x3_ista_sym = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x4_ista_sym = tf.nn.conv2d(x3_ista_sym, Weights11, strides=[1, 1, 1, 1], padding='SAME')
    x6_ista_sym = tf.nn.relu(tf.nn.conv2d(x4_ista_sym, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x7_ista_sym = tf.nn.conv2d(x6_ista_sym, Weights22, strides=[1, 1, 1, 1], padding='SAME')

    x11_ista = x7_ista_sym - x3_ista

    return [x8_ista, x11_ista]


def inference_ista(input_tensor, n, X_output, reuse, PhiTPhi, PhiTb):
    layers = []
    layers_symetric = []
    layers.append(input_tensor)
    for i in range(n):
        with tf.variable_scope('conv_%d' %i, reuse=reuse):
            [conv1, conv1_sym] = ista_block(layers, X_output, i, PhiTPhi, PhiTb)
            layers.append(conv1)
            layers_symetric.append(conv1_sym)
    return [layers, layers_symetric]


def compute_cost(Prediction, X_output, PhaseNumber, Pre_symetric):
    cost = tf.reduce_mean(tf.square(Prediction[-1] - X_output))
    cost_sym = 0
    for k in range(PhaseNumber):
        cost_sym += tf.reduce_mean(tf.square(Pre_symetric[k]))
    return [cost, cost_sym]



#clean_images, Training_inputs, Phi_input, measurement_matrix
#clean_images is a [-1x784 vector]
# Training_inputs: is clean_images X phi_input
# Phi_input: []
# PhiTPhi
# PHiTb

x_train, y_train, x_test, y_test = load_data()


def make_matrices(clean_images, Phi_input, Training_inputs):
    # Computing Initialization Matrix
    XX = clean_images.transpose()
    BBB = Training_inputs.T.dot(Training_inputs)
    CCC = XX.dot(Training_inputs)
    PhiT_ = CCC.dot(np.linalg.pinv(BBB))
    PhiInv_input = np.conj(PhiT_.transpose())
    PhiTPhi_input = np.dot(Phi_input, Phi_input.transpose())
    Phi = tf.constant(Phi_input, dtype=tf.float32)
    PhiTPhi = tf.constant(PhiTPhi_input, dtype=tf.float32)
    PhiInv = tf.constant(PhiInv_input, dtype=tf.float32)
    X_input = tf.placeholder(tf.float32, [None, Phi_input.shape[1]])
    X_output = tf.placeholder(tf.float32, [None, n_output])
    X0 = tf.matmul(X_input, PhiInv)
    PhiTb = tf.matmul(X_input, tf.transpose(Phi))
    return X0, X_input, X_output, PhiTPhi, PhiTb
