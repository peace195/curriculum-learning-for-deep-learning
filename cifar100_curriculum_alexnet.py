import scipy.io
import tensorflow as tf
import os
from pylab import *
import numpy as np
import pickle
from numpy import *

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    if 'data' in dict:
        dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3) / 256.
    return dict
  
def load_data_one(f):
    batch = unpickle(f)
    data = batch['data']
    labels = batch['fine_labels']
    print "Loading %s: %d" % (f, len(data))
    return data, labels

def load_data(files, data_dir, label_count):
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([ [ float(i == label) for i in xrange(label_count) ] for label in labels ])
    return data, labels

TRAINING_ITERATIONS = 200000
WEIGHT_DECAY = 0.0001
batch_size = 32
learning_rate = 0.01

data_dir = '/home/binhdt/cifar100'
image_size = 32
image_dim = image_size * image_size * 3
meta = unpickle(data_dir + '/meta')
label_names = meta['fine_label_names']
label_count = len(label_names)

train_data, train_labels = load_data(['train'], data_dir, label_count)
test_data, test_labels = load_data(['test'], data_dir, label_count)
print "Train:", np.shape(train_data), np.shape(train_labels)
print "Test:", np.shape(test_data), np.shape(test_labels)
data = {'train_data': train_data, 'train_labels': train_labels, 'test_data': test_data, 'test_labels': test_labels}
cluster_density_sorted = pickle.load(open("cluster.p", "rb"))
nb_cluster = len(cluster_density_sorted)

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape, name=name)
    return tf.Variable(initial)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

def conv2d(x, W, stride_h, stride_w, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride_h, stride_w, 1], padding=padding)

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder('float', shape=[None, image_dim])
    y_ = tf.placeholder('float', shape=[None, label_count])
    lr = tf.placeholder("float", shape=[])

    conv1W = weight_variable([3, 3, 3, 64], 'conv1W')
    conv1b = bias_variable([64], 'conv1b')
    conv2W = weight_variable([5, 5, 64, 192], 'conv2W')
    conv2b = bias_variable([192], 'conv2b')
    conv3W = weight_variable([3, 3, 192, 256], 'conv3W')
    conv3b = bias_variable([256], 'conv3b')
    fc8W = weight_variable([1 * 1 * 256, label_count], 'fc8W')
    fc8b = bias_variable([label_count], 'fc8b')
    keep_prob = tf.placeholder('float')

    def model(x):
        k_h = 3; k_w = 3; c_o = 64; s_h = 4; s_w = 4; group = 1
        conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv1 = tf.nn.relu(conv1_in)
        radius = 5; alpha = 0.0001; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        k_h = 5; k_w = 5; c_o = 192; s_h = 1; s_w = 1; group = 1
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        maxpool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 1
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        maxpool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        fc7_drop = tf.nn.dropout(maxpool3, keep_prob)
        print_activations(fc7_drop)
        fc8 = tf.nn.xw_plus_b(tf.reshape(fc7_drop, [-1, int(prod(fc7_drop.get_shape()[1:]))]), fc8W, fc8b)
        return fc8

    logits = model(tf.reshape(x, [ -1, 32, 32, 3 ]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    regularizers =  tf.nn.l2_loss(conv1W) + tf.nn.l2_loss(conv1b) +\
                    tf.nn.l2_loss(conv2W) + tf.nn.l2_loss(conv2b) +\
                    tf.nn.l2_loss(conv3W) + tf.nn.l2_loss(conv3b) +\
                    tf.nn.l2_loss(fc8W) + tf.nn.l2_loss(fc8b)
    loss = tf.reduce_mean(cross_entropy + WEIGHT_DECAY * regularizers)

    train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits),1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(nb_cluster):
        id = []
        for j in range(i + 1):
            id = id + cluster_density_sorted[j][1]
        xtrain = train_data[id]
        ytrain = train_labels[id]

        pi = np.random.permutation(len(xtrain))
        xtrain, ytrain = xtrain[pi], ytrain[pi]

        if i > 0:
            saver.restore(sess, './curriculum_alexnet_cluster' + str(i - 1) + '.ckpt')

        for it in range(TRAINING_ITERATIONS):
            if it == TRAINING_ITERATIONS * 50/100: learning_rate = 0.001
            if it == TRAINING_ITERATIONS * 75/100: learning_rate = 0.0001
            if it * batch_size % len(xtrain) + batch_size <= len(xtrain):
                start = it * batch_size % len(xtrain)
                end = start + batch_size
            else:
                start = it * batch_size % len(xtrain)
                end = len(xtrain)

            _, train_accuracy, cost = sess.run([train_step, accuracy, cross_entropy],
                feed_dict={x: xtrain[start:end], y_: ytrain[start:end], keep_prob: 0.5, lr: learning_rate})

            if it % 200 == 0:
                print i, it, train_accuracy, cost, accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})

        saver.save(sess, './curriculum_alexnet_cluster' + str(i) + '.ckpt')

sess.close()