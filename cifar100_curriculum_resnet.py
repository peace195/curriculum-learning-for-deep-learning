import scipy.io
import tensorflow as tf
import os
from pylab import *
import numpy as np
import pickle
from numpy import *
import math

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
batch_size = 64

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


def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):                              
    res = [ 0 ] * len(tensors)                                                                                           
    batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]                    
    total_size = len(batch_tensors[0][1])                                                                                
    batch_count = (total_size + batch_size - 1) / batch_size                                                             
    for batch_idx in xrange(batch_count):                                                                                
        current_batch_size = None                                                                                          
        for (placeholder, tensor) in batch_tensors:                                                                        
            batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                                         
            current_batch_size = len(batch_tensor)                                                                           
            feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                               
        tmp = session.run(tensors, feed_dict=feed_dict)                                                                    
        res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]                                                   
    return [ r / float(total_size) for r in res ]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, stride):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    return tf.nn.conv2d(input, W, [ 1, stride, stride, 1 ], padding='SAME')

def basic_block(input, in_features, out_features, stride, is_training, keep_prob):
    if stride == 1:
        shortcut = input
    else:
        shortcut = tf.nn.avg_pool(input, [ 1, stride, stride, 1 ], [1, stride, stride, 1 ], 'VALID')
        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0],
            [(out_features-in_features)//2, (out_features-in_features)//2]])
    current = conv2d(input, in_features, out_features, 3, stride)
    current = tf.nn.dropout(current, keep_prob)
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = conv2d(current, out_features, out_features, 3, 1)
    current = tf.nn.dropout(current, keep_prob)
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    # No final relu as per http://torch.ch/blog/2016/02/04/resnets.html
    return current + shortcut

def block_stack(input, in_features, out_features, stride, depth, is_training, keep_prob):
    current = basic_block(input, in_features, out_features, stride, is_training, keep_prob)
    for _d in xrange(depth - 1):
        current = basic_block(current, out_features, out_features, 1, is_training, keep_prob)
    return current

graph = tf.Graph()
with graph.as_default():
    xs = tf.placeholder("float", shape=[None, image_dim])
    ys = tf.placeholder("float", shape=[None, label_count])
    lr = tf.placeholder("float", shape=[])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder("bool", shape=[])

    current = tf.reshape(xs, [ -1, 32, 32, 3 ])
    current = conv2d(current, 3, 16, 3, 1)
    current = tf.nn.relu(current)

    # dimension is 32x32x16
    current = block_stack(current, 16, 16, 1, 18, is_training, keep_prob)
    current = block_stack(current, 16, 32, 2, 18, is_training, keep_prob)
    # dimension is 16x16x32
    current = block_stack(current, 32, 64, 2, 18, is_training, keep_prob)
    # dimension is 8x8x64

    current = tf.reduce_mean(current, reduction_indices=[1, 2], name="avg_pool")
    final_dim = 64
    current = tf.reshape(current, [ -1, final_dim ])
    Wfc = weight_variable([ final_dim, label_count ])
    bfc = bias_variable([ label_count ])
    ys_ = tf.nn.softmax( tf.matmul(current, Wfc) + bfc )

    cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
    train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    for i in xrange(0, nb_cluster):
        id = []
        for j in range(i + 1):
            id = id + cluster_density_sorted[nb_cluster-j-1][1]
        xtrain = train_data[id]
        ytrain = train_labels[id]

        pi = np.random.permutation(len(xtrain))
        xtrain, ytrain = xtrain[pi], ytrain[pi]

        if i > 0:
            saver.restore(session, './curriculum_alexnet_cluster' + str(i - 1) + '.ckpt')

        batch_count = len(xtrain) / batch_size
        batches_data = np.split(xtrain[:batch_count*batch_size], batch_count)
        batches_labels = np.split(ytrain[:batch_count*batch_size], batch_count)
        learning_rate = 0.1
        
        if i < nb_cluster - 1:
            nb_epoch = 9
        else:
            nb_epoch = 300

        for epoch in xrange(1, 1+nb_epoch):
            if epoch == math.floor(nb_epoch/3): learning_rate = 0.01
            if epoch == math.floor(2*nb_epoch/3): learning_rate = 0.001
            for batch_idx in xrange(batch_count):
                batch_data = batches_data[batch_idx]
                batch_labels = batches_labels[batch_idx]
            
                batch_res = session.run([ train_step, cross_entropy, accuracy ],
                    feed_dict = { xs: batch_data, ys: batch_labels, lr: learning_rate, is_training: True, keep_prob: 0.8 })

            test_results = run_in_batch_avg(session, [ cross_entropy, accuracy ], [ xs, ys ],
                    feed_dict = { xs: data['test_data'], ys: data['test_labels'], is_training: False, keep_prob: 1. })
            print epoch, batch_res[1:], test_results

        saver.save(session, './curriculum_alexnet_cluster' + str(i) + '.ckpt')