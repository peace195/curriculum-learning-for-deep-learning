import numpy as np
from sklearn.cluster import KMeans
import operator
from collections import defaultdict
import pickle
from sklearn.decomposition import PCA

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data_one(f):
    batch = unpickle(f)
    print batch.keys()
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

def grayscale(a):
        return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1) / 256.

def run():
    data_dir = '../../cifar100'
    image_size = 32
    image_dim = image_size * image_size * 3
    meta = unpickle(data_dir + '/meta')
    label_names = meta['fine_label_names']
    label_count = len(label_names)

    train_files = [ 'train' ]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    train_data = grayscale(train_data)
    test_data, test_labels = load_data([ 'test' ], data_dir, label_count)
    test_data = grayscale(test_data)
    print "Train:", np.shape(train_data), np.shape(train_labels)
    print "Test:", np.shape(test_data), np.shape(test_labels)
    data = { 'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels }

    reduced_data = PCA(n_components=2).fit_transform(train_data)
    kmeans = KMeans(n_clusters=100, random_state=0, precompute_distances=True, max_iter=1000, n_init=20).fit(reduced_data)

    cluster_density = dict()
    cluster = defaultdict(list)

    for i in range(len(kmeans.labels_)):
        cluster[kmeans.labels_[i]].append(i)
        if kmeans.labels_[i] in cluster_density:
            cluster_density[kmeans.labels_[i]] = cluster_density[kmeans.labels_[i]] + np.linalg.norm(reduced_data[i] - kmeans.cluster_centers_[kmeans.labels_[i]])
        else:
            cluster_density[kmeans.labels_[i]] = np.linalg.norm(reduced_data[i] - kmeans.cluster_centers_[kmeans.labels_[i]])

    for i in set(kmeans.labels_):
        cluster_density[i] = cluster_density[i]/len(cluster[i])

    curriculum_cluster = sorted(cluster_density.items(), key=operator.itemgetter(1))
    print curriculum_cluster
    cluster_density_sorted = list()
    for tup in curriculum_cluster:
        cluster_density_sorted.append((tup[0], cluster[tup[0]]))
    pickle.dump(cluster_density_sorted, open( "cluster.p", "wb" ))

run()