from datasets.base import DataLoader
from datasets.preprocessing import normalize_data

from utils.misc import flush_last_line
from config import Configuration as Cfg

import gzip
import numpy as np


class MNIST_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "mnist"

        self.n_train = 50000
        self.n_val = 10000
        self.n_test = 10000

        self.n_classes = 10

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = "../data/"

        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        self.load_data()

    def load_data(self):

        print("Loading data...")

        X = load_mnist_images('%strain-images-idx3-ubyte.gz' %
                              self.data_path)
        y = load_mnist_labels('%strain-labels-idx1-ubyte.gz' %
                              self.data_path)
        X_test = load_mnist_images('%st10k-images-idx3-ubyte.gz' %
                                   self.data_path)
        y_test = load_mnist_labels('%st10k-labels-idx1-ubyte.gz' %
                                   self.data_path)

        # split into training and validation sets
        np.random.seed(self.seed)
        perm = np.random.permutation(len(X))

        self._X_train = X[perm[self.n_val:]]
        self._y_train = y[perm[self.n_val:]]
        self._X_val = X[perm[:self.n_val]]
        self._y_val = y[perm[:self.n_val]]

        self._X_test = X_test
        self._y_test = y_test

        # normalize data (divide by fixed value)
        normalize_data(self._X_train, self._X_val, self._X_test,
                       scale=np.float32(256))

        flush_last_line()
        print("Data loaded.")

    def build_architecture(self, nnet):

        nnet.addInputLayer(shape=(None, 1, 28, 28))

        nnet.addConvLayer(num_filters=12,
                          filter_size=(5, 5),
                          pad='same')
        nnet.addReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addConvLayer(num_filters=12,
                          filter_size=(5, 5),
                          pad='same')

        nnet.addReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addDenseLayer(num_units=256)
        nnet.addReLU()

        nnet.addSVMLayer()

    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu


def load_mnist_images(filename):

    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    # reshaping and normalizing
    data = data.reshape(-1, 1, 28, 28).astype(np.float32)

    return data


def load_mnist_labels(filename):

    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data
