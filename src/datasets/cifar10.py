from datasets.base import DataLoader
from datasets.modules import conv_module_cifar
from datasets.preprocessing import center_data, normalize_data

from utils.misc import flush_last_line
from config import Configuration as Cfg

import os
import numpy as np
import cPickle as pickle


class CIFAR_10_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "cifar10"

        self.n_train = 45000
        self.n_val = 5000
        self.n_test = 10000

        self.n_classes = 10

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = "../data/cifar-10-batches-py/"

        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        self.load_data()

    def load_data(self):

        print("Loading data...")

        # load training data
        X, y = [], []
        count = 1
        filename = '%s/data_batch_%i' % (self.data_path, count)
        while os.path.exists(filename):
            with open(filename, 'rb') as f:
                batch = pickle.load(f)
            X.append(batch['data'])
            y.append(batch['labels'])
            count += 1
            filename = '%s/data_batch_%i' % (self.data_path, count)

        # reshape data and cast them properly
        X = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32)
        y = np.concatenate(y).astype(np.int32)

        # load test set
        path = '%s/test_batch' % self.data_path
        with open(path, 'rb') as f:
            batch = pickle.load(f)

        # reshaping and casting for test data
        X_test = batch['data'].reshape(-1, 3, 32, 32).astype(np.float32)
        y_test = np.array(batch['labels'], dtype=np.int32)

        # split into training and validation sets with stored seed
        np.random.seed(self.seed)
        perm = np.random.permutation(len(X))

        self._X_train = X[perm[self.n_val:]]
        self._y_train = y[perm[self.n_val:]]
        self._X_val = X[perm[:self.n_val]]
        self._y_val = y[perm[:self.n_val]]

        self._X_test = X_test
        self._y_test = y_test

        # center data per pixel (mean from X_train)
        center_data(self._X_train, self._X_val, self._X_test,
                    mode="per pixel")

        # normalize data per pixel (std from X_train)
        normalize_data(self._X_train, self._X_val, self._X_test,
                       mode="per pixel")

        flush_last_line()
        print("Data loaded.")

    def build_architecture(self, nnet):

        nnet.addInputLayer(shape=(None, 3, 32, 32))
        conv_module_cifar(nnet, nfilters=64)
        conv_module_cifar(nnet, nfilters=128)
        conv_module_cifar(nnet, nfilters=256)
        nnet.addSVMLayer()

    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu
