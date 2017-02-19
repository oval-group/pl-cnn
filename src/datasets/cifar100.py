from datasets.base import DataLoader
from datasets.modules import conv_module_cifar
from datasets.preprocessing import center_data, normalize_data

from config import Configuration as Cfg
from utils.misc import flush_last_line

import numpy as np
import cPickle as pickle


class CIFAR_100_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "cifar100"

        self.n_train = 45000
        self.n_val = 5000
        self.n_test = 10000

        self.n_classes = 100

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = "../data/cifar-100-python/"

        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        self.load_data()

    def load_data(self):

        print("Loading data...")

        train_file = self.data_path + 'train'
        test_file = self.data_path + 'test'

        train_dict = pickle.load(open(train_file, 'rb'))
        test_dict = pickle.load(open(test_file, 'rb'))

        X = train_dict["data"]
        X = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32)
        y = np.array(train_dict["fine_labels"], dtype=np.int32)

        X_test = test_dict["data"]
        y_test = np.array(test_dict["fine_labels"], dtype=np.int32)
        X_test = X_test.reshape(-1, 3, 32, 32).astype(np.float32)

        # split into training and validation sets
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
