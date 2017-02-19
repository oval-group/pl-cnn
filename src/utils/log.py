import time
import os
import numpy as np

import cPickle as pickle

from config import Configuration


class Log(dict):

    def __init__(self, dataset_name):

        dict.__init__(self)

        self['dataset_name'] = dataset_name

        self['date_and_time'] = time.strftime('%d-%m-%Y--%H-%M-%S')

        self['time_stamp'] = []
        self['layer_tag'] = []

        self['train_objective'] = []
        self['train_accuracy'] = []

        self['val_objective'] = []
        self['val_accuracy'] = []

        self['test_objective'] = -1
        self['test_accuracy'] = -1

        self['primal_objective'] = []
        self['hinge_loss'] = []
        self['dual_objective'] = []

        for key in Configuration.__dict__:
            if key.startswith('__'):
                continue
            if key not in ('C', 'D', 'learning_rate'):
                self[key] = getattr(Configuration, key)
            else:
                self[key] = getattr(Configuration, key).get_value()

    def store_architecture(self, nnet):

        self['layers'] = dict()
        for layer in nnet.all_layers:
            self['layers'][layer.name] = dict()
            if layer.isdense or layer.issvm:
                self['layers'][layer.name]["n_in"] = \
                    np.prod(layer.input_shape[1:])
                self['layers'][layer.name]["n_out"] = layer.num_units

            if layer.isconv:
                self['layers'][layer.name]["n_filters"] = layer.num_filters
                self['layers'][layer.name]["f_size"] = layer.filter_size

            if layer.ismaxpool:
                self['layers'][layer.name]["pool_size"] = layer.pool_size

    def save_to_file(self, filename=None):

        if not filename:
            filename = '../log/all/{}-0'.format(self['date_and_time'])
            count = 1
            while os.path.exists(filename):
                filename = '../log/all/{}-{}'\
                    .format(self['date_and_time'], count)
                count += 1
            filename += '.p'

        pickle.dump(self, open(filename, 'wb'))
        print('Experiment logged in {}'.format(filename))
