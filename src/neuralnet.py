import time
import pycrayon
import socket
import os
from lasagne.layers import InputLayer

import opt.sgd.train
import opt.sgd.updates
import opt.svm.train
import layers.fun

from datasets.main import load_dataset
from utils.misc import flush_last_line
from utils.pickle import dump_weights, load_weights
from utils.patches import patch_lasagne
from utils.log import Log
from layers import ConvLayer, ReLU, MaxPool, DenseLayer, SVMLayer, BatchNorm,\
    DropoutLayer
from config import Configuration as Cfg


class NeuralNet:

    def __init__(self,
                 dataset,
                 use_weights=None,
                 profile=False):
        """ initialize instance
        """

        # whether to enable profiling in Theano functions
        self.profile = profile

        # patch lasagne creation of parameters
        # (otherwise broadcasting issue with latest versions)
        patch_lasagne()

        self.initialize_variables(dataset)

        # load dataset
        load_dataset(self, dataset.lower())

        if use_weights:
            # special case for VGG pre-trained network
            if use_weights.endswith('vgg16.pkl'):
                self.data.load_weights(self)
            else:
                self.load_weights(use_weights)

    def initialize_variables(self, dataset):

        self.all_layers, self.trainable_layers = (), ()

        self.n_conv_layers = 0
        self.n_dense_layers = 0
        self.n_relu_layers = 0
        self.n_bn_layers = 0
        self.n_maxpool_layers = 0
        self.n_dropout_layers = 0

        self.it = 0
        self.clock = 0

        self.log = Log(dataset_name=dataset)

        if Cfg.draw_on_board:
            self.xp_on_board(dataset)

        self.dense_layers, self.conv_layers, = [], []

    def xp_on_board(self, dataset):
        """
        Connect to tensorboard server and create pycrayon experiment
        """

        hostname = socket.gethostname()
        hostname = "local" if hostname == Cfg.hostname \
            else hostname.split(".").pop(0)
        client = pycrayon.CrayonClient(hostname=Cfg.hostname)
        count = 0
        while True:
            experiment_name_train = "{}_{}_{}_train"\
                .format(dataset, hostname, count)
            experiment_name_val = "{}_{}_{}_val"\
                .format(dataset, hostname, count)
            try:
                self.board_monitor_train = \
                    client.create_experiment(experiment_name_train)
                self.board_monitor_val = \
                    client.create_experiment(experiment_name_val)
                print("Connected to pycrayon server: experiment '{}_{}_{}'"
                      .format(dataset, hostname, count))
                break
            except:
                count += 1
            # safeguard
            if count > 1000:
                raise ValueError("Failed to connect to server board")

    def compile_updates(self):
        """ create network from architecture given in modules (determined by dataset)
        create Theano compiled functions
        """

        opt.sgd.updates.create_update(self)

        # avoid unnecessary compilation of not using lwsvm
        if self.solver not in ('svm', 'bpfw'):
            return

        if self.solver == 'bpfw':
            self.update_bpfw = opt.bpfw.updates.compile_update_bpfw(self)
            return

        if Cfg.store_on_gpu:
            from opt.svm.full_gpu import compile_update_conv,\
                compile_update_dense, compile_update_svm
        else:
            from opt.svm.part_gpu import compile_update_conv,\
                compile_update_dense, compile_update_svm

        for layer in self.trainable_layers:

            print("Compiling updates for {}...".format(layer.name))

            layer.hinge_avg = layers.fun.compile_hinge_avg(self, layer)

            if layer.isconv:
                layer.svm_update = compile_update_conv(self, layer)

            if layer.isdense:
                layer.svm_update = compile_update_dense(self, layer)

            if layer.issvm:
                layer.svm_update = compile_update_svm(self, layer)

            flush_last_line()

        print("Updates compiled.")

    def load_data(self, data_loader=None):

        self.data = data_loader()
        self.data.build_architecture(self)

        for layer in self.all_layers:
            setattr(self, layer.name + "_layer", layer)

        self.log.store_architecture(self)

    def next_layers(self, layer):

        flag = False
        for current_layer in self.all_layers:
            if flag:
                yield current_layer
            if current_layer is layer:
                flag = True

    def previous_layers(self, layer):

        flag = False
        for current_layer in reversed(self.all_layers):
            if flag:
                yield current_layer
            if current_layer is layer:
                flag = True

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def train(self, solver, n_epochs=10, save_at=0, save_to=''):

        self.solver = solver.lower()
        self.n_epochs = n_epochs
        self.save_at = save_at
        self.save_to = save_to

        self.log['solver'] = self.solver
        self.log['save_at'] = self.save_at

        self.compile_updates()

        if solver == "svm":
            assert Cfg.compile_lwsvm
            from opt.svm.train import train_network
        elif solver == "bpfw":
            from opt.bpfw.train import train_network
        else:
            from opt.sgd.train import train_network

        self.start_clock()
        train_network(self)
        self.stop_clock()

        self.log.save_to_file()

    def addInputLayer(self, **kwargs):

        self.input_layer = InputLayer(name="input", **kwargs)
        self.input_layer.inp_ndim = len(kwargs["shape"])

    def addConvLayer(self, use_batch_norm=False, **kwargs):
        """
        Add convolutional layer.
        If batch norm flag is True, the convolutional layer
        will be followed by a batch-normalization layer
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_conv_layers += 1
        name = "conv%i" % self.n_conv_layers

        new_layer = ConvLayer(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)
        self.trainable_layers += (new_layer,)

        if use_batch_norm:
            self.n_bn_layers += 1
            name = "bn%i" % self.n_bn_layers
            self.all_layers += (BatchNorm(new_layer, name=name),)

    def addDenseLayer(self, use_batch_norm=False, **kwargs):
        """
        Add dense layer.
        If batch norm flag is True, the dense layer
        will be followed by a batch-normalization layer
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_dense_layers += 1
        name = "dense%i" % self.n_dense_layers

        new_layer = DenseLayer(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)
        self.trainable_layers += (new_layer,)

        if use_batch_norm:
            self.n_bn_layers += 1
            name = "bn%i" % self.n_bn_layers
            self.all_layers += (BatchNorm(new_layer, name=name),)

    def addSVMLayer(self, **kwargs):
        """
        Add classification layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]
        new_layer = SVMLayer(input_layer, num_units=self.data.n_classes,
                             **kwargs)

        self.all_layers += (new_layer,)
        self.trainable_layers += (new_layer,)

        self.n_layers = len(self.all_layers)

    def addReLU(self, **kwargs):
        """
        Add ReLU activation layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_relu_layers += 1
        name = "relu%i" % self.n_relu_layers

        new_layer = ReLU(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addMaxPool(self, **kwargs):
        """
        Add MaxPooling activation layer.
        """

        input_layer = self.input_layer if not self.all_layers\
            else self.all_layers[-1]

        self.n_maxpool_layers += 1
        name = "maxpool%i" % self.n_maxpool_layers

        new_layer = MaxPool(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addDropoutLayer(self, **kwargs):
        """
        Add Dropout layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_dropout_layers += 1
        name = "dropout%i" % self.n_dropout_layers

        new_layer = DropoutLayer(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def dump_weights(self, filename=None):

        dump_weights(self, filename)

    def load_weights(self, filename=None):

        assert filename and os.path.exists(filename)

        load_weights(self, filename)
