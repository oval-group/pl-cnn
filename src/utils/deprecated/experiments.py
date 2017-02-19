import time
import numpy as np
import os
import cPickle as pickle
import theano

class Experiment:

    def __init__(self, nnet, seed, comment=""):

        self.seed = seed
        self.date_and_time = time.strftime("%d-%m-%Y--%H-%M-%S")

        self.comment = comment

        self.architecture = Architecture(nnet)
        self.dataset = Dataset(nnet)

        for key in ["eps", "floatX", "net_type"]:
            setattr(self, key, getattr(nnet, key))

        for key in ("time_stamp",
                    "layer_tag",
                    "train_objective",
                    "train_accuracy",
                    "val_objective",
                    "val_accuracy",
                    "test_objective",
                    "test_accuracy",
                    "primal_objective",
                    "hinge_loss",
                    "dual_objective"):
            setattr(self, key, getattr(nnet.results, key))

        if "gpu" in theano.config.device:
            print("GPU experiment, saving results automatically")
            self.log()
        else:
            print("CPU experiment, not saving results automatically")

    def log(self, filename=None):

        if filename is None:
            filename = "./experiments/log/" + self.date_and_time + "-0"
            count = 1
            while os.path.exists(filename):
                filename = filename[:-1] + str(count)
                count += 1
            filename += ".p"

        pickle.dump(self, open(filename, "wb"))
        print("Experiment logged in %s" % filename)

class Architecture:

    def __init__(self, nnet):

        self.layers = dict()
        for layer in nnet.layers_list:
            self.layers[layer.name] = dict()
            if layer.isdense or layer.issvm:
                self.layers[layer.name]["n_in"] = np.prod(layer.input_shape[1:])
                self.layers[layer.name]["n_out"] = layer.num_units

            if layer.isconv:
                self.layers[layer.name]["n_filters"] = layer.num_filters
                self.layers[layer.name]["f_size"] = layer.filter_size

            if layer.ismaxpool:
                self.layers[layer.name]["pool_size"] = layer.pool_size

class Dataset:

    def __init__(self, nnet):

        self.name = nnet.dataset

        self.keys = ('learning_rate',
                     'momentum',
                     'n_epochs',
                     "batch_size",
                     "solver",
                     "n_samples",
                     "n_classes")

        for key in self.keys:
            setattr(self, key, getattr(nnet, key))

        self.C = nnet.C.get_value()
        self.D = nnet.D.get_value()

class Schedule:

    def __init__(self, nnet):

        return

class Results:

    def __init__(self):

        self.time_stamp = []
        self.layer_tag = []

        self.train_objective = []
        self.train_accuracy = []

        self.val_objective = []
        self.val_accuracy = []

        self.test_objective = -1
        self.test_accuracy = -1

        self.primal_objective = []
        self.hinge_loss = []
        self.dual_objective = []


