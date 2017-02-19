import os
import cPickle as pickle

from neuralnet import NeuralNet
from utils.monitoring import performance
import opt.sgd.updates
from config import Configuration as Cfg


def dump_results(xp_dir, out_file):

    results = dict()

    if os.path.exists('{}/log_mnist.txt'.format(xp_dir)):
        dataset = 'mnist'

    elif os.path.exists('{}/log_cifar10.txt'.format(xp_dir)):
        dataset = 'cifar10'

    elif os.path.exists('{}/log_cifar100.txt'.format(xp_dir)):
        dataset = 'cifar100'

    else:
        raise NotImplementedError('Could not find appropriate log file in {}'
                                  .format(xp_dir))

    results['dataset'] = dataset

    for base_solver in ["adagrad", "adadelta", "adam"]:

        lwsvm_solver = "{}_lwsvm".format(base_solver)
        full_solver = "{}_full".format(base_solver)

        # base solver: baseline
        results[base_solver] = dict()

        # full solver: baseline trained for longer to check
        # training was not stopped prematurely
        results[full_solver] = dict()

        # lwsvm solver: lwsvm applied after base solver
        results[lwsvm_solver] = dict()

        # unpickle results dumped from experiments
        base = pickle.load(open("{}/{}_{}_svm_results.p"
                                .format(xp_dir, dataset, base_solver), "rb"))
        lwsvm = pickle.load(open("{}/{}_{}_svm_lwsvm_results.p"
                                 .format(xp_dir, dataset, base_solver), "rb"))

        # performance of full solver
        results[full_solver]["train_objective"] = base['train_objective'][-1]
        results[full_solver]["train_accuracy"] = base['train_accuracy'][-1]
        results[full_solver]["test_accuracy"] = base['test_accuracy']
        results[full_solver]["n_epochs"] = len(base['time_stamp'])
        results[full_solver]["time"] = base['time_stamp'][-1]

        # performance of lwsvm solver
        results[lwsvm_solver]["train_objective"] = lwsvm['train_objective'][-1]
        results[lwsvm_solver]["train_accuracy"] = lwsvm['train_accuracy'][-1]
        results[lwsvm_solver]["test_accuracy"] = lwsvm['test_accuracy']
        results[lwsvm_solver]["time"] = lwsvm['time_stamp'][-1]

        # compute performance of base solver based on saved weights
        use_weights = "{}/{}_{}_svm_weights.p"\
            .format(xp_dir, dataset, base_solver)
        nnet = NeuralNet(dataset=dataset, use_weights=use_weights)
        Cfg.C.set_value(base['C'])
        Cfg.softmax_loss = False

        opt.sgd.updates.create_update(nnet)
        train_obj, train_acc = performance(nnet, which_set='train')
        _, test_acc = performance(nnet, which_set='test')

        # number of epochs of pre-training
        n_epochs = base['save_at']

        results[base_solver]["n_epochs"] = n_epochs
        results[base_solver]["time"] = base['time_stamp'][n_epochs]
        results[base_solver]["train_objective"] = train_obj
        results[base_solver]["train_accuracy"] = train_acc
        results[base_solver]["test_accuracy"] = test_acc

    pickle.dump(results, open(out_file, "wb"))
