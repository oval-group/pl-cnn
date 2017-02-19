import numpy as np
import os
import cPickle as pickle

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

color_set = "Set1"
sns.set(style="white", palette=color_set)
colors = sns.color_palette(color_set)


def plot(xp_dir, export_pdf, show):

    if os.path.exists('{}/log_mnist.txt'.format(xp_dir)):
        dataset = 'mnist'

    elif os.path.exists('{}/log_cifar10.txt'.format(xp_dir)):
        dataset = 'cifar10'

    elif os.path.exists('{}/log_cifar100.txt'.format(xp_dir)):
        dataset = 'cifar100'

    else:
        raise NotImplementedError('Could not find appropriate log file in {}'
                                  .format(xp_dir))

    fig = plt.figure(figsize=(12, 6))

    matplotlib.rcParams.update({'font.size': 13})
    matplotlib.rc('xtick', labelsize=10)
    nb = 0
    for solver in ["adagrad", "adadelta", "adam"]:

        sns.set(style="white", palette=color_set)
        ax1 = fig.add_subplot(231 + nb)
        sns.set(style="white", palette=color_set)
        ax2 = fig.add_subplot(234 + nb)
        nb += 1

        base = pickle.load(open("%s/%s_%s_svm_results.p" %
                                (xp_dir, dataset, solver), "rb"))
        lwsvm = pickle.load(open("%s/%s_%s_svm_lwsvm_results.p" %
                                 (xp_dir, dataset, solver), "rb"))

        start_epoch = base['save_at']

        # join end of SGD and beginning of LWSVM
        lwsvm['time_stamp'] = np.array([0] + lwsvm['time_stamp']) / 3600. + \
            base['time_stamp'][start_epoch] / 3600.
        base['time_stamp'] = np.array(base['time_stamp']) / 3600.

        lwsvm['train_objective'] = [base['train_objective'][start_epoch]] + \
            lwsvm['train_objective']
        lwsvm['train_accuracy'] = [base['train_accuracy'][start_epoch]] + \
            lwsvm['train_accuracy']
        lwsvm['val_accuracy'] = [base['val_accuracy'][start_epoch]] + \
            lwsvm['val_accuracy']

        # find stop index for SGD (index of SGD where LWSVM stops)
        try:
            stop = [i for i in range(len(base['time_stamp']))
                    if base['time_stamp'][i] > lwsvm['time_stamp'][-1]][0]
        except:
            stop = -1
        stop = -1

        # don't display first epochs (scaling reasons)
        start = 0

        train_objective1, = \
            ax1.plot(base['time_stamp'][start: start_epoch],
                     base['train_objective'][start: start_epoch],
                     label="%s" % solver.title())
        train_objective2, = \
            ax1.plot(lwsvm['time_stamp'],
                     lwsvm['train_objective'],
                     label="LW_SVM")
        ax1_handles = [train_objective1, train_objective2]
        ax1.plot(base['time_stamp'][start: stop],
                 base['train_objective'][start: stop],
                 color=colors[0], alpha=0.5)

        train_accuracy1, = \
            ax2.plot(base['time_stamp'][start: start_epoch],
                     base['train_accuracy'][start: start_epoch],
                     label="Training %s" % solver.title())
        train_accuracy2, = \
            ax2.plot(lwsvm['time_stamp'], lwsvm['train_accuracy'],
                     label="Training LW-SVM")
        val_accuracy1, = \
            ax2.plot(base['time_stamp'][start: start_epoch],
                     base['val_accuracy'][start: start_epoch],
                     label="Validation %s" % solver.title())
        val_accuracy2, = \
            ax2.plot(lwsvm['time_stamp'],
                     lwsvm['val_accuracy'],
                     label="Validation LW-SVM")
        ax2_handles = [train_accuracy1, train_accuracy2,
                       val_accuracy1, val_accuracy2]
        ax2.plot(base['time_stamp'][start: stop],
                 base['train_accuracy'][start: stop],
                 color=colors[0], alpha=0.5)
        ax2.plot(base['time_stamp'][start: stop],
                 base['val_accuracy'][start: stop],
                 color=colors[2], alpha=0.5)

        ax1.legend(handles=ax1_handles)
        ax2.legend(handles=ax2_handles, loc=4)

        if dataset == "mnist":
            ax1.set_ylim([0.02, 0.1])
            ax2.set_ylim([96, 100])

        if dataset == "cifar10":
            ax1.set_ylim([0, 0.15])
            ax2.set_ylim([45, 101])

        if dataset == "cifar100":
            ax1.set_ylim([0, 0.4])
            ax2.set_ylim([0, 101])

        if solver == "adagrad":
            ax2.set_ylabel("Accuracy (%)")
            ax1.set_ylabel("Training Objective Function")

        if solver == "adadelta":
            ax2.set_xlabel("Time (h)")

    if export_pdf:
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
