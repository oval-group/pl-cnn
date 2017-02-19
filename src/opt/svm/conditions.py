"""
Declare in this file the conditions of the optimization.
There are four conditions to input:
- break_outer_passes:
    when to stop performing passes over the layers (from end to start)
- break_inner_passes:
    when to stop a pass over the layers
- break_outer_cccp:
    when to stop performing iterations of the CCCP on a layer
- break inner_cccp:
    when to stop solving the convex inner problem of the CCCP on a layer

On top of these, one can declare initial and final conditions in:
- initialize_run
- finalize_run
"""

import numpy as np
import time

from config import Configuration as Cfg
from opt.svm.epoch import epoch_imagenet, epoch_full_gpu, epoch_part_gpu

from utils.monitoring import performance


def initialize_run(nnet):

    if nnet.data.dataset_name == 'imagenet':
        nnet.max_passes = 1
        nnet.max_inner_iterations = 5
        nnet.max_outer_iterations = 1
        nnet.epoch = epoch_imagenet

    elif Cfg.store_on_gpu:
        nnet.max_passes = 50
        nnet.max_inner_iterations = 100
        nnet.max_outer_iterations = 1
        nnet.epoch = epoch_full_gpu

        nnet.old_objective = np.infty
        nnet.old_validation_acc = 0.

        performance(nnet, which_set='train', print_=True)

    else:
        nnet.max_passes = 50
        nnet.max_inner_iterations = 100
        nnet.max_outer_iterations = 1
        nnet.epoch = epoch_part_gpu

        nnet.old_objective = np.infty
        nnet.old_validation_acc = 0.

        performance(nnet, which_set='train', print_=True)

    return


def break_outer_passes(nnet):

    # stop when validation accuracy does not improve anymore
    new_validation_acc = nnet.log['val_accuracy'][-1]

    if new_validation_acc < nnet.old_validation_acc:
        return True
    else:
        nnet.old_validation_acc = new_validation_acc
        return False


def break_inner_passes(nnet, layer):

    # stop if reaching conv layers on ImageNet, otherwise perform full pass
    if nnet.data.dataset_name == 'imagenet' and layer.isconv:
        return True
    else:
        return False


def break_inner_cccp(nnet):

    if nnet.data.dataset_name == "imagenet":
        nnet.dump_weights("imagenet_lwsvm_{}".format(nnet.it))
        return False

    # the stop is fixed by a flag computed inside the epoch function
    return nnet.stop_inner_cccp_flag


def break_outer_cccp(nnet, layer):

    # only one iteration of CCCP on each layer
    return True


def finalize_run(nnet):

    # nothing to do on ImageNet
    if nnet.data.dataset_name == 'imagenet':
        return

    # otherwise compute final performance on test data set
    test_objective, test_accuracy = performance(nnet, which_set='test',
                                                print_=True)

    # log final performance
    nnet.log['test_objective'] = test_objective
    nnet.log['test_accuracy'] = test_accuracy
    nnet.test_time = time.time() - nnet.clock
