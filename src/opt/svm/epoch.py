import sys
import numpy as np
from tqdm import tqdm

from utils.monitoring import check_dual
from utils.misc import flush_last_line
from config import Configuration as Cfg


def epoch_full_gpu(nnet, layer):

    # remember dual value before epoch
    old_f = layer.get_dual()

    for batch in tqdm(nnet.data.get_epoch_train(),
                      total=Cfg.n_batches,
                      desc='Current Epoch', leave=False):

        # unpack batch arguments
        X, y, idx = batch

        args = (idx, X, y) if layer.isconv else (idx, y)

        layer.svm_update(*args)

    # stop if not enough improvement
    new_f = layer.get_dual()
    nnet.stop_inner_cccp_flag = new_f < (1. + 1e-2) * old_f


def epoch_part_gpu(nnet, layer):

    # remember dual value before epoch
    old_f = layer.get_dual()

    for batch in tqdm(nnet.data.get_epoch_train(),
                      total=Cfg.n_batches,
                      desc='Current Epoch', leave=False):

        # unpack batch arguments
        X, y, idx = batch

        # transfer primal variable to GPU
        host_to_gpu(layer, idx)

        # perform update
        layer.svm_update(X, y)

        # transfer back updated primal variable from GPU
        gpu_to_host(layer, idx)

    # stop if not enough improvement
    new_f = layer.get_dual()
    nnet.stop_inner_cccp_flag = new_f < (1. + 1e-2) * old_f


def epoch_imagenet(nnet, layer):

    n_batches = Cfg.n_batches
    count = 0

    for batch in nnet.data.get_epoch_train():

        # unpack batch arguments
        X, y, idx = batch

        # transfer primal variable to gpu
        host_to_gpu(layer, idx)

        # perform BCFW update
        layer.svm_update(X, y)

        # transfer primal variable back to host
        gpu_to_host(layer, idx)

        count += 1
        if count % 50 == 0:
            check_dual(nnet, layer)

            if len(nnet.log['dual_objective']) > 0:
                dual_obj = nnet.log['dual_objective'][-1]
                flush_last_line()
                print("Dual objective: %g (it %i out of %i)" %
                      (dual_obj, count, n_batches))


def test_epoch_on_each_layers(nnet):

    for layer in nnet.trainable_layers:
        if layer.isconv:
            continue

        layer.initialize_primal(nnet)
        count = 0
        for batch in nnet.data.get_epoch_train():

            # unpack batch arguments
            X, y, idx = batch

            # transfer primal variable to gpu
            host_to_gpu(layer, idx)

            # perform BCFW update
            layer.svm_update(X, y)

            # transfer primal variable back to host
            gpu_to_host(layer, idx)

            # only five it
            count += 1
            flush_last_line()
            if count == 5:
                print("OK on layer %s." % layer.name)
                break

        layer.use_average()

    count = 0
    for batch in nnet.data.get_epoch_val():
        print("Validation batch %i" % count)
        inputs, targets, _ = batch
        err, acc = nnet.lasagne_val_fn(inputs, targets)

        count += 1
        flush_last_line()
        if count == 5:
            print("OK for validation.")
            break

    print("All good.")

    sys.exit()


def host_to_gpu(layer, idx):

    layer.W_i_buffer.set_value(np.squeeze(layer.W_i[idx]))
    layer.b_i_buffer.set_value(np.squeeze(layer.b_i[idx]))
    layer.l_i_buffer.set_value(np.squeeze(layer.l_i[idx]))


def gpu_to_host(layer, idx):

    layer.W_i[idx] = layer.W_i_buffer.get_value()
    layer.b_i[idx] = layer.b_i_buffer.get_value()
    layer.l_i[idx] = layer.l_i_buffer.get_value()
