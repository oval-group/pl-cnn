from opt.svm.conditions import break_inner_cccp, break_inner_passes,\
    break_outer_cccp, break_outer_passes, \
    initialize_run, finalize_run

from utils.monitoring import checkpoint


def train_network(nnet):

    initialize_run(nnet)

    for nnet.pass_ in range(nnet.max_passes):
        for layer in reversed(nnet.trainable_layers):
            for _ in range(nnet.max_outer_iterations):
                optimize_layer(nnet, layer)
                if break_outer_cccp(nnet, layer):
                    break
            if break_inner_passes(nnet, layer):
                break
        if break_outer_passes(nnet):
            break

    finalize_run(nnet)


def optimize_layer(nnet,
                   layer):

    layer.initialize_primal(nnet)

    for nnet.it in range(nnet.max_inner_iterations):
        nnet.epoch(nnet, layer)
        checkpoint(nnet, layer)
        if break_inner_cccp(nnet):
            break

    layer.use_average()
    checkpoint(nnet, layer)
