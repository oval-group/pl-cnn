import theano.tensor as T

from config import Configuration as Cfg


def _update_cps(nnet, layer, X, dW, db, loss, idx=None):
    """
    update with compressed feature vectors
    """

    assert layer.isdense or layer.issvm

    if Cfg.store_on_gpu:
        assert idx is not None

    C = Cfg.C
    D = Cfg.D
    eps = Cfg.eps

    k = layer.k

    K = (C * D) / (C + D)

    W_s = dW * K * T.cast(1. / nnet.data.n_train, 'floatX')
    b_s = db * K * T.cast(1. / nnet.data.n_train, 'floatX')
    l_s = loss * T.cast(1. / nnet.data.n_train, 'floatX')

    if Cfg.store_on_gpu:
        Deltaw = W_s - layer.W_i[idx]
        Deltab = b_s - layer.b_i[idx]
        Deltal = l_s - layer.l_i[idx]
    else:
        Deltaw = W_s - layer.W_i_buffer
        Deltab = b_s - layer.b_i_buffer
        Deltal = l_s - layer.l_i_buffer

    # uncompress feature vectors and sum over mini-batch

    # Method 1: memory inefficient (full allocation before sum)
    # DeltaW = T.sum(T.shape_padaxis(X, 2) *
    #                T.shape_padaxis(Deltaw, 1), axis=0)

    # Method 2: same result but accumulates
    # results inplace on first dimension
    dummy = T.dot(X, layer.W)
    DeltaW = T.grad(cost=None,
                    wrt=layer.W,
                    known_grads={dummy: Deltaw})

    gamma = (K * Deltal +
             T.sum(DeltaW * layer.W) +
             T.sum(Deltab * layer.b)) / \
        (eps + T.sum(DeltaW ** 2) + T.sum(Deltab ** 2))

    gamma = gamma.clip(0, 1)

    W = layer.W - gamma * DeltaW
    b = layer.b - gamma * Deltab
    l = layer.l + gamma * Deltal

    if Cfg.store_on_gpu:
        # new value to assign
        W_i = T.inc_subtensor(layer.W_i[idx], gamma * Deltaw)
        b_i = T.inc_subtensor(layer.b_i[idx], gamma * Deltab)
        l_i = T.inc_subtensor(layer.l_i[idx], gamma * Deltal)

        # shared variable to update
        layer_W_i = layer.W_i
        layer_b_i = layer.b_i
        layer_l_i = layer.l_i
    else:
        # new value to assign
        W_i = layer.W_i_buffer + gamma * Deltaw
        b_i = layer.b_i_buffer + gamma * Deltab
        l_i = layer.l_i_buffer + gamma * Deltal

        # shared variable to update
        layer_W_i = layer.W_i_buffer
        layer_b_i = layer.b_i_buffer
        layer_l_i = layer.l_i_buffer

    # average
    W_avg = T.cast((k * 1. / (k + 2)), 'floatX') * layer.W_avg + \
        T.cast((2. / (k + 2)), 'floatX') * W
    b_avg = T.cast((k * 1. / (k + 2)), 'floatX') * layer.b_avg + \
        T.cast((2. / (k + 2)), 'floatX') * b
    k = k + 1

    updates = ((layer.W, W),
               (layer.b, b),
               (layer.W_avg, W_avg),
               (layer.b_avg, b_avg),
               (layer.k, k),
               (layer.l, l),
               (layer_W_i, W_i),
               (layer_b_i, b_i),
               (layer_l_i, l_i),
               (layer.gamma, gamma))

    return updates


def _update_std(nnet, layer, dW, db, loss, idx=None):
    """
    update with standard feature vectors (i.e. non-compressed)
    """

    assert layer.isconv or layer.issvm

    if Cfg.store_on_gpu:
        assert idx is not None

    C = Cfg.C
    D = Cfg.D
    eps = Cfg.eps

    k = layer.k

    K = (C * D) / (C + D)

    W_s = dW * K * T.cast(1. / nnet.data.n_train, 'floatX')
    b_s = db * K * T.cast(1. / nnet.data.n_train, 'floatX')
    l_s = loss * T.cast(1. / nnet.data.n_train, 'floatX')

    if Cfg.store_on_gpu:
        DeltaW = W_s - layer.W_i[idx]
        Deltab = b_s - layer.b_i[idx]
        Deltal = l_s - layer.l_i[idx]
    else:
        DeltaW = W_s - layer.W_i_buffer
        Deltab = b_s - layer.b_i_buffer
        Deltal = l_s - layer.l_i_buffer

    gamma = (K * Deltal +
             T.sum(DeltaW * layer.W) +
             T.sum(Deltab * layer.b)) / \
        (eps + T.sum(DeltaW ** 2) + T.sum(Deltab ** 2))

    gamma = gamma.clip(0, 1)

    W = layer.W - gamma * DeltaW
    b = layer.b - gamma * Deltab
    l = layer.l + gamma * Deltal

    if Cfg.store_on_gpu:
        # new value
        W_i = T.inc_subtensor(layer.W_i[idx], gamma * DeltaW)
        b_i = T.inc_subtensor(layer.b_i[idx], gamma * Deltab)
        l_i = T.inc_subtensor(layer.l_i[idx], gamma * Deltal)

        # shared variable to update
        layer_W_i = layer.W_i
        layer_b_i = layer.b_i
        layer_l_i = layer.l_i
    else:
        # new value
        W_i = layer.W_i_buffer + gamma * DeltaW
        b_i = layer.b_i_buffer + gamma * Deltab
        l_i = layer.l_i_buffer + gamma * Deltal

        # shared variable to update
        layer_W_i = layer.W_i_buffer
        layer_b_i = layer.b_i_buffer
        layer_l_i = layer.l_i_buffer

    # average
    W_avg = T.cast((k * 1. / (k + 2)), 'floatX') * layer.W_avg + \
        T.cast((2. / (k + 2)), 'floatX') * W
    b_avg = T.cast((k * 1. / (k + 2)), 'floatX') * layer.b_avg + \
        T.cast((2. / (k + 2)), 'floatX') * b
    k = k + 1

    updates = ((layer.W, W),
               (layer.b, b),
               (layer.W_avg, W_avg),
               (layer.b_avg, b_avg),
               (layer.k, k),
               (layer.l, l),
               (layer_W_i, W_i),
               (layer_b_i, b_i),
               (layer_l_i, l_i),
               (layer.gamma, gamma))

    return updates
