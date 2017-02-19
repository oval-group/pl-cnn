def conv_module_cifar(nnet, nfilters):

    filter_size = (3, 3)
    pad = 'same'
    use_batch_norm = True

    nnet.addConvLayer(num_filters=nfilters,
                      filter_size=filter_size,
                      pad=pad,
                      use_batch_norm=use_batch_norm)
    nnet.addReLU()

    nnet.addConvLayer(num_filters=nfilters,
                      filter_size=filter_size,
                      pad=pad,
                      use_batch_norm=use_batch_norm)
    nnet.addReLU()
    nnet.addMaxPool(pool_size=(2, 2))


def conv_module_vgg(nnet, nfilters, nrepeat):

    for _ in range(nrepeat):
        nnet.addConvLayer(num_filters=nfilters,
                          filter_size=(3, 3),
                          pad=1,
                          flip_filters=False)
        nnet.addReLU()

    nnet.addMaxPool(pool_size=(2, 2))
