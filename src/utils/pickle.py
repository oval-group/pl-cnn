import cPickle as pickle


def dump_weights(nnet, filename=None):

    if filename is None:
        filename = nnet.pickle_filename

    weight_dict = dict()

    for layer in nnet.trainable_layers:
        weight_dict[layer.name + "_w"] = layer.W.get_value()
        weight_dict[layer.name + "_b"] = layer.b.get_value()

    for layer in nnet.all_layers:
        if layer.isbatchnorm:
            weight_dict[layer.name + "_beta"] = layer.beta.get_value()
            weight_dict[layer.name + "_gamma"] = layer.gamma.get_value()
            weight_dict[layer.name + "_mean"] = layer.mean.get_value()
            weight_dict[layer.name + "_inv_std"] = layer.inv_std.get_value()

    with open(filename, 'wb') as f:
        pickle.dump(weight_dict, f)

    print("Parameters saved in %s" % filename)


def load_weights(nnet, filename=None):

    if filename is None:
        filename = nnet.pickle_filename

    with open(filename, 'rb') as f:
        weight_dict = pickle.load(f)

    for layer in nnet.trainable_layers:
        layer.W.set_value(weight_dict[layer.name + "_w"])
        layer.b.set_value(weight_dict[layer.name + "_b"])

    for layer in nnet.all_layers:
        if layer.isbatchnorm:
            layer.beta.set_value(weight_dict[layer.name + "_beta"])
            layer.gamma.set_value(weight_dict[layer.name + "_gamma"])
            layer.mean.set_value(weight_dict[layer.name + "_mean"])
            layer.inv_std.set_value(weight_dict[layer.name + "_inv_std"])

    print("Parameters loaded in network")
