import theano
import theano.tensor as T
import lasagne.layers
import lasagne.updates as l_updates
import lasagne.objectives as l_objectives

from config import Configuration as Cfg


def get_updates(nnet,
                train_obj,
                trainable_params):

    implemented_solvers = ("nesterov", "adagrad", "adadelta", "adam")

    if not hasattr(nnet, "solver") or nnet.solver not in implemented_solvers:
        nnet.sgd_solver = "nesterov"
    else:
        nnet.sgd_solver = nnet.solver

    if nnet.sgd_solver == "nesterov":
        updates = l_updates.nesterov_momentum(train_obj,
                                              trainable_params,
                                              learning_rate=Cfg.learning_rate,
                                              momentum=0.9)

    elif nnet.sgd_solver == "adagrad":
        updates = l_updates.adagrad(train_obj,
                                    trainable_params,
                                    learning_rate=Cfg.learning_rate)

    elif nnet.sgd_solver == "adadelta":
        updates = l_updates.adadelta(train_obj,
                                     trainable_params,
                                     learning_rate=Cfg.learning_rate)

    elif nnet.sgd_solver == "adam":
        updates = l_updates.adam(train_obj,
                                 trainable_params,
                                 learning_rate=Cfg.learning_rate)

    return updates


def create_update(nnet):
    """ create an SVM loss for network given in argument
    """

    inputs = T.tensor4('inputs')
    targets = T.ivector('targets')

    C = Cfg.C
    floatX = Cfg.floatX

    svm_layer = nnet.svm_layer

    trainable_params = lasagne.layers.get_all_params(svm_layer, trainable=True)

    prediction = lasagne.layers.get_output(svm_layer, inputs=inputs,
                                           deterministic=False)

    if Cfg.softmax_loss:
        print("Using softmax output")
        out = lasagne.nonlinearities.softmax(prediction)

        train_loss = l_objectives.categorical_crossentropy(out, targets).mean()
        train_acc = T.mean(T.eq(T.argmax(prediction, axis=1),
                                targets), dtype='floatX')
    else:
        objective, train_acc = svm_layer.objective(prediction, targets)

        train_loss = T.cast((objective) / targets.shape[0], 'floatX')
        train_acc = T.cast(train_acc * 1. / targets.shape[0], 'floatX')

    # NB: biases in L2-regularization
    l2_penalty = 0
    for layer in nnet.trainable_layers:
        l2_penalty = l2_penalty + T.sum(layer.W ** 2) + T.sum(layer.b ** 2)

    train_obj = floatX(0.5) / C * l2_penalty + train_loss

    updates = get_updates(nnet, train_obj, trainable_params)

    nnet.backprop = theano.function([inputs, targets],
                                    [train_obj, train_acc],
                                    updates=updates)

    nnet.hinge_loss = theano.function([inputs, targets],
                                      [train_loss, train_acc])

    prediction = lasagne.layers.get_output(svm_layer, inputs=inputs,
                                           deterministic=True)
    objective, test_acc = svm_layer.objective(prediction, targets)
    test_loss = T.cast(objective / targets.shape[0], 'floatX')
    test_acc = T.cast(test_acc * 1. / targets.shape[0], 'floatX')
    test_obj = floatX(0.5) / C * l2_penalty + test_loss

    nnet.forward = theano.function([inputs, targets],
                                   [test_obj, test_acc])
