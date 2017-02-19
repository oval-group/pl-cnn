import argparse
import numpy as np
import theano
import theano.tensor as T
import lasagne

from tqdm import tqdm

from neuralnet import NeuralNet
from layers import ConvLayer, ReLU
from config import Configuration as Cfg


def compile_make_fully_convolutional(nnet):

    # for naming convenience
    nnet.dense3_layer = nnet.svm_layer

    pad = 'valid'

    nnet.dense1_conv_layer = ConvLayer(nnet.maxpool5_layer,
                                       num_filters=4096,
                                       filter_size=(7, 7),
                                       pad=pad,
                                       flip_filters=False)

    relu_ = ReLU(nnet.dense1_conv_layer)

    nnet.dense2_conv_layer = ConvLayer(relu_,
                                       num_filters=4096,
                                       filter_size=(1, 1),
                                       pad=pad,
                                       flip_filters=False)

    relu_ = ReLU(nnet.dense2_conv_layer)

    nnet.dense3_conv_layer = ConvLayer(relu_,
                                       num_filters=1000,
                                       filter_size=(1, 1),
                                       pad=pad,
                                       flip_filters=False)

    W_dense1_reshaped = \
        nnet.dense1_layer.W.T.reshape(nnet.dense1_conv_layer.W.shape)
    W_dense2_reshaped = \
        nnet.dense2_layer.W.T.reshape(nnet.dense2_conv_layer.W.shape)
    W_dense3_reshaped = \
        nnet.dense3_layer.W.T.reshape(nnet.dense3_conv_layer.W.shape)

    updates = ((nnet.dense1_conv_layer.W, W_dense1_reshaped),
               (nnet.dense2_conv_layer.W, W_dense2_reshaped),
               (nnet.dense3_conv_layer.W, W_dense3_reshaped),
               (nnet.dense1_conv_layer.b, nnet.dense1_layer.b),
               (nnet.dense2_conv_layer.b, nnet.dense2_layer.b),
               (nnet.dense3_conv_layer.b, nnet.dense3_layer.b))

    return theano.function([], updates=updates)


def compile_eval_function(nnet):

    X = T.tensor4()
    y = T.ivector()

    # get prediciton by fully convolutional network
    prediction = lasagne.layers.get_output(nnet.dense3_conv_layer,
                                           deterministic=True, inputs=X)

    # get output scores on first dim
    # before flattening on 2dim and then get scores on second dim
    prediction = prediction.transpose((1, 0, 2, 3))\
        .flatten(2).transpose((1, 0))
    prediction = T.nnet.softmax(prediction)

    # spatial averaging
    prediction = T.mean(prediction, axis=0)

    # compute top1 and top5 accuracies
    sorted_pred = T.argsort(prediction)
    top1_acc = T.mean(T.eq(sorted_pred[-1], y), dtype='floatX')
    top5_acc = T.mean(T.any(T.eq(sorted_pred[-5:],
                                 T.shape_padright(y)), axis=1), dtype='floatX')

    return theano.function([X, y], [top1_acc, top5_acc])


def evaluate(weights_file):

    Cfg.compile_lwsvm = False
    Cfg.batch_size = 1
    Cfg.C.set_value(1e3)

    nnet = NeuralNet(dataset="imagenet", use_weights=weights_file)

    n_batches = int(50000. / Cfg.batch_size)
    make_fully_convolutional = compile_make_fully_convolutional(nnet)
    print("Weight transformation compiled.")
    make_fully_convolutional()
    print("Network has been made fully convolutional.")

    eval_fun = compile_eval_function(nnet)
    print("Evaluation function compiled")

    # full pass over the validation data:
    top1_acc = 0
    top5_acc = 0
    val_batches = 0
    count_images = 0
    for batch in tqdm(nnet.data.get_epoch_val(), total=n_batches):

        inputs, targets, _ = batch
        inputs = np.concatenate((inputs, inputs[:, :, :, ::-1]))
        top1, top5 = eval_fun(inputs, targets)
        top1_acc += top1
        top5_acc += top5
        val_batches += 1
        count_images += len(targets)

    print("(Used %i samples in validation)" % count_images)
    top1_acc *= 100. / val_batches
    top5_acc *= 100. / val_batches

    print("Top-1 validation accuracy: %g%%" % top1_acc)
    print("Top-5 validation accuracy: %g%%" % top5_acc)
