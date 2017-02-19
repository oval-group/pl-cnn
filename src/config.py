import numpy as np
import theano


class Configuration(object):

    batch_size = 200
    floatX = np.float32

    C = theano.shared(floatX(1e3), name="C")
    D = theano.shared(floatX(1e2), name="D")
    learning_rate = theano.shared(floatX(1e-4), name="learning rate")

    eps = floatX(1e-8)

    horizontal_flip = False

    draw_on_board = False

    compile_lwsvm = False

    softmax_loss = False

    hostname = "dhcp45.robots.ox.ac.uk"
