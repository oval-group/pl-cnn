import unittest
import numpy as np
import theano
import theano.tensor as T

import layers.relu as relu
import layers.maxpool as maxpool


def compile_maxpool_like(pool_size):

    X_1 = T.tensor4()
    X_2 = T.tensor4()

    Y_1, Y_2 = maxpool.mirror_activations(X_1, X_2, pool_size)

    return theano.function([X_1, X_2], [Y_1, Y_2])


def compile_relu_like():

    X_1 = T.tensor4()
    X_2 = T.tensor4()

    Y_1, Y_2 = relu.mirror_activations(X_1, X_2)

    return theano.function([X_1, X_2], [Y_1, Y_2])


class TestFixedActivations(unittest.TestCase):

    def setUp(self):

        np.random.seed(0)

        self.input_shape = [3, 4, 10, 10]
        self.output_shape = [3, 4, 5, 5]
        self.pool_size = (2, 2)

        self.relu_like = compile_relu_like()
        self.maxpool_like = compile_maxpool_like(self.pool_size)

    def test_relu_like(self):
        """
        Test
        """

        X_1 = np.random.random(size=self.input_shape).astype(np.float32) - 0.5
        X_2 = np.random.random(size=self.input_shape).astype(np.float32) - 0.5
        Y_1, Y_2 = self.relu_like(X_1, X_2)

        Z_1 = np.zeros_like(Y_1)
        Z_2 = np.zeros_like(Y_2)

        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                for k in range(self.input_shape[2]):
                    for l in range(self.input_shape[3]):
                        if X_2[i, j, k, l] > 0.:
                            Z_1[i, j, k, l] = X_1[i, j, k, l]
                            Z_2[i, j, k, l] = X_2[i, j, k, l]

        assert np.all(np.isclose(Y_1, Z_1))
        assert np.all(np.isclose(Y_2, Z_2))

    def test_maxpool_like(self):
        """
        Test
        """

        X_1 = np.random.random(size=self.input_shape).astype(np.float32)
        X_2 = np.random.random(size=self.input_shape).astype(np.float32)
        Y_1, Y_2 = self.maxpool_like(X_1, X_2)

        Z_1 = np.zeros_like(Y_1)
        Z_2 = np.zeros_like(Y_2)

        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    for l in range(self.output_shape[3]):
                        max_ = -np.inf
                        for m in range(self.pool_size[0]):
                            for n in range(self.pool_size[1]):
                                kk = self.pool_size[0] * k + m
                                ll = self.pool_size[1] * l + n
                                if X_2[i, j, kk, ll] > max_:
                                    max_ = X_2[i, j, kk, ll]
                                    Z_1[i, j, k, l] = X_1[i, j, kk, ll]
                                    Z_2[i, j, k, l] = X_2[i, j, kk, ll]

        assert np.all(np.isclose(Y_1, Z_1))
        assert np.all(np.isclose(Y_2, Z_2))
