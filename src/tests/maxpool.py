import unittest
import numpy as np
import theano
import theano.tensor as T

from utils.patches import my_pool_2d


def compile_maxpool(output_shape, pool_size):

    X = T.tensor4()

    # compute output with both methods
    out1 = T.signal.pool.pool_2d(X, pool_size,
                                 ignore_border=True, st=None,
                                 padding=(0, 0), mode='max')

    out2 = my_pool_2d(X, pool_size,
                      ignore_border=True, st=None,
                      padding=(0, 0), mode='max')

    # compute gradient with random incoming gradient for both cases
    incoming_grad = T.as_tensor_variable(np.random.random(size=output_shape)
                                         .astype(np.float32))
    grad1 = T.grad(None, wrt=X, known_grads={out1: incoming_grad})
    grad2 = T.grad(None, wrt=X, known_grads={out2: incoming_grad})

    return theano.function([X], [out1, out2, grad1, grad2])


class TestMaxPoolCPU(unittest.TestCase):

    def setUp(self):

        np.random.seed(0)

        self.input_shape = [3, 4, 10, 10]
        self.output_shape = [3, 4, 5, 5]
        self.pool_size = (2, 2)

        self.maxpool = compile_maxpool(self.output_shape, self.pool_size)

    def test_maxpool_non_edge_case(self):
        """
        Test MaxPooling on a non-edge case:
        inputs have different values in a patch
        """

        X = np.random.random(size=self.input_shape).astype(np.float32)
        out1, out2, grad1, grad2 = self.maxpool(X)

        assert np.all(np.isclose(out1, out2))
        assert np.all(np.isclose(grad1, grad2))

    def test_maxpool_edge_case(self):
        """
        Test MaxPooling on an edge case: inputs have same values in a patch
        Check one and only one gradient is back-propagated in each patch
        """

        X = np.zeros(shape=self.input_shape, dtype=np.float32)
        out1, out2, _, grad = self.maxpool(X)

        assert np.all(np.isclose(out1, out2))

        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    for l in range(self.output_shape[3]):
                        count = 0
                        for m in range(self.pool_size[0]):
                            for n in range(self.pool_size[1]):
                                kk = self.pool_size[0] * k + m
                                ll = self.pool_size[1] * l + n
                                if grad[i, j, kk, ll] != 0.:
                                    count += 1
                        assert count == 1
