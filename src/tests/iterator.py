import unittest
import numpy as np

from datasets.iterator import indices_generator


class TestIterator(unittest.TestCase):

    def test_deterministic_multiple(self):
        """ Test indices generator when number of samples
        is a multiple of the batch size
        and when generation is deterministic
        """

        shuffle = False
        batch_size = 5
        n = 10

        n_batches = n / batch_size

        sample_indices = []
        batch_indices = []
        for batch in indices_generator(shuffle, batch_size, n):
            new_sample_indices, new_batch_index = batch
            sample_indices += new_sample_indices.tolist()
            batch_indices.append(new_batch_index)

        assert sample_indices == np.arange(n).tolist()
        assert batch_indices == np.arange(n_batches).tolist()

    def test_deterministic_non_multiple(self):
        """ Test indices generator when number of samples
        is not a multiple of the batch size
        and when generation is deterministic
        """

        shuffle = False
        batch_size = 3
        n = 10

        n_batches = int(np.ceil(n * 1. / batch_size))

        sample_indices = []
        batch_indices = []
        for batch in indices_generator(shuffle, batch_size, n):
            new_sample_indices, new_batch_index = batch
            sample_indices += new_sample_indices.tolist()
            batch_indices.append(new_batch_index)

        assert sample_indices == np.arange(n).tolist()
        assert batch_indices == np.arange(n_batches).tolist()

    def test_random_multiple(self):
        """ Test indices generator when number of samples
        is a multiple of the batch size
        and when generation is random
        """

        shuffle = True
        batch_size = 5
        n = 10
        np.random.seed(0)

        n_batches = n / batch_size

        sample_indices = []
        batch_indices = []
        for batch in indices_generator(shuffle, batch_size, n):
            new_sample_indices, new_batch_index = batch
            sample_indices += new_sample_indices.tolist()
            batch_indices.append(new_batch_index)

        assert sample_indices != np.arange(n).tolist()
        assert batch_indices != np.arange(n_batches).tolist()

        assert set(sample_indices) == set(np.arange(n))
        assert set(batch_indices) == set(np.arange(n_batches))

    def test_random_non_multiple(self):
        """ Test indices generator when number of samples
        is not a multiple of the batch size
        and when generation is random
        """

        shuffle = True
        batch_size = 3
        n = 10
        np.random.seed(0)

        n_batches = int(np.ceil(n * 1. / batch_size))

        sample_indices = []
        batch_indices = []
        for batch in indices_generator(shuffle, batch_size, n):
            new_sample_indices, new_batch_index = batch
            sample_indices += new_sample_indices.tolist()
            batch_indices.append(new_batch_index)

        assert sample_indices != np.arange(n).tolist()
        assert batch_indices != np.arange(n_batches).tolist()

        assert set(sample_indices) == set(np.arange(n))
        assert set(batch_indices) == set(np.arange(n_batches))
