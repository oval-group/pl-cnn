import unittest
import numpy as np


class TestDataAugmentation(unittest.TestCase):

    def test_horizontal_flip_no_swap(self):
        """
        Test horizontal flip without swapping axes
        NB: horizontal flip <=> reverse order on columns
        """

        np.random.seed(0)

        # 5 images of size 10 x 10 x 3
        shape = [5, 3, 10, 10]
        batch = np.random.random(size=shape).astype(np.float32)
        flipped = batch[:, :, :, ::-1]

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        ll = shape[3] - (l + 1)
                        assert batch[i, j, k, l] == flipped[i, j, k, ll]

    def test_horizontal_flip_with_swap(self):
        """
        Test horizontal flip with swapping axes
        NB: horizontal flip <=> reverse order on columns
        """

        np.random.seed(0)

        # 5 images of size 10 x 10 x 3
        shape = [5, 10, 10, 3]
        batch = np.random.random(size=shape).astype(np.float32)

        # move axis and then flip horizontally
        flipped_1 = np.moveaxis(batch, 3, 1)[:, :, :, ::-1]

        # flip horizontally without moving axis
        flipped_2 = batch[:, :, ::-1, :]

        for i in range(shape[0]):
            for j in range(shape[3]):
                assert np.all(np.isclose(flipped_1[i, j],
                                         flipped_2[i, :, :, j]))
