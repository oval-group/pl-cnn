import unittest
import theano
import theano.tensor as T
import numpy as np

from layers.svm import SVMLayer


def compile_objective():

    scores = T.matrix()
    y_truth = T.ivector()

    objective, acc = SVMLayer.objective(scores, y_truth)

    return theano.function([scores, y_truth], [objective, acc])


def objective_py(scores, y_truth):

    objective, acc = 0, 0
    n_samples = scores.shape[0]
    n_classes = scores.shape[1]
    for i in range(n_samples):
        # find maximally violated constraint
        loss_augmented = np.array([scores[i, y] + int(y != y_truth[i])
                                  for y in range(n_classes)])
        y_star = np.argmax(loss_augmented)

        # update metrics
        delta = int(y_truth[i] != y_star)
        acc += int(y_truth[i] == np.argmax(scores[i]))
        objective += delta + scores[i, y_star] - scores[i, y_truth[i]]

    return objective, acc


class TestSVM(unittest.TestCase):

    def setUp(self):

        self.n_samples = 20
        self.n_classes = 100
        self.k = 5

    def test_objective_svm(self):
        """ Test objective function of standard top-1 SVM
        """

        objective_theano = compile_objective()

        scores = np.random.normal(size=(self.n_samples, self.n_classes)) \
            .astype(np.float32)
        y_truth = np.random.randint(0, self.n_classes, size=self.n_samples) \
            .astype(np.int32)

        objective_1, acc_1 = objective_theano(scores, y_truth)
        objective_2, acc_2 = objective_py(scores, y_truth)

        assert np.isclose(objective_1, objective_2)
        assert np.isclose(acc_1, acc_2)
