import theano
import theano.tensor as T


def symbolic_grad(nnet, layer, X, y):

    scores = layer.get_output_for(X)
    objective, acc = layer.objective(scores, y)

    dW = T.grad(objective, layer.W)
    db = T.grad(objective, layer.b)

    loss = objective - (T.sum(dW * layer.W) + T.sum(db * layer.b))

    return dW, db, loss


def compile_grad(nnet,
                 layer):
    """
    """

    assert layer.issvm

    y = T.ivector()
    idx = T.iscalar()

    dW, db, loss = symbolic_grad(nnet, layer, idx, y)

    updates = [(layer.dW, dW),
               (layer.db, db),
               (layer.loss, loss)]

    # return compiled function
    return theano.function([idx, y],
                           updates=updates,
                           profile=nnet.profile)
