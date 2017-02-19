import theano
import theano.tensor as T

from grad.conv import symbolic_conditional_grad as grad_conv
from grad.dense import symbolic_dc_grad as grad_dense
from grad.svm import symbolic_grad as grad_svm

from opt.svm.updates import _update_cps, _update_std

from config import Configuration as Cfg


def compile_update_conv(nnet, layer):

    assert layer.isconv and not Cfg.store_on_gpu

    X = T.tensor4("X")
    y = T.ivector("y")

    dW, db, loss = grad_conv(nnet=nnet, layer=layer, X=X, y=y)

    updates = _update_std(nnet=nnet, layer=layer,
                          dW=dW, db=db, loss=loss)

    return theano.function([X, y],
                           updates=updates,
                           profile=nnet.profile)


def compile_update_dense(nnet, layer):

    assert layer.isdense and not Cfg.store_on_gpu

    X = T.tensor4()
    y = T.ivector()

    XX = layer.get_input_for(X)
    if XX.ndim > 2:
        XX = XX.flatten(2)

    Z_cvx = layer.get_output_for(XX)
    Z_ccv = layer.forward_with(XX, layer.W_fixed, layer.b_fixed)

    _, err_cvx = grad_dense(nnet=nnet, layer=layer, Z=Z_cvx, y=y)
    err_ccv, _ = grad_dense(nnet=nnet, layer=layer, Z=Z_ccv, y=y)

    dW_cvx = T.grad(err_cvx, Z_cvx)
    dW_ccv = T.grad(err_ccv, Z_ccv)
    dW = dW_cvx - dW_ccv
    db = dW.sum(axis=0)

    loss = (err_cvx - T.sum(Z_cvx * dW_cvx)) - \
        (err_ccv - T.sum(Z_ccv * dW_ccv))

    updates = _update_cps(nnet=nnet, layer=layer,
                          X=XX, dW=dW, db=db, loss=loss)

    return theano.function([X, y],
                           updates=updates,
                           profile=nnet.profile)


def compile_update_svm(nnet, layer):

    assert layer.issvm and not Cfg.store_on_gpu

    X = T.tensor4()
    y = T.ivector()

    XX = layer.get_input_for(X)
    if XX.ndim > 2:
        XX = XX.flatten(2)

    dW, db, loss = grad_svm(nnet, layer, XX, y)

    updates = _update_cps(nnet=nnet, layer=layer,
                          X=XX, dW=dW, db=db, loss=loss)

    return theano.function([X, y],
                           updates=updates,
                           profile=nnet.profile)
