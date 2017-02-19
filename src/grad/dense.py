import theano.tensor as T


def symbolic_dc_grad(nnet, layer, Z, y):
    """
    Symbolic computation of the gradient with a DC decomposition
    Z should be the output of the dense layer
    """

    svm = nnet.svm_layer

    Z_cvx = Z
    Z_ccv = T.zeros_like(Z)

    # feature fed to the SVM
    for next_layer in nnet.next_layers(layer):
        if not next_layer.issvm:
            Z_ccv, Z_cvx = next_layer.forward_prop(Z_ccv, Z_cvx)

    err_ccv, err_cvx = svm.forward_prop(Z_ccv, Z_cvx, y)

    return err_ccv, err_cvx
