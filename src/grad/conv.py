import theano.tensor as T


def dc_forward(nnet, layer, Z_cvx, y):

    # forward through piecewise maxima (ReLU, Maxpool), which keep convexity
    for next_layer in nnet.next_layers(layer):
        if next_layer in nnet.trainable_layers:
            break
        Z_cvx = next_layer.get_output_for(Z_cvx, deterministic=True)

    start_layer = next_layer

    # initialize concave part
    Z_ccv = T.zeros_like(Z_cvx)

    # compute DC forward decomposition until hinge loss
    for next_layer in nnet.next_layers(start_layer):
        Z_ccv, Z_cvx = next_layer.forward_prop(Z_ccv, Z_cvx)

    err_ccv, err_cvx = nnet.svm_layer.forward_prop(Z_ccv, Z_cvx, y)

    return err_ccv, err_cvx


def symbolic_conditional_grad(nnet, layer, X, y):

    assert layer.isconv

    X_layer = layer.get_input_for(X, deterministic=True)

    # forward pass on current network with non-linear activations
    # chosen by fixed network
    Z_cur = layer.get_output_for(X_layer, deterministic=True)
    Z_fix = layer.forward_with(X_layer, layer.W_fixed, layer.b_fixed,
                               deterministic=True)

    for next_layer in nnet.next_layers(layer):
        if next_layer in nnet.trainable_layers:
            Z_cur = next_layer.get_output_for(Z_cur, deterministic=True)
            Z_fix = next_layer.get_output_for(Z_fix, deterministic=True)
        else:
            Z_cur, Z_fix = next_layer.forward_like(Z_cur, Z_fix,
                                                   deterministic=True)

    svm = nnet.svm_layer

    objective, acc = svm.objective(Z_cur, y)

    dW = T.grad(objective, layer.W)
    db = T.grad(objective, layer.b)

    loss = objective - T.sum(dW * layer.W) - T.sum(db * layer.b)

    return dW, db, loss
