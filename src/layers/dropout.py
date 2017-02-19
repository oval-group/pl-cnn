import lasagne.layers


class DropoutLayer(lasagne.layers.DropoutLayer):

    # for convenience
    isdense, isbatchnorm, isconv, ismaxpool, isrelu, issvm = (False,) * 6
    isdropout = True

    def __init__(self, incoming_layer, p=0.5, rescale=True, name=None):

        lasagne.layers.DropoutLayer.__init__(self, incoming_layer, name=name,
                                             p=0.5, rescale=True)

    def forward_prop(self, ZIn_ccv, ZIn_cvx, **kwargs):

        ZOut_ccv = self.get_output_for(ZIn_ccv, **kwargs)
        ZOut_cvx = self.get_output_for(ZIn_cvx, **kwargs)

        return ZOut_ccv, ZOut_cvx

    def forward_like(self, input, input_fixed, **kwargs):

        out = self.get_output_for(input, deterministic=True)
        out_fixed = self.get_output_for(input_fixed, deterministic=True)

        return out, out_fixed
