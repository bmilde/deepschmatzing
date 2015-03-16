### some non-merged features of lasagne

from lasagne.init import Initializer


class ReluNormal(Initializer):
    """
    Initializer based on He et al, "Delving Deep into Rectifiers:
        Surpassing Human-Level Performance on Imagenet Classification".
    """

    def __init__(self):
        pass

    def sample(self, shape):
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are "
                               "supported.")
        # Same assumptions about meaning of different dimensions as Uniform.
        # eg, for a Conv2D layer with spatial 5x5 dimensions this would be 25.
        receptive_field_size = np.prod(shape[2:])
        input_channels = shape[1]
        n_l = input_channels*receptive_field_size
        std = np.sqrt(2.0/(n_l))
        return floatX(np.random.normal(0, std, size=shape))
