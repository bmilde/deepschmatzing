### some non-merged features of lasagne

from lasagne.init import Initializer
import itertools

import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.ticker import FormatStrFormatter

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

#shows the train history of a neural net
def plotTrainHistories(nets,names,y_lim_start=0.95,y_lim_stop=2.0):
    pyplot.gca().set_color_cycle(['c', 'c','r','r', 'g','g', 'y', 'y', 'k', 'k', 'm', 'm', 'b', 'b'])
    for net,name in itertools.izip(nets,names):
        train_loss = np.array([i['train_loss'] for i in net.train_history_])
        valid_loss = np.array([i['valid_loss'] for i in net.train_history_])
        pyplot.plot(train_loss, linewidth=3, label=name+' train')
        pyplot.plot(valid_loss, linewidth=3, linestyle='--',alpha=0.6, label=name+' valid')
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.yscale('log')
    pyplot.gca().yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    pyplot.ylim(y_lim_start, y_lim_stop)
    pyplot.show()
