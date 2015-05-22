### some non-merged features of lasagne

from lasagne.init import Initializer
import itertools
import theano
import theano.tensor as T

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

def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf

    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)
 
    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
 
        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
 
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates
