from lasagne import layers
from lasagne import init

from lasagne.updates import sgd,nesterov_momentum
from nolearn.lasagne import NeuralNet

import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

DATA_PATH = '~/data'

mnist = fetch_mldata('MNIST original', data_home=DATA_PATH)

train = mnist.data[:60000].astype(np.float32)
train_labels = mnist.target[:60000].astype(np.int32)

train, train_labels = shuffle(train, train_labels, random_state=42) 

print 'train.shape:',train.shape,'train.dtype:',train.dtype,'train_labels.dtype:',train_labels.dtype

clf = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape = (None, train.shape[1]),
    hidden_num_units=100,
    output_num_units=10, 
    output_nonlinearity=None,

    update=nesterov_momentum,
    #update=sgd,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,
    max_epochs=1000,
    verbose=1,

    #W=init.Uniform()

    )

clf.fit(train,train_labels)
