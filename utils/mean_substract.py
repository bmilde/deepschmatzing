#From https://gist.github.com/duschendestroyer/5170087
#Implements a sci-kit preprocessor for ZCA whitening of the data, see also http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

import numpy as np
from scipy import linalg
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
 
class MeanNormalize(BaseEstimator, TransformerMixin):

    def __init__(self, copy=False):
        self.copy = copy

    def fit(self, X, y=None):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        self._mean = np.mean(X, axis=0)
        #X -= self._mean
        return self

    def transform(self, X):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        X -= self._mean
        return X
