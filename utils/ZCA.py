#From https://gist.github.com/duschendestroyer/5170087
#Implements a sci-kit preprocessor for ZCA whitening of the data, see also http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

import numpy as np
from scipy import linalg
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
 
class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        self._mean = np.mean(X, axis=0)
        X -= self._mean
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self._components = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X = array2d(X)
        X_transformed = X - self._mean
        X_transformed = np.dot(X_transformed, self._components.T)
        return X_transformed
