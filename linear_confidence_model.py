from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Ridge

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LinearConfidenceModel(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classproba):
        self.linear_models = []
        for i in xrange(num_classproba):
            self.linear_models.append(Ridge(alpha=0.5,normalize=True)) #ElasticNet(normalize=True))

    def fit(self, X, y):
        assert(len(X)==2)

        probas = X[0]
        extras = X[1]
        
        argmaxes = np.argmax(probas, axis=1)

        num_probas = probas.shape[1]

        #assert right dimensions
        assert(num_probas==len(self.linear_models))

        #one linear model per class probability
        for i in xrange(num_probas):
            mask = np.equal(argmaxes,i)
            self.linear_models[i].fit(np.hstack([probas[mask],extras[mask]]),y[mask])
            print 'LinearConfidenceModel: class',i,'distribtuion:'
            print np.bincount(y[mask].astype(np.int32))
            print 'Weights:',self.linear_models[i].coef_

    def predict(self, X):
        assert(len(X)==2)

        probas = X[0]
        extras = X[1]

        len_predicts = probas.shape[0]

        confidences = np.zeros(len_predicts)

        #select linear model for the weight
        for i,elem in enumerate(probas):
            class_no = np.argmax(elem)
            confidences[i] = self.linear_models[class_no].predict(np.hstack([elem,extras[i]]))

        #print confidences

        return confidences
