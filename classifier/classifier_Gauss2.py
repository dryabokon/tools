import numpy
from sklearn import gaussian_process
from sklearn import preprocessing
# --------------------------------------------------------------------------------------------------------------------
class classifier_Gauss2(object):
    def __init__(self):
        self.name = "Gauss2"
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (X.shape[0], -1))
# ----------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):
        self.model = gaussian_process.GaussianProcessClassifier()
        self.model.fit(preprocessing.normalize(self.maybe_reshape(data_train), axis=1), target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        return self.model.predict_proba(preprocessing.normalize(self.maybe_reshape(array), axis=1))
# ----------------------------------------------------------------------------------------------------------------------