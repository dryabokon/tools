import numpy
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_Ada(object):
    def __init__(self):
        self.name = "Ada"
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (X.shape[0], -1))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):
        self.model = AdaBoostClassifier()
        #self.model.fit(preprocessing.minmax_scale(self.maybe_reshape(data_train), axis=0), target_train)
        self.model.fit(self.maybe_reshape(data_train), target_train)
        return
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        #return self.model.predict_proba(preprocessing.minmax_scale(self.maybe_reshape(array), axis=0))
        return self.model.predict_proba(self.maybe_reshape(array))
# ----------------------------------------------------------------------------------------------------------------------