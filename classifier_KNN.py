import numpy
from sklearn.neighbors import KNeighborsClassifier
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_KNN(object):
    def __init__(self):
        self.name = "KNN"
        self.norm = 100.0
# ----------------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self,X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X,(X.shape[0],-1))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):
        self.model = KNeighborsClassifier(2)
        self.model.fit(self.maybe_reshape(data_train), target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self,array):
        return self.model.predict_proba(self.maybe_reshape(array))
# ----------------------------------------------------------------------------------------------------------------------
