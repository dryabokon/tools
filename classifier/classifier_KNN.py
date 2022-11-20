import numpy
from sklearn.neighbors import KNeighborsClassifier
# --------------------------------------------------------------------------------------------------------------------
class classifier_KNN(object):
    def __init__(self):
        self.name = "KNN"
        self.norm = 100.0
        self.model = KNeighborsClassifier(n_neighbors=5, leaf_size=30)
# ----------------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self,X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X,(-1,X.shape[0]))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):

        self.model.fit(self.maybe_reshape(data_train), target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self,array):
        return self.model.predict_proba(self.maybe_reshape(array))
# ----------------------------------------------------------------------------------------------------------------------
