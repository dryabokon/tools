import numpy
from sklearn import svm
from sklearn import preprocessing
# --------------------------------------------------------------------------------------------------------------------
class classifier_SVM(object):
    def __init__(self,kernel='rbf'):
        self.kernel = kernel
        self.name = 'SVM'
# ----------------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X.astype(numpy.float32)
        else:
            return numpy.reshape(X, (X.shape[0], -1))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):

        if self.kernel == 'rbf':
            self.model = svm.SVC(probability=True,gamma=2, C=1)
        else:
            self.model = svm.SVC(probability=True,kernel='linear')

        X = self.maybe_reshape(data_train)
        X = preprocessing.normalize(X,axis=1)
        self.model.fit(X, target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self,array):
        X = self.maybe_reshape(array)
        X = preprocessing.normalize(X,axis=1)
        return self.model.predict_proba(X)
# ----------------------------------------------------------------------------------------------------------------------