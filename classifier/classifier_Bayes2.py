import numpy
from sklearn import naive_bayes
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_Bayes2(object):
    def __init__(self):
        self.name = "NaiveBayes2"
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (X.shape[0], -1))
# ----------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):
        self.model = naive_bayes.MultinomialNB()
        X = self.maybe_reshape(data_train)
        self.model.fit(X, target_train)#score = self.model.predict_proba(data_train)
        return
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        return self.model.predict_proba(self.maybe_reshape(array))
# ----------------------------------------------------------------------------------------------------------------------