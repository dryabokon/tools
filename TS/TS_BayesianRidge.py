import numpy
from sklearn.kernel_ridge import KernelRidge
# --------------------------------------------------------------------------------------------------------------------
class TS_BayesianRidge(object):
    def __init__(self,folder_debug=None,do_debug=False):
        self.name = 'TS_BayesianRidge'
        self.model = KernelRidge(kernel='linear')
        self.folder_debug = folder_debug
        self.do_debug = do_debug
        return
# ----------------------------------------------------------------------------------------------------------------
    def train(self, array_X, array_Y):
        self.train_X = array_X
        self.train_Y = array_Y
        self.model.fit(array_X, array_Y)
        Y_pred = self.model.predict(array_X)
        return Y_pred
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, X, Y, ongoing_retrain):
        predictions = numpy.empty(0)
        if ongoing_retrain:

            for t in range(0, Y.shape[0]):
                array_X = numpy.concatenate((self.train_X, X[:t]), axis=0)
                array_Y = numpy.hstack((self.train_Y, Y[:t]))
                #model = KernelRidge(kernel='rbf')
                #model = KernelRidge(kernel='linear')
                self.model.fit(array_X, array_Y)

                prediction = self.model.predict(X[t:t + 1])
                predictions = numpy.append(predictions, prediction)
        else:
            XX = numpy.concatenate((self.train_X, X), axis=0)
            predictions = self.model.predict(XX)[self.train_X.shape[0]:]

        return predictions
# ----------------------------------------------------------------------------------------------------------------------
