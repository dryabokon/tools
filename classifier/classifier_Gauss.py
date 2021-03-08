import os
import numpy
import sys
import math
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_Gauss(object):
    def __init__(self):
        self.name = "Gauss"
        self.model = []
        self.mean = 0
        self.mean0 = []
        self.mean1 = []
        self.D0 = []
        self.D1 = []
        self.iD0 = []
        self.iD1 = []
# ----------------------------------------------------------------------------------------------------------------
    def set_model(self, model):
        self.model = model
        return
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X.astype(numpy.float32)
        else:
            return numpy.reshape(X, (X.shape[0], -1)).astype(numpy.float32)
# ----------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):

        X = self.maybe_reshape(data_train)

        X0 = numpy.array(X[target_train <=0])
        X1 = X[target_train >0]

        self.mean0 = numpy.average(X0, axis=0)
        self.mean1 = numpy.average(X1, axis=0)

        self.D0 = numpy.cov(X0.T)
        self.D1 = numpy.cov(X1.T)

        self.iD0 = numpy.linalg.pinv(self.D0)
        self.iD1 = numpy.linalg.pinv(self.D1)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):

        X = self.maybe_reshape(array)

        pred_score_train=numpy.zeros((X.shape[0], 2))
        for n in range(0,pred_score_train.shape[0]):
            for i in range (0,self.mean0.shape[0]):
                for j in range(0, self.mean0.shape[0]):

                    pred_score_train[n,1] -= self.iD0[i, j] * (X[n, i] - self.mean0[i]) * (X[n, j] - self.mean0[j]) -  \
                                             self.iD1[i, j] * (X[n, i] - self.mean1[i]) * (X[n, j] - self.mean1[j])



        return -pred_score_train
# ----------------------------------------------------------------------------------------------------------------------
