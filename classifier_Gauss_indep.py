import os
import numpy
import sys
import math
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_Gauss_indep(object):
    def __init__(self):
        self.name = "GaussIndep"
        self.mean0 = []
        self.mean1 = []
        self.D0 = []
        self.D1 = []
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X.astype(numpy.float32)
        else:
            return numpy.reshape(X, (X.shape[0], -1)).astype(numpy.float32)
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):

        X = self.maybe_reshape(data_train)

        X0 = numpy.array(X[target_train <=0])
        X1 = numpy.array(X[target_train  >0])

        self.mean0 = numpy.average(X0, axis=0)
        self.mean1 = numpy.average(X1, axis=0)
        self.D0 = numpy.zeros(self.mean0.shape[0])
        self.D1 = numpy.zeros(self.mean0.shape[0])


        for i in range (0,self.mean0.shape[0]):
            self.D0[i] = numpy.cov(X0[:, i], X0[:, i])[0,0]
            self.D1[i] = numpy.cov(X1[:, i], X1[:, i])[0,0]

        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):

        X = self.maybe_reshape(array)

        pred_score_train=numpy.zeros((X.shape[0], 2))

        for n in range(0,pred_score_train.shape[0]):
            for i in range (0,self.mean0.shape[0]):
                if(self.D0[i]!=0):
                    pred_score_train[n,1] -= (X[n, i] - self.mean0[i]) * (X[n, i] - self.mean0[i]) / self.D0[i]

                if(self.D1[i] != 0):
                    pred_score_train[n, 1] += (X[n, i] - self.mean1[i]) * (X[n, i] - self.mean1[i]) / self.D1[i]

        return -pred_score_train
# ----------------------------------------------------------------------------------------------------------------------
