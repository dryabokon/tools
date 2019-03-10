import os
import numpy
import sys
import math
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_Bayes(object):
    def __init__(self):
        self.name = "Bayes"
        self.ctgr0 = []
        self.freq0 = []
        self.ctgr1 = []
        self.freq1 = []
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self,X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X,(X.shape[0],-1))
# ----------------------------------------------------------------------------------------------------------------
    def learn(self, X, Y):

        XX= self.maybe_reshape(X)

        X0 = numpy.array(XX[Y <= 0]).astype(int)
        X1 = numpy.array(XX[Y > 0]).astype(int)

        for j in range(0, XX.shape[1]):
            ct0 = numpy.unique(X0[:, j])
            ct1 = numpy.unique(X1[:, j])
            ct0 = numpy.hstack((ct0,1000000))
            ct1 = numpy.hstack((ct1,1000000))
            fr0 = numpy.histogram(X0[:, j], bins=ct0)[0]/X0.shape[0]
            fr1 = numpy.histogram(X1[:, j], bins=ct1)[0]/X1.shape[0]

            self.freq0.append(fr0)
            self.freq1.append(fr1)
            self.ctgr0.append(ct0)
            self.ctgr1.append(ct1)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):

        X = self.maybe_reshape(array)

        pred_score_train=numpy.zeros((X.shape[0], 2))

        for j in range(0, X.shape[1]):
            for n in range(0,pred_score_train.shape[0]):
                idx0 = IO.smart_index(self.ctgr0[j], int(X[n, j]))
                idx1 = IO.smart_index(self.ctgr1[j], int(X[n, j]))
                p0 = 0.0001
                p1 = 0.0001
                if idx0>=0:
                    p0=self.freq0[j][idx0][0]
                if idx1>=0:
                    p1=self.freq1[j][idx1][0]

                pred_score_train[n,1] += -math.log(p0)+math.log(p1)


        return pred_score_train
# ----------------------------------------------------------------------------------------------------------------------