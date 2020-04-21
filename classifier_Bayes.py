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
        self.model = []
        return
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self,X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X,(-1,X.shape[0]))
# ----------------------------------------------------------------------------------------------------------------
    def learn(self, X, Y):

        self.domains = []

        for c in range(X.shape[1]):
            domain = {}
            feature = X[:, c]
            cnt = 1
            for val in feature:
                if val not in domain:
                    domain[val] = cnt
                    cnt += 1
            self.domains.append(domain)

        self.dct_pos = [{} for i in range(X.shape[1])]
        self.dct_neg = [{} for i in range(X.shape[1])]

        for x in X[Y.astype(int)>0]:
            for i,v in enumerate(x):
                if v in self.dct_pos[i]:
                    self.dct_pos[i][v] += 1
                else:
                    self.dct_pos[i][v] = 1

        for x in X[Y.astype(int)<=0]:
            for i,v in enumerate(x):
                if v in self.dct_neg[i]:
                    self.dct_neg[i][v] += 1
                else:
                    self.dct_neg[i][v] = 1

        return
# ----------------------------------------------------------------------------------------------------------------------
    def sigmoid(self, Z):
        return 1 / (1 + numpy.exp(-Z))
# ----------------------------------------------------------------------------------------------------------------------
    def predict_one(self, x):

        score =0
        for i,v in enumerate(x):
            p0 = 0.0001
            p1 = 0.0001
            if v in self.dct_neg[i]:
                p0 = self.dct_neg[i][v]
            if v in self.dct_pos[i]:
                p1 = self.dct_pos[i][v]

            score += -math.log(p0)+math.log(p1)

        return numpy.array([[0, score]])
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        if len(array.shape)==2 and array.shape[0]>1:
            result = []
            for x in array:
                result.append(self.predict_one(x)[0])
            result = numpy.array(result)
        else:
            result = self.predict_one(array)
        return result
# ----------------------------------------------------------------------------------------------------------------------

