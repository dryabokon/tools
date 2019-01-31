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
    def learn_on_arrays(self,data_train, target_train):
        X0 = numpy.array(data_train[target_train <=0]).astype(int)
        X1 = numpy.array(data_train[target_train > 0]).astype(int)

        for j in range(0,data_train.shape[1]):
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
    def learn_on_features_file(self, file_train, delimeter='\t', path_models_detector=None):
        X_train = (IO.load_mat(file_train, numpy.chararray,delimeter))
        Y_train = X_train[:,0].astype('float32')
        X_train = (X_train[:, 1:]).astype('float32')
        self.learn_on_arrays(X_train, Y_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict_proba(self,x_train):

        pred_score_train=numpy.zeros((x_train.shape[0],2))

        for j in range(0,x_train.shape[1]):
            for n in range(0,pred_score_train.shape[0]):
                idx0 = IO.smart_index(self.ctgr0[j], int(x_train[n, j]))
                idx1 = IO.smart_index(self.ctgr1[j], int(x_train[n, j]))
                p0 = 0.0001
                p1 = 0.0001
                if idx0>=0:
                    p0=self.freq0[j][idx0][0]
                if idx1>=0:
                    p1=self.freq1[j][idx1][0]

                pred_score_train[n,1] += -math.log(p0)+math.log(p1)


        return pred_score_train
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_array(self, array):
        score = self.predict_proba(array)
        return score
# ----------------------------------------------------------------------------------------------------------------------