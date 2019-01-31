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
    def learn_on_arrays(self,data_train, target_train):

        X0 = numpy.array(data_train[target_train <=0])
        X1 = numpy.array(data_train[target_train  >0])

        self.mean0 = numpy.average(X0, axis=0)
        self.mean1 = numpy.average(X1, axis=0)
        self.D0 = numpy.zeros(self.mean0.shape[0])
        self.D1 = numpy.zeros(self.mean0.shape[0])


        for i in range (0,self.mean0.shape[0]):
            self.D0[i] = numpy.cov(X0[:, i], X0[:, i])[0,0]
            self.D1[i] = numpy.cov(X1[:, i], X1[:, i])[0,0]

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

        for n in range(0,pred_score_train.shape[0]):
            for i in range (0,self.mean0.shape[0]):
                if(self.D0[i]!=0):
                    pred_score_train[n,1] -=  (x_train[n, i] - self.mean0[i]) * (x_train[n, i] - self.mean0[i])/self.D0[i]

                if(self.D1[i] != 0):
                    pred_score_train[n, 1] += (x_train[n, i] - self.mean1[i]) * (x_train[n, i] - self.mean1[i])/self.D1[i]

        return -pred_score_train
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_array(self,array):
        score = self.predict_proba(array)
        return score
# ----------------------------------------------------------------------------------------------------------------------
