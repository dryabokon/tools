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
    def learn_on_arrays(self,data_train, target_train):
        X0 = numpy.array(data_train[target_train <=0])
        X1 = data_train[target_train >0]

        self.mean0 = numpy.average(X0, axis=0)
        self.mean1 = numpy.average(X1, axis=0)

        self.D0 = numpy.cov(X0.T)
        self.D1 = numpy.cov(X1.T)

        self.iD0 = numpy.linalg.pinv(self.D0)
        self.iD1 = numpy.linalg.pinv(self.D1)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def learn_on_features_file(self, file_train, delimeter='\t', path_models_detector=None):
        X_train = (IO.load_mat(file_train, numpy.chararray,delimeter))

        Y_train = X_train[:,0].astype('float32')
        X_train = (X_train[:, 1:]).astype('float32')

        classifier_Gauss.learn_on_arrays(self, X_train, Y_train)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict_proba(self,x_train):
        pred_score_train=numpy.zeros((x_train.shape[0],2))
        for n in range(0,pred_score_train.shape[0]):
            for i in range (0,self.mean0.shape[0]):
                for j in range(0, self.mean0.shape[0]):

                    pred_score_train[n,1] -= self.iD0[i, j] * (x_train[n, i] - self.mean0[i]) * (x_train[n, j] - self.mean0[j]) -  \
                                             self.iD1[i, j] * (x_train[n, i] - self.mean1[i]) * (x_train[n, j] - self.mean1[j])

        #for n in range(0, pred_score_train.shape[0]):
            #if(pred_score_train[n, 1]>0):
                #pred_score_train[n, 1] = 1
            #else:
                #pred_score_train[n, 1] = 0


        return -pred_score_train
# ----------------------------------------------------------------------------------------------------------------------
    def score_feature_file(self, file_test, out_filename,delimeter='\t',append=0):

        data_test = IO.load_mat(file_test,numpy.chararray,delimeter)
        labels_test = (data_test[:, 0]).astype(numpy.str)
        data_test = data_test[:, 1:]

        if data_test[0,-1]==b'':
            data_test =data_test[:,:-1]

        data_test = data_test.astype('float32')

        score = classifier_Gauss.predict_proba(self,data_test)

        min = numpy.min(score[:, 1])
        max = numpy.max(score[:, 1])

        score = 100*((score[:, 1])-min)/(max-min).astype(int)

        mat = numpy.concatenate((numpy.matrix(labels_test).T,numpy.matrix(score).T),axis=1).astype('float32')
        IO.save_mat(mat,out_filename,delim=delimeter)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_array(self,array):
        score = classifier_Gauss.predict_proba(self,array)
        return score
# ----------------------------------------------------------------------------------------------------------------------
