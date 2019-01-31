import os
import numpy
import sys
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_LM(object):
    def __init__(self):
        self.model = []
        self.model_name = "model_LM.dmp"
        self.name = "LM"
        self.A = []
# ----------------------------------------------------------------------------------------------------------------
    def set_model(self, model):
        self.model = model
        return
# ----------------------------------------------------------------------------------------------------------------
    def load_model(self,filename):
        self.model = joblib.load(filename)
        return self.model
# ----------------------------------------------------------------------------------------------------------------------
    def save_to_disk(self, filename=None):
        if(filename != None):
            joblib.dump(self.model, filename)
        return
# ----------------------------------------------------------------------------------------------------------------
    def learn_on_arrays(self,data_train, target_train):

        a=numpy.dot(data_train.T , data_train)
        b=numpy.linalg.pinv(a)
        c=numpy.dot(b,data_train.T)
        self.A = numpy.dot(c, target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def learn_on_features_file(self, file_train, delimeter='\t', path_models_detector=None):
        X_train = (IO.load_mat(file_train, numpy.chararray,delimeter))

        Y_train = X_train[:,0].astype('float32')
        X_train = (X_train[:, 1:]).astype('float32')

        self.model = classifier_LM.learn_on_arrays(self, X_train, Y_train)
        joblib.dump(self.model, path_models_detector + self.model_name)
        return

# ----------------------------------------------------------------------------------------------------------------------
    def predict_proba(self, x_train):
        score = 100 * numpy.dot(x_train, self.A)
        res= numpy.full((score.shape[0],2),0)
        res[:,1]= score
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_array(self,array):
        score = self.predict_proba(array)
        return score
# ----------------------------------------------------------------------------------------------------------------------
