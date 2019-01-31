import numpy
from sklearn import gaussian_process
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_Gauss2(object):
    def __init__(self):
        self.name = "Gauss2"
# ----------------------------------------------------------------------------------------------------------------
    def learn_on_arrays(self,data_train, target_train):
        self.model = gaussian_process.GaussianProcessClassifier()
        self.model.fit(data_train, target_train)#score = self.model.predict_proba(data_train)
        return
#----------------------------------------------------------------------------------------------------------------------
    def learn_on_tensors(self, X_train_4d, Y_train_2d, output_path_models=None):
        data_array = X_train_4d.reshape((X_train_4d.shape[0], X_train_4d.shape[1] * X_train_4d.shape[2])).astype('float32')
        target_array = IO.from_categorical(Y_train_2d)
        self.learn_on_arrays(data_array, target_array)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def learn_on_features_file(self, file_train, delimeter='\t', output_path_models=None):
        X_train = (IO.load_mat(file_train, numpy.chararray,delimeter))
        Y_train = X_train[:, 0].astype('float32')
        X_train = (X_train[:, 1:]).astype('float32')
        self.learn_on_arrays(X_train, Y_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_tensor(self, tensors):
        arrays = tensors.reshape((tensors.shape[0], tensors.shape[1] * tensors.shape[2]))
        return self.predict_probability_of_array(arrays)
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_array(self, array):
        return self.model.predict_proba(array)
# ----------------------------------------------------------------------------------------------------------------------