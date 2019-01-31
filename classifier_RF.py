import numpy
from sklearn.ensemble import RandomForestClassifier
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_RF(object):
    def __init__(self):
        self.name = "RF"
# ----------------------------------------------------------------------------------------------------------------------
    def learn_on_arrays(self,data_train, target_train):
        self.model = RandomForestClassifier(n_estimators=1000,max_depth=10)
        self.model.fit(data_train, target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def learn_on_tensors(self, X_train_4d, Y_train_2d, output_path_models=None):
        data_array = X_train_4d.reshape((X_train_4d.shape[0], X_train_4d.shape[1] * X_train_4d.shape[2])).astype('float32')
        target_array = IO.from_categorical(Y_train_2d)
        self.learn_on_arrays(data_array, target_array)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def learn_on_features_file(self, file_train, delimeter='\t', path_models_detector=None):
        X_train = (IO.load_mat(file_train, numpy.chararray, delimeter))
        Y_train = X_train[:, 0].astype('float32')
        X_train = (X_train[:, 1:]).astype('float32')
        self.model = classifier_RF.learn_on_arrays(self, X_train, Y_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_tensor(self, tensors):
        arrays = tensors.reshape((tensors.shape[0], tensors.shape[1] * tensors.shape[2]))
        return self.predict_probability_of_array(arrays)
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_array(self, array):
        return self.model.predict_proba(array)
# ----------------------------------------------------------------------------------------------------------------------