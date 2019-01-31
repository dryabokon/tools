import numpy
from sklearn import svm
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_SVM(object):
    def __init__(self,kernel='rbf'):
        self.kernel = kernel
        self.name = 'SVM'
        self.norm = 100.0
# ----------------------------------------------------------------------------------------------------------------------
    def learn_on_arrays(self,data_train, target_train):
        self.norm = numpy.max(data_train)
        if self.kernel == 'rbf':
            self.model = svm.SVC(probability=True,gamma=2, C=1)
        else:
            self.model = svm.SVC(probability=True,kernel='linear')
        self.model.fit(data_train/self.norm, target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
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
    def predict_probability_of_tensor(self, X_test_4d):
        array = X_test_4d.reshape((X_test_4d.shape[0], X_test_4d.shape[1] * X_test_4d.shape[2]))
        return self.predict_probability_of_array(array)
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_array(self,array):
        return self.model.predict_proba(array/self.norm)
# ----------------------------------------------------------------------------------------------------------------------