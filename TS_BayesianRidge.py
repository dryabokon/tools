from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
# --------------------------------------------------------------------------------------------------------------------
class TS_BayesianRidge(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = 'TS_BayesianRidge'
        self.model = []
        self.folder_debug = folder_debug
        return
# ----------------------------------------------------------------------------------------------------------------
    def learn(self, array_X, array_Y):
        self.model = KernelRidge(kernel='rbf')
        self.model.fit(array_X, array_Y)
        return self.predict(array_X,array_Y)
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, array_X,array_Y):
        return self.model.predict(array_X)
# ----------------------------------------------------------------------------------------------------------------------