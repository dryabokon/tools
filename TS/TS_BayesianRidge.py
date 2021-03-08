from sklearn.kernel_ridge import KernelRidge
# --------------------------------------------------------------------------------------------------------------------
class TS_BayesianRidge(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = 'TS_BayesianRidge'
        self.model = []
        self.folder_debug = folder_debug
        return
# ----------------------------------------------------------------------------------------------------------------
    def train(self, array_X, array_Y):
        self.model = KernelRidge(kernel='rbf')
        self.model = self.model.fit(array_X, array_Y)
        Y_pred = self.predict(array_X,array_Y)
        return Y_pred
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, array_X,array_Y):
        Y_pred = self.model.predict(array_X)
        return Y_pred
# ----------------------------------------------------------------------------------------------------------------------