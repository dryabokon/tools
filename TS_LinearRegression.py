from sklearn.linear_model import LinearRegression

import numpy
# --------------------------------------------------------------------------------------------------------------------
class TS_LinearRegression(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = 'TS_LinearRegression'
        self.model = []
        self.folder_debug = folder_debug
        return
# ----------------------------------------------------------------------------------------------------------------
    def train(self, array_X, array_Y):
        self.train_X = array_X
        self.train_Y = array_Y
        self.model = LinearRegression()
        self.model.fit(array_X, array_Y)
        return self.predict(array_X,array_Y)
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, test_X,test_Y):
        return self.model.predict(test_X)
# ----------------------------------------------------------------------------------------------------------------------
