import numpy
# --------------------------------------------------------------------------------------------------------------------
class TS_Naive(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = 'TS_Naive'
        self.model = []
        self.folder_debug = folder_debug
        return
# ----------------------------------------------------------------------------------------------------------------
    def learn(self, array_X, array_Y):
        return numpy.hstack((array_Y[0], array_Y[:-1]))
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, array_X,array_Y):
        return numpy.hstack((array_Y[0],array_Y[:-1]))
# ----------------------------------------------------------------------------------------------------------------------