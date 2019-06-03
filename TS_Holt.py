import numpy
from statsmodels.tsa.api import Holt
# --------------------------------------------------------------------------------------------------------------------
class TS_Holt(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = 'TS_Holt'
        self.model = []
        self.folder_debug = folder_debug

        self.exponential = False
        self.damped = True
        self.optimized = False
        self.damping_slope = 0.98
        self.smoothing_slope = 0.2
        self.smoothing_level = 0.8
        self.train_X = []
        self.train_Y = []
        return
# ----------------------------------------------------------------------------------------------------------------
    def learn(self, array_X, array_Y):
        self.train_X = array_X
        self.train_Y = array_Y
        self.model = Holt(array_Y, exponential=self.exponential, damped=self.damped)
        self.fit = self.model.fit(smoothing_level=self.smoothing_level, smoothing_slope=self.smoothing_slope,damping_slope=self.damping_slope, optimized=self.optimized)
        res = self.fit.fittedvalues
        return res
#----------------------------------------------------------------------------------------------------------------------
    def predict_n_steps_ahead(self, n_steps):
        res = self.fit.forecast(n_steps)[0]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, test_X, test_Y):
        predictions = numpy.empty(0)
        for t in range(0, test_Y.shape[0]):
            array = numpy.hstack((self.train_Y, test_Y[:t]))
            model = Holt(array, exponential=self.exponential, damped = self.damped)
            fit = model.fit(smoothing_level=self.smoothing_level, smoothing_slope=self.smoothing_slope,damping_slope=self.damping_slope, optimized=self.optimized)
            predictions = numpy.append(predictions,fit.forecast(1)[0])

        return predictions
# ----------------------------------------------------------------------------------------------------------------------
