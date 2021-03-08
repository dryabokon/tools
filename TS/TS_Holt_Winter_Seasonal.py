import numpy
from statsmodels.tsa.api import ExponentialSmoothing
# --------------------------------------------------------------------------------------------------------------------
class TS_Holt_Winter_Seasonal(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = 'TS_Holt_Winter_Seasonal'
        self.model = []
        self.folder_debug = folder_debug

        self.seasonal_periods=6
        self.train_X = []
        self.train_Y = []
        return
# ----------------------------------------------------------------------------------------------------------------
    def train(self, array_X, array_Y):
        self.train_X = array_X
        self.train_Y = array_Y
        self.model = ExponentialSmoothing(array_Y, seasonal_periods=self.seasonal_periods, trend='add', seasonal='add')
        self.fit = self.model.fit()
        res = self.fit.fittedvalues
        return res
    # ----------------------------------------------------------------------------------------------------------------------
    def predict_n_steps_ahead(self, n_steps):
        res = self.fit.forecast(n_steps)[0]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array_X,array_Y):
        predictions=numpy.empty(0)
        for t in range(0,array_Y.shape[0]):
            array = numpy.hstack((self.train_Y,array_Y[:t]))
            model = ExponentialSmoothing(array, seasonal_periods=self.seasonal_periods, trend='add', seasonal='add')
            fit = model.fit()
            predictions = numpy.append(predictions,fit.forecast(1)[0])

        return predictions
# ----------------------------------------------------------------------------------------------------------------------
