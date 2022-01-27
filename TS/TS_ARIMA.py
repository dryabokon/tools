from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

import numpy
# --------------------------------------------------------------------------------------------------------------------
class TS_ARIMA(object):
    def __init__(self,folder_out,order=(1,0,0),do_debug=False):
        self.name = 'TS_ARIMA'
        self.model = []
        self.folder_out = folder_out
        self.order = order
        self.do_debug = do_debug
        return
# ----------------------------------------------------------------------------------------------------------------
    def train(self, array_X, array_Y):

        self.train_Y = array_Y
        self.model = ARIMA(array_Y,order=self.order)
        self.fit = self.model.fit()
        predict = array_Y + self.fit.resid

        if self.do_debug:
            self.fit.plot_diagnostics(figsize=(15, 12))
            plt.savefig(self.folder_out + self.name+'_diagnostics.png')

        return predict
# ---------------------------------------------------------------------------------------------------------------------
    def predict(self, test_X, Y,ongoing_retrain):
        predictions = []
        if ongoing_retrain:
            history = self.train_Y.copy().flatten()
            for t in range(0, Y.shape[0]):
                model = ARIMA(history,order=self.order)
                fit = model.fit()
                predictions.append(fit.forecast())
                history = numpy.append(history,[Y[t]])
        else:
            predictions, ci = self.predict_n_steps_ahead(Y.shape[0])

        return numpy.array(predictions).flatten()
# ----------------------------------------------------------------------------------------------------------------------
    def predict_n_steps_ahead(self, n_steps):
        start = self.train_Y.shape[0]
        res = self.fit.predict(start=start, end=start + n_steps - 1, dynamic=False)
        ci = self.fit.get_forecast(steps=n_steps).conf_int(0.05)

        return res, ci
# ----------------------------------------------------------------------------------------------------------------------
