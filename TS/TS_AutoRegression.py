from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import numpy
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)
# --------------------------------------------------------------------------------------------------------------------
class TS_AutoRegression(object):
    def __init__(self, folder_out,lags=(1, 24), seasonal=True, period=24,do_debug=False):
        self.name = 'TS_AutoRegression'
        self.model = []
        self.folder_out = folder_out

        self.lags = lags
        self.seasonal = seasonal
        self.period = period
        self.do_debug = do_debug
        return

# ----------------------------------------------------------------------------------------------------------------
    def train(self, array_X, array_Y):

        self.train_Y = array_Y
        self.model = AutoReg(array_Y,lags=self.lags,seasonal=self.seasonal,period=self.period,old_names=False)
        self.fit = self.model.fit()

        resid = self.fit.resid
        resid = numpy.hstack((numpy.zeros(array_Y.shape[0]-resid.shape[0]),resid))
        self.predict_train = array_Y + resid

        if self.do_debug:
            summary = self.fit.summary(0.05)
            self.fit.plot_diagnostics(figsize=(15, 12))
            plt.savefig(self.folder_out + self.name+'_diagnostics.png')

        return self.predict_train
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, test_X,test_Y,ongoing_retrain=False):
        if ongoing_retrain:
            predictions = numpy.empty(0)
            for t in range(0, test_Y.shape[0]):
                array = numpy.hstack((self.train_Y, test_Y[:t]))
                model = AutoReg(array,lags=self.lags,seasonal=self.seasonal,period=self.period,old_names=False)
                fit = model.fit()
                prediction = fit.predict(start=array.shape[0], end=array.shape[0], dynamic=False)
                predictions = numpy.append(predictions,prediction)
        else:
            predictions, ci = self.predict_n_steps_ahead(test_Y.shape[0])

        return predictions
# ----------------------------------------------------------------------------------------------------------------------
    def predict_n_steps_ahead(self, n_steps):
        start = self.train_Y.shape[0]
        predict = self.fit.predict(start=start, end=start+n_steps-1, dynamic=False)
        ci      = self.fit.get_prediction(start=start, end=start+n_steps-1, dynamic=False).conf_int()
        #ci = numpy.hstack(((predict - numpy.std(resid)).reshape((-1, 1)),(predict + numpy.std(resid)).reshape((-1, 1))))
        return predict,ci
# ----------------------------------------------------------------------------------------------------------------------
