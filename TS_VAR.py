from statsmodels.tsa.vector_ar.var_model import VAR
import statsmodels.api as sm
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
import pandas as pd
import numpy
# --------------------------------------------------------------------------------------------------------------------
class TS_VAR(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = 'TS_VAR'
        self.model = []
        self.folder_debug = folder_debug
        return
# ----------------------------------------------------------------------------------------------------------------
    def train(self, array_X, array_Y):
        self.train_X = array_X
        self.train_Y = array_Y
        array = numpy.concatenate((numpy.matrix(array_Y).T, array_X), axis=1)
        model = VAR(endog=pd.DataFrame(data=array))
        fit = model.fit()
        res = fit.fittedvalues.values[:,0]
        res = numpy.hstack((res[0], res))
        return res
# ---------------------------------------------------------------------------------------------------------------------
    def predict(self, test_X, test_Y):
        predictions = numpy.empty(0)

        array_train = numpy.concatenate((numpy.array([self.train_Y]).T, self.train_X), axis=1)
        array_test  = numpy.concatenate((numpy.array([test_Y]).T, test_X), axis=1)

        for t in range(0, test_Y.shape[0]):
            array = numpy.vstack((array_train, array_test[:t]))
            model = VAR(endog=pd.DataFrame(data=array))
            fit = model.fit()
            lag = fit.k_ar
            pred = fit.forecast(array[-lag:],1)[0]
            predictions = numpy.append(predictions,pred[0])

        return predictions
# ----------------------------------------------------------------------------------------------------------------------
