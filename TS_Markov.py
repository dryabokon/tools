from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import pandas as pd
import numpy
# --------------------------------------------------------------------------------------------------------------------
class TS_Markov(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = 'TS_Markov'
        self.model = []
        self.folder_debug = folder_debug
        return
# ----------------------------------------------------------------------------------------------------------------
    def learn(self, array_X, array_Y):
        self.train_X = array_X
        self.train_Y = array_Y
        model = MarkovRegression(endog=pd.DataFrame(data=array_Y), k_regimes=12)
        fit = model.fit()
        res = fit.fittedvalues.values
        return res
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, array_X,array_Y):
        model = MarkovRegression(endog=pd.DataFrame(data=array_Y), k_regimes=12)
        fit = model.fit()
        res = fit.fittedvalues.values
        return res
# ----------------------------------------------------------------------------------------------------------------------