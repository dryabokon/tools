from statsmodels.tsa.ar_model import AR
import numpy
# --------------------------------------------------------------------------------------------------------------------
class TS_AutoRegression(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = 'TS_AutoRegression'
        self.model = []
        self.folder_debug = folder_debug
        return
# ----------------------------------------------------------------------------------------------------------------
    def learn(self, array_X, array_Y):
        self.train_X = array_X
        self.train_Y = array_Y
        self.model = AR(array_Y)
        self.fit = self.model.fit()
        res = self.fit.fittedvalues
        res = numpy.hstack((array_Y[:array_Y.shape[0]-res.shape[0]],res))

        window = self.fit.k_ar
        res2 = self.fit.predict(start=window, end=array_Y.shape[0]-1, dynamic=False)
        res2 = numpy.hstack((array_Y[:array_Y.shape[0] - res2.shape[0]], res2))

        return res2
#----------------------------------------------------------------------------------------------------------------------
    def predict_n_steps_ahead(self, n_steps):
        start = self.train_Y.shape[0]
        res = self.fit.predict(start=start, end=start+n_steps-1, dynamic=False)
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, test_X,test_Y):#ongoing_relearn
        predictions = numpy.empty(0)
        for t in range(0, test_Y.shape[0]):
            array = numpy.hstack((self.train_Y, test_Y[:t]))
            model = AR(array)
            fit = model.fit()
            prediction = fit.predict(start=array.shape[0], end=array.shape[0], dynamic=False)
            predictions = numpy.append(predictions,prediction)
        return predictions
# ----------------------------------------------------------------------------------------------------------------------
