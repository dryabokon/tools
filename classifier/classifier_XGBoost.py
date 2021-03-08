import numpy
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
# --------------------------------------------------------------------------------------------------------------------
import tools_IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_XGBoost(object):
    def __init__(self):
        self.name = "XGBoost"
        self.model = GradientBoostingClassifier()
        return
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (-1,X.shape[0]))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):

        self.model.fit(preprocessing.normalize(self.maybe_reshape(data_train), axis=1), target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def learn_file_batch(self, filename_train, has_header=True, n_epochs=1, batch_size=64, verbose=False):
        self.model = GradientBoostingClassifier(warm_start=True, n_estimators=1)
        N = tools_IO.count_lines(filename_train)

        splits = []
        for s in range(N // batch_size):
            splits.append(numpy.arange(s * batch_size, (s + 1) * batch_size, 1))

        for e in range(n_epochs):
            if verbose: print('Epoch ', e)
            for split in splits:
                data = numpy.array(tools_IO.get_lines(filename_train, start=has_header * 1 + split[0],end=has_header * 1 + split[-1]))
                self.model.fit(self.maybe_reshape(data[:, 1:]), data[:, 0])
                self.model.n_estimators += 1
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        return self.model.predict_proba(preprocessing.normalize(self.maybe_reshape(array), axis=1))
# ----------------------------------------------------------------------------------------------------------------------