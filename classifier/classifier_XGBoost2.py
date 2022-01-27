import numpy
from xgboost import XGBClassifier
from sklearn import preprocessing
# --------------------------------------------------------------------------------------------------------------------
import tools_IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_XGBoost2(object):
    def __init__(self):
        self.name = "XGBoost2"
        self.max_depth = 3
        self.model = XGBClassifier(max_depth=self.max_depth,n_estimators=50,use_label_encoder=False)
        return
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (-1,X.shape[0]))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train,data_test=None, target_test=None):

        if data_test is not None and target_test is not None:
            eval_set = [(preprocessing.normalize(self.maybe_reshape(data_test), axis=1), target_test)]
        else:
            eval_set = None

        self.model.fit(preprocessing.normalize(self.maybe_reshape(data_train), axis=1), target_train.astype('int'),eval_set=eval_set)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        return self.model.predict_proba(preprocessing.normalize(self.maybe_reshape(array), axis=1))
# ----------------------------------------------------------------------------------------------------------------------