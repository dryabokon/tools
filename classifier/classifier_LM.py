from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import numpy
# --------------------------------------------------------------------------------------------------------------------
class classifier_LM(object):
    def __init__(self,multiclass_type=None):
        self.name = "LM"
        self.multiclass_type = multiclass_type  #ovr,None
        self.model = LogisticRegression(solver='liblinear')

        if self.multiclass_type=='ovr':
            self.model = OneVsRestClassifier(self.model)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (-1,X.shape[0]))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):

        self.model.fit(self.maybe_reshape(data_train), target_train)

        # param_grid = {'penalty': ['l1', 'l2'], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
        # clf = GridSearchCV(self.model, param_grid=param_grid,cv=3,scoring='f1_micro')
        # self.model = clf.fit(self.maybe_reshape(data_train), target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        res = self.model.predict_proba(self.maybe_reshape(array).astype(float))
        return res
# ----------------------------------------------------------------------------------------------------------------------