from sklearn import datasets
import numpy
from sklearn.model_selection import train_test_split
from vowpalwabbit.sklearn_vw import VWClassifier
# --------------------------------------------------------------------------------------------------------------------
class classifier_Vowpalwabbit(object):
    def __init__(self):
        self.name = "Vowpalwabbit"
        self.model = VWClassifier()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (-1,X.shape[0]))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):
        yyy = target_train.astype(float)
        yyy[yyy>0]=1
        yyy[yyy<=0]=-1
        self.model.fit(self.maybe_reshape(data_train).astype(numpy.float), yyy)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        xxx = self.maybe_reshape(array).astype(numpy.float)

        res = numpy.array(self.model.decision_function(xxx))
        #res = numpy.array(self.model.predict(xxx))
        return numpy.vstack((numpy.zeros_like(res), res)).T
# ----------------------------------------------------------------------------------------------------------------------
    def sanitycheck(self):
        X, y = datasets.make_hastie_10_2(n_samples=1000, random_state=1)
        X = X.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=256)

        model = VWClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score_train = model.score(X_train, y_train)
        scoer_test = model.score(X_test, y_test)
        return
# ----------------------------------------------------------------------------------------------------------------------