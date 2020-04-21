#graphwiz!!
from sklearn import tree
import nltk
import numpy
import graphviz
from IPython.display import display
# --------------------------------------------------------------------------------------------------------------------
class classifier_DT(object):
    def __init__(self):
        self.name = "DT"
        self.model = tree.DecisionTreeClassifier(max_depth=11)
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

        #dot_data = tree.export_graphviz(self.model,filled=True)
        #graph = graphviz.Source(dot_data)
        #graph.format = "png"
        #graph.render('./data/output/tree.png')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        return self.model.predict_proba(self.maybe_reshape(array))
# ----------------------------------------------------------------------------------------------------------------------