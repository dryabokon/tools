#from dtreeviz.trees import dtreeviz
#import pydotplus
from sklearn import tree
import numpy
#import graphviz
import os
from matplotlib import pyplot as plt
# --------------------------------------------------------------------------------------------------------------------
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
# --------------------------------------------------------------------------------------------------------------------
class classifier_DT(object):
    def __init__(self,max_depth=2,folder_out=None):
        self.name = "DT"
        self.model = tree.DecisionTreeClassifier(max_depth=max_depth)
        self.folder_out = folder_out
        return
# ----------------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (-1,X.shape[0]))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train,columns=None,do_debug=False):
        self.model.fit(self.maybe_reshape(data_train), target_train)
        if do_debug and columns is not None:
            self.export_tree(feature_names=columns, filename_out=self.folder_out+'DT.png')
            self.export_tree_advanced(data_train, target_train, feature_names=columns, filename_out=self.folder_out+'DT.svg')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        return self.model.predict_proba(self.maybe_reshape(array))
# ----------------------------------------------------------------------------------------------------------------------
    def export_tree(self,feature_names=None,filename_out=None):

        plt.figure(figsize=(12, 8))
        dot_data = tree.export_graphviz(self.model,filled=True,rounded=False,feature_names=feature_names)
        graph = graphviz.Source(dot_data)

        ext = filename_out.split('.')[-1]
        name = filename_out.split('.'+ext)[0]
        graph.format = ext
        graph.render(name,cleanup=True)

        if os.path.isfile(name):
            os.remove(name)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_tree_advanced(self,X,Y,feature_names,filename_out):

        viz = dtreeviz(self.model, X, Y, feature_names=feature_names,colors={'classes': [None,  None,['#0080FF80', '#FF800080']]})
        viz.save(filename_out)
        ext = filename_out.split('.')[-1]
        name = filename_out.split('.' + ext)[0]
        if os.path.isfile(name):
            os.remove(name)

        return
# ----------------------------------------------------------------------------------------------------------------------