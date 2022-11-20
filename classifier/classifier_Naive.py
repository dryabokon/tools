import numpy
from collections import Counter
# --------------------------------------------------------------------------------------------------------------------
class classifier_Naive(object):
    def __init__(self):
        self.name = "Naive"
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):
        self.model = []
        cntr = Counter(target_train)
        self.classes = numpy.sort(numpy.unique(target_train))
        self.best_class = numpy.array([k for k in cntr.keys()])[numpy.argmax([v for v in cntr.values()])]
        self.best_class_id = numpy.where(self.classes==self.best_class)[0][0]
        return
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        P = numpy.zeros((array.shape[0],len(self.classes)))
        P[:,self.best_class_id] = 1.0
        return P
# ----------------------------------------------------------------------------------------------------------------------