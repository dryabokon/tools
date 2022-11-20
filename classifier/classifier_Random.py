import numpy
from collections import Counter
# --------------------------------------------------------------------------------------------------------------------
class classifier_Random(object):
    def __init__(self):
        self.name = "Random"
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):
        self.model = []
        cntr = Counter(target_train)
        self.classes = numpy.sort(numpy.unique(target_train))
        self.p = numpy.array([cntr[c]/target_train.shape[0] for c in self.classes])
        return
#----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        P = numpy.zeros((array.shape[0],len(self.classes)))
        Y = numpy.random.choice(len(self.classes),P.shape[0],p=self.p)

        for i,y in enumerate(Y):
            P[i,y] = 1.0
        return P
# ----------------------------------------------------------------------------------------------------------------------