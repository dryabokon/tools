#graphwiz!!
from sklearn.ensemble import RandomForestClassifier
import numpy
import tools_IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_RF(object):
    def __init__(self):
        self.name = "RF"
        self.max_depth = 3
        self.model = RandomForestClassifier(max_depth=self.max_depth,n_estimators=50)
# ----------------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (-1,X.shape[0]))
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self,data_train, target_train):
        self.model.fit(self.maybe_reshape(data_train), target_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def learn_file_batch(self, filename_train, has_header=True,n_epochs=1, batch_size=20,verbose=False):
        self.model = RandomForestClassifier(warm_start=True, n_estimators=1,max_depth=self.max_depth)
        N = tools_IO.count_lines(filename_train)

        splits = []
        for s in range(N // batch_size):
            splits.append(numpy.arange(s*batch_size,(s+1)*batch_size,1))

        for e in range(n_epochs):
            for b,split in enumerate(splits):
                if verbose: print('Batch %03d/%d'%(b,len(splits)))
                data = numpy.array(tools_IO.get_lines(filename_train, start=has_header*1 +split[0], end=has_header*1+split[-1]))
                self.model.fit(self.maybe_reshape(data[:,1:]), data[:,0])
                self.model.n_estimators += 1
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        return self.model.predict_proba(self.maybe_reshape(array))
# ----------------------------------------------------------------------------------------------------------------------