import numpy
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
from sklearn.datasets import make_gaussian_quantiles
# --------------------------------------------------------------------------------------------------------------------
class generator_Gauss(object):
    def __init__(self,dim=2):
        self.N0 = 100
        self.N1 = 100

        if (dim==2):
            self.mean0 = [10, 20]
            self.cov0 = [[10, 5], [5, 10]]
            self.mean1 = [20, 20]
            self.cov1 = [[10, 00], [0, 10]]
        else:
            self.mean0 = numpy.random.choice(5,dim).tolist()
            self.cov0  = numpy.eye(dim)
            self.cov0  +=numpy.random.rand(dim,dim)
            self.cov0  = numpy.dot(self.cov0,self.cov0.transpose())
            self.cov0  = self.cov0.tolist()

            self.mean1 = numpy.random.choice(5,dim).tolist()
            self.cov1  = numpy.eye(dim)
            self.cov1  +=numpy.random.rand(dim,dim)
            self.cov1  = numpy.dot(self.cov1, self.cov1.transpose())
            self.cov1  = self.cov1.tolist()


# ----------------------------------------------------------------------------------------------------------------------

    def create_pos_neg_samples(self,filename_pos,filename_neg):

        X0 = numpy.random.multivariate_normal(self.mean0, self.cov0, self.N0)
        X1 = numpy.random.multivariate_normal(self.mean1, self.cov1, self.N1)
        Y0 = numpy.full(self.N0, -1).astype('float32')
        Y1 = numpy.full(self.N1, +1).astype('float32')

        #X0 = numpy.hstack((X0, Y0))
        #X1 = numpy.hstack((X1, Y1))
        #Y0 = numpy.full(self.N0, -1).astype('float32')
        #Y1 = numpy.full(self.N1, +1).astype('float32')

        IO.save_data_to_feature_file_float(filename_pos,X1,Y1)
        IO.save_data_to_feature_file_float(filename_neg,X0,Y0)

        return
# ----------------------------------------------------------------------------------------------------------------------