import numpy
import tools_IO
from sklearn.datasets import make_blobs,make_multilabel_classification
# --------------------------------------------------------------------------------------------------------------------
class generator_TS(object):
    def __init__(self, dim=2,len=20):
        self.dim = dim
        self.len = len
        return
# --------------------------------------------------------------------------------------------------------------------
    def generate_linear(self, filename_output,noise_signal_ratio= 0.0):

        X = numpy.random.rand(self.len,self.dim)
        Y = numpy.zeros(self.len,dtype=numpy.float32)
        A = numpy.random.rand(self.dim+1)
        for i in range(0,self.dim):
            Y[:]=X[:,i]*A[i]
        Y+=A[-1]

        noise = self.generate_noise(noise_signal_ratio)

        Y+=noise

        mat = numpy.vstack((Y,X.T)).T
        tools_IO.save_mat(mat,filename_output)

        return mat
# ---------------------------------------------------------------------------------------------------------------------
    def generate_sine(self, filename_output,max_periods=3,noise_signal_ratio= 0.0):

        numpy.random.seed(125)
        max_periods = max_periods

        X = numpy.random.rand(self.len,self.dim)
        Y = numpy.zeros(self.len,dtype=numpy.float32)
        A = numpy.random.rand(self.dim+1)
        ph = 2*numpy.pi*numpy.random.rand(self.dim)

        for i in range(0,self.dim):
            T = 0.5+0.5*numpy.random.rand(1)[0]
            X[:,i] = T*numpy.arange(self.len)/self.len

        periods = 1+max_periods*numpy.random.rand(self.dim)

        for i in range(0, self.dim):
            Y[:]=A[i]*numpy.sin(ph[i]+2*numpy.pi*X[:,i]*periods[i])

        Y+=A[-1]

        noise = self.generate_noise(noise_signal_ratio)

        Y+=noise

        mat = numpy.vstack((Y,X.T)).T
        tools_IO.save_mat(mat,filename_output)

        return mat
# ---------------------------------------------------------------------------------------------------------------------
    def generate_noise(self,noise_signal_ratio= 0.0):
        noise = 2*(numpy.random.rand(self.len)-0.5)*noise_signal_ratio
        return noise

# ---------------------------------------------------------------------------------------------------------------------