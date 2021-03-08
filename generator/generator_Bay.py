import numpy
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class generator_Bay(object):
    def __init__(self,dim=2):

        if(dim==2):
            self.ctgr0 = ([1  , 2  , 3  , 4  , 5  ,6],[7  ,8  ,  9])
            self.freq0 = ([0.0, 0.2, 0.3, 0.3, 0.0,0.2],[0.3,0.3,0.4])

            self.ctgr1 = ([0  , 2  , 3  , 4  , 8  ],[8  ,9  , 10, 11])
            self.freq1 = ([0.3, 0.0, 0.3, 0.4, 0.0],[0.5,0.3,0.1, 0.1])
        else:
            self.ctgr0 = ([1  , 2  , 3  , 4  , 5  ,6],[7  ,8  ,  9])
            self.freq0 = ([0.0, 0.2, 0.3, 0.3, 0.0,0.2],[0.3,0.3,0.4])

            self.ctgr1 = ([0  , 2  , 3  , 4  , 8  ],[8  ,9  , 10, 11])
            self.freq1 = ([0.3, 0.0, 0.3, 0.4, 0.0],[0.5,0.3,0.1, 0.1])


# ----------------------------------------------------------------------------------------------------------------------
    def create_pos_neg_samples(self,filename_pos,filename_neg):
        N0 = 100
        N1 = 100

        x0 = numpy.random.choice(self.ctgr0[0], N0, p=self.freq0[0])
        y0 = numpy.random.choice(self.ctgr0[1], N0, p=self.freq0[1])
        x1 = numpy.random.choice(self.ctgr1[0], N1, p=self.freq1[0])
        y1 = numpy.random.choice(self.ctgr1[1], N1, p=self.freq1[1])
        X0 = numpy.vstack((x0, y0))
        X1 = numpy.vstack((x1, y1))
        Y0 = numpy.full(N0, -1).astype('float32')
        Y1 = numpy.full(N1, +1).astype('float32')
        IO.save_data_to_feature_file_float(filename_pos,X1.T,Y1)
        IO.save_data_to_feature_file_float(filename_neg,X0.T,Y0)

        return
# ----------------------------------------------------------------------------------------------------------------------