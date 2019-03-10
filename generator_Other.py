import numpy
import tools_IO
from sklearn.datasets import make_blobs,make_multilabel_classification
# --------------------------------------------------------------------------------------------------------------------
class generator_Other(object):
    def __init__(self, dim=2):
        self.dim = dim
        return

    def create_pos_neg_samples(self,filename_pos,filename_neg):

        #x,y = make_blobs(n_samples=100, n_features=self.dim, centers=self.dim, cluster_std=1.0,shuffle=True, random_state=None)
        x,y = make_multilabel_classification(n_samples=100, n_features=self.dim, n_classes=self.dim); y = y[:,0]

        tools_IO.save_data_to_feature_file_float(filename_pos, x[y >  0], y[y >  0])
        tools_IO.save_data_to_feature_file_float(filename_neg, x[y <= 0], y[y <= 0])

        return
# ---------------------------------------------------------------------------------------------------------------------

    def create_multi_samples(self, folder_output, num_classes):

        x,y = make_blobs(n_samples=200, n_features=self.dim, centers=num_classes)
        for i in range(0,num_classes):
            tools_IO.save_data_to_feature_file_float(folder_output+'%02d.txt'%i, x[y == i], y[y == i])


        return
# ---------------------------------------------------------------------------------------------------------------------