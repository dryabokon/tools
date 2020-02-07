import numpy
import cv2
import os
import tensorflow as tf
import numpy as np
import re
from tensorflow.python.platform import gfile
import tools_IO
import progressbar
# ---------------------------------------------------------------------------------------------------------------------
def preprocess(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y
# ---------------------------------------------------------------------------------------------------------------------
def load_model_sess(model, sess, input_map=None):

    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    return
# ---------------------------------------------------------------------------------------------------------------------
class face_descriptor:

    def __init__(self,weights= "../_weights/facenet/20180402-114759.pb"):
        self.name = 'facenet'
        self.weights = weights
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.sess = tf.Session()
        load_model_sess(self.weights, self.sess)
        self.images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        self.embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

        return
    # ---------------------------------------------------------------------------------------------------------------------
    def __inference(self, img_list, sess=tf.get_default_session()):
        feed_dict = {self.images_placeholder: img_list, self.phase_train_placeholder: False}
        emb = sess.run(self.embeddings, feed_dict=feed_dict)
        return emb
    # ---------------------------------------------------------------------------------------------------------------------
    def get_embedding(self, img):
        pre_whitened = preprocess(cv2.resize(img, (160, 160)))
        feature = self.__inference([pre_whitened], self.sess)
        return feature[0]
# ---------------------------------------------------------------------------------------------------------------------
    def generate_features(self, path_input, path_output, limit=1000000, mask='*.png,*.jpg'):

        if not os.path.exists(path_output):
            os.makedirs(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = tools_IO.get_filenames(path_input + each, mask)[:limit]
            feature_filename = path_output + '/' + each + '_' + self.name + '.txt'
            features, filenames = [], []

            if not os.path.isfile(feature_filename):
                bar = progressbar.ProgressBar(max_value=len(local_filenames))
                for b, local_filename in enumerate(local_filenames):
                    bar.update(b)
                    image = cv2.imread(path_input + each + '/' + local_filename)
                    if image is None: continue

                    pre_whitened = preprocess(cv2.resize(image, (160, 160)))
                    feature = self.__inference([pre_whitened], self.sess)

                    features.append(feature[0])
                    filenames.append(local_filename)

                features = numpy.array(features)

                mat = numpy.zeros((features.shape[0], features.shape[1] + 1)).astype(numpy.str)
                mat[:, 0] = filenames
                mat[:, 1:] = features
                tools_IO.save_mat(mat, feature_filename, fmt='%s', delim='\t')

        return
# ---------------------------------------------------------------------------------------------------------------------