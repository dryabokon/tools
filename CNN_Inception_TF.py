# ----------------------------------------------------------------------------------------------------------------------
import os.path
from os import listdir
import sys
import tarfile
import numpy
import fnmatch
import tensorflow as tf
from tensorflow.python.platform import gfile
import urllib
import progressbar
# --------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_CNN_view
# --------------------------------------------------------------------------------------------------------------------
class CNN_Inception_TF(object):
    def __init__(self):
        self.name = 'CNN_Inception_TF'
        self.input_shape = (299,299)
        self.model = []
        self.DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        self.dest_directory = '../_weights/inception_v3/'
        self.class_names = tools_CNN_view.class_names

        CNN_Inception_TF.maybe_download_and_extract(self)

        with gfile.FastGFile(self.dest_directory+ 'classify_image_graph_def.pb', 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
            self.tensor_bottleneck, self.tensor_jpeg_data, self.tensor_resized_input,self.tensor_classification  = (tf.import_graph_def(self.graph_def, name='', return_elements=['pool_3/_reshape:0', 'DecodeJpeg/contents:0', 'ResizeBilinear:0','softmax:0']))
        return
# ----------------------------------------------------------------------------------------------------------------
    def maybe_download_and_extract(self):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
            return

        if not os.path.exists(self.dest_directory):
            os.makedirs(self.dest_directory)

        filename = self.DATA_URL.split('/')[-1]
        filepath = self.dest_directory+filename
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(self.DATA_URL,filepath,_progress)
            print()
            print('Successfully downloaded', filename, os.stat(filepath).st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(self.dest_directory)
        return

# ----------------------------------------------------------------------------------------------------------------------
    def generate_features(self, path_input, path_output,mask='*.png',limit=1000000):

        if not os.path.exists(path_output):
            os.makedirs(path_output)
        else:
            tools_IO.remove_files(path_output)
            tools_IO.remove_folders(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)


        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            feature_filename = path_output + each + '.txt'
            features = []

            if not os.path.isfile(feature_filename):
                bar = progressbar.ProgressBar(max_value=len(local_filenames))
                for b, local_filename in enumerate(local_filenames):
                    bar.update(b)
                    image_data = gfile.FastGFile(path_input + each + '/' + local_filename, 'rb').read()
                    feature = sess.run(self.tensor_bottleneck, {self.tensor_jpeg_data: image_data})[0]
                    features.append(feature)

                features = numpy.array(features)

                mat = numpy.zeros((features.shape[0], features.shape[1] + 1)).astype(numpy.str)
                mat[:, 0] = local_filenames
                mat[:, 1:] = features
                tools_IO.save_mat(mat, feature_filename, fmt='%s', delim='\t')


        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict_classes(self, path_input, filename_output, limit=1000000,mask='*.png'):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            for i in range(0, local_filenames.shape[0]):
                image_data = gfile.FastGFile(path_input + each + '/' + local_filenames[i], 'rb').read()
                prob = sess.run(self.tensor_classification, {self.tensor_jpeg_data: image_data})[0][:1000]
                idx = numpy.argsort(-prob)[0]
                label = self.class_names[idx]
                tools_IO.save_labels(path_input + each + '/' + filename_output, numpy.array([local_filenames[i]]),numpy.array([label]), append=i, delim=' ')

        sess.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
