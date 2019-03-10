# ----------------------------------------------------------------------------------------------------------------------
import os.path
from os import listdir
import numpy
import fnmatch
import cv2
# --------------------------------------------------------------------------------------------------------------------
from keras.models import Model
from keras.applications import MobileNet
from keras.applications.xception import Xception
from keras.applications.mobilenet import preprocess_input
import tools_IO
import tools_CNN_view
# --------------------------------------------------------------------------------------------------------------------
class CNN_App_Keras(object):
    def __init__(self):
        self.name = 'CNN_App_Keras'
        self.input_shape = (224,224)
        #self.model = Xception()
        self.model = MobileNet()
        self.class_names = tools_CNN_view.class_names

        return
# ----------------------------------------------------------------------------------------------------------------------
    def generate_features(self, path_input, path_output,mask='*.png',limit=1000000):

        if not os.path.exists(path_output):
            os.makedirs(path_output)
        else:
            tools_IO.remove_files(path_output)
            tools_IO.remove_folders(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))


        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            feature_filename = path_output + each + '.txt'
            features = []

            if not os.path.isfile(feature_filename):
                for i in range (0,local_filenames.shape[0]):
                    img = cv2.imread(path_input + each + '/' + local_filenames[i])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.input_shape).astype(numpy.float32)
                    model = Model(inputs=self.model.input, outputs=self.model.get_layer('global_average_pooling2d_1').output)
                    feature = model.predict(preprocess_input(numpy.array([img])))[0]
                    features.append(feature)

                features = numpy.array(features)

                mat = numpy.zeros((features.shape[0], features.shape[1] + 1)).astype(numpy.str)
                mat[:, 0] = local_filenames
                mat[:, 1:] = features
                tools_IO.save_mat(mat, feature_filename, fmt='%s', delim='\t')


        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict_classes(self, path_input, filename_output, limit=1000000,mask='*.png'):

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            for i in range(0, local_filenames.shape[0]):
                img = cv2.imread(path_input + each + '/' + local_filenames[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.input_shape).astype(numpy.float32)

                model = Model(inputs=self.model.input, outputs=self.model.output)
                prob = model.predict(preprocess_input(numpy.array([img])))[0]

                idx = numpy.argsort(-prob)[0]
                label = self.class_names[idx]
                tools_IO.save_labels(path_input + each + '/' + filename_output, numpy.array([local_filenames[i]]),numpy.array([label]), append=i, delim=' ')

        return
# ----------------------------------------------------------------------------------------------------------------------
