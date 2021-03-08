import numpy as numpy
import tensorflow as tf
import cv2
import os
from os import listdir
import fnmatch
import tensorflow_hub as hub
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_CNN_view
# ----------------------------------------------------------------------------------------------------------------------
class CNN_Resnet_TF_Hub():
    def __init__(self):
        self.name = 'CNN_ResNet_TF'
        self.dest_directory = '../_weights/resnet_v2_50/'
        self.maybe_download_and_extract()
        self.input_shape = hub.get_expected_image_size(self.module)
        self.nb_classes = 2048
        self.class_names = tools_CNN_view.class_names
        var = self.module.variables
        input_info_dict = self.module.get_input_info_dict()
        output_info_dict = self.module.get_output_info_dict()
        signature_names = self.module.get_signature_names()
        i=0
        return
# ---------------------------------------------------------------------------------------------------------------------
    def maybe_download_and_extract(self):
        if not os.path.exists(self.dest_directory):
            os.makedirs(self.dest_directory)
            self.module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1")

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            self.module.export(self.dest_directory, session=sess)
            sess.close()

        else:
            self.module = hub.Module(self.dest_directory)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def generate_features(self, path_input, path_output,limit=1000000,mask='*.png'):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        if not os.path.exists(path_output):
            os.makedirs(path_output)
        else:
            tools_IO.remove_files(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            feature_filename = path_output + each + '.txt'
            features = []

            if not os.path.isfile(feature_filename):
                for i in range (0,local_filenames.shape[0]):
                    image= cv2.imread(path_input + each + '/' + local_filenames[i])
                    image = cv2.resize(image,(self.input_shape[0],self.input_shape[1]))
                    image = numpy.array([image]).astype(numpy.float32)/255
                    feature = self.module(image).eval(session=sess)
                    features.append(feature[0])

                features = numpy.array(features)

                mat = numpy.zeros((features.shape[0], features.shape[1] + 1)).astype(numpy.str)
                mat[:, 0] = local_filenames
                mat[:, 1:] = features
                tools_IO.save_mat(mat, feature_filename, fmt='%s', delim='\t')
        sess.close()
        return

# ---------------------------------------------------------------------------------------------------------------------
    def predict_classes(self, path_input, filename_output, limit=1000000,mask='*.png'):

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())



        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            for i in range(0, local_filenames.shape[0]):
                image = cv2.imread(path_input + each + '/' + local_filenames[i])
                image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
                image = numpy.array([image]).astype(numpy.float32)/255
                outputs = self.module(dict(images=image), signature="image_classification",as_dict=True)
                prob = outputs["default"].eval(session=sess)[0]
                idx = numpy.argsort(-prob)[0]
                label = self.class_names[idx]

                tools_IO.save_labels(path_input+each+'/'+filename_output, numpy.array([local_filenames[i]]), numpy.array([label]), append=i, delim=' ')

        sess.close()
        return

# ---------------------------------------------------------------------------------------------------------------------
