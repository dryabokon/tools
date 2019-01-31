import numpy as numpy
import tensorflow as tf
import cv2
import os
from os import listdir
import fnmatch
import tensorflow_hub as hub
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
class CNN_Faster_TF():
    def __init__(self):
        self.name = 'CNN_Faster_TF'
        self.dest_directory = '../_weights/faster/'
        self.maybe_download_and_extract()
        self.input_shape = [224,224]#hub.get_expected_image_size(self.module)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def maybe_download_and_extract(self):
        if not os.path.exists(self.dest_directory):
            os.makedirs(self.dest_directory)
            self.module = hub.Module("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            self.module.export(self.dest_directory, session=sess)
            sess.close()

        else:
            self.module = hub.Module(self.dest_directory)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def detect_object(self, filename_input):

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(tf.tables_initializer())

        image = cv2.imread(filename_input)
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        image = numpy.array([image]).astype(numpy.float32) / 255
        result = self.module(image,as_dict=True)
        boxes = result['detection_boxes'].eval(session=sess)
        entts = result['detection_class_entities'].eval(session=sess)
        labls = result['detection_class_labels'].eval(session=sess)
        names = result['detection_class_names'].eval(session=sess)
        scors = result['detection_scores'].eval(session=sess)

        sess.close()

        return

# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
