from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D
from keras.optimizers import SGD
from keras import backend as K
import numpy
import cv2
import os
from os import listdir
import fnmatch
# ----------------------------------------------------------------------------------------------------------------------
K.set_image_dim_ordering('th')
import tools_IO
import tools_CNN_view
# ----------------------------------------------------------------------------------------------------------------------
#https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
path_data = '../_weights/vgg16_weights.h5'
# ----------------------------------------------------------------------------------------------------------------------
class CNN_VGG16_Keras():
    def __init__(self):
        self.name = 'CNN_VGG16_Keras'
        self.input_shape = (224, 224)
        self.nb_classes = 4096
        self.class_names = tools_CNN_view.class_names
        self.build()
        return

    def build(self):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, self.input_shape[0], self.input_shape[1])))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        model.load_weights(path_data)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        self.model = model
        return
# ---------------------------------------------------------------------------------------------------------------------
    def save_model(self, filename_output):
        self.model.save(filename_output)
        return
    # ----------------------------------------------------------------------------------------------------------------
    def generate_features(self, path_input, path_output,limit=1000000,mask='*.png'):

        if not os.path.exists(path_output):
            os.makedirs(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            feature_filename = path_output + each + '.txt'
            features = []

            if not os.path.isfile(feature_filename):
                for i in range(0, local_filenames.shape[0]):
                    image = cv2.imread(path_input + each + '/' + local_filenames[i])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
                    image = image.transpose((2, 0, 1))
                    image = numpy.expand_dims(image,axis=0)

                    feature = Model(inputs=self.model.input, outputs=self.model.get_layer('dropout_2').output).predict(image)
                    features.append(feature[0])

                features = numpy.array(features)

                mat = numpy.zeros((features.shape[0],features.shape[1]+1)).astype(numpy.str)
                mat[:, 0] = local_filenames
                mat[:,1:]=features
                tools_IO.save_mat(mat,feature_filename,fmt='%s',delim='\t')

        return
# ---------------------------------------------------------------------------------------------------------------------
    def predict_classes(self, path_input, filename_output, limit=1000000,mask='*.png'):

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]

            for i in range(0, local_filenames.shape[0]):
                image = cv2.imread(path_input + each + '/' + local_filenames[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
                image = image.transpose((2, 0, 1))
                image = numpy.expand_dims(image, axis=0)

                prob = self.model.predict(image)
                prob = prob[0]
                idx = numpy.argsort(-prob)[0]
                label = self.class_names[idx]
                tools_IO.save_labels(path_input + each + '/' + filename_output, numpy.array([local_filenames[i]]),numpy.array([label]), append=i, delim=' ')


        return
# ---------------------------------------------------------------------------------------------------------------------
