import math
import os
from os import listdir
import fnmatch
import numpy
import math
from keras.models import Model, load_model, Sequential
from keras.layers import BatchNormalization, Input, Dense, Dropout, Flatten, Activation, InputLayer, Conv2DTranspose,UpSampling2D, LeakyReLU, Softmax, Concatenate
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D, Conv3D, AveragePooling2D

from keras.models import load_model
from keras.callbacks import EarlyStopping
import cv2
# --------------------------------------------------------------------------------------------------------------------
#K.set_image_dim_ordering('th')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# --------------------------------------------------------------------------------------------------------------------
import tools_YOLO
import tools_draw_numpy
import tools_IO
import tools_CNN_view
import tools_image
# --------------------------------------------------------------------------------------------------------------------
class detector_Zebra(object):
    def __init__(self,filename_weights=None):
        self.name = "detector_Zebra"
        self.model = []
        self.input_shape = (600,800,3)
        self.verbose = 2
        self.epochs = 100
        self.init_model()
        if filename_weights is not None:
            self.load_model(filename_weights)
# ----------------------------------------------------------------------------------------------------------------
    def save_model(self,filename_output):
        self.model.save(filename_output)
        return
# ----------------------------------------------------------------------------------------------------------------
    def load_model(self, filename_weights):
        self.model = load_model(filename_weights)
        return
# ----------------------------------------------------------------------------------------------------------------
    def init_model(self):
        return self.init_model_zebra_2()
# ----------------------------------------------------------------------------------------------------------------
    def init_model_16_corners(self):

        self.model = Sequential()
        self.model.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='valid', input_shape=self.input_shape))
        self.model.layers[0].set_weights(tools_CNN_view.construct_filters_2x2())


        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return
# ----------------------------------------------------------------------------------------------------------------
    def init_model_zebra_1(self):
        self.model = Sequential()
        n=4
        h,w = 8,16

        self.model.add(Conv2D(filters=n, kernel_size=(h, w), strides=(2, 2), padding='same', input_shape=self.input_shape))
        self.model.layers[0].set_weights(tools_CNN_view.construct_filters(n,h,w))
        self.model.add(MaxPooling2D(pool_size=(h//2, w//2), strides=(2,2), padding='same'))
        self.model.add(AveragePooling2D(pool_size=(h , h*4), strides=(2, 2), padding='same'))
        self.model.add(Dense(1))
        weights = self.model.layers[-1].get_weights()
        for i in range(len(weights[0])):weights[0][i] = 1
        self.model.layers[-1].set_weights(weights)
        self.model.layers[-1].trainable = True

        self.model.add(Activation('relu'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.output_shape =self.model.output.shape

        return
# ----------------------------------------------------------------------------------------------------------------
    def init_model_zebra_2(self):
        layer_input = Input(shape=(self.input_shape))
        h1, w1 = 4, 16
        h2, w2 = 8, 32

        layer_1_01 = Conv2D(filters=4, kernel_size=(h1, w1), strides=(2, 2), padding='same')(layer_input)
        layer_1_02 = MaxPooling2D(pool_size=(h1, w1) , strides=(2, 2), padding='same')(layer_1_01)
        layer_1_03 = Activation('relu')(layer_1_02)

        #layer_2_01 = Conv2D(filters=4, kernel_size=(h2, w2), strides=(2, 2), padding='same')(layer_input)
        #layer_2_02 = MaxPooling2D(pool_size=(h2, w2) , strides=(2, 2), padding='same')(layer_2_01)
        #layer_2_03 = Activation('relu')(layer_2_02)


        #layer_3_01 = Concatenate()([layer_1_03, layer_2_03])
        layer_3_02 = Dense(1)(layer_1_03)
        #layer_3_03 = Activation('relu')(layer_3_02)

        self.model = Model(inputs=layer_input, outputs=layer_3_02)

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.output_shape = self.model.output.shape

        self.model.layers[1].set_weights(tools_CNN_view.construct_filters(4, h1, w1))


        return
# ----------------------------------------------------------------------------------------------------------------
    def process_file(self, filename_in, filename_out):

        if not os.path.isfile(filename_in):
            return []

        image = cv2.imread(filename_in)
        if image is None:
            return []

        image = cv2.resize(image,(self.input_shape[1], self.input_shape[0]))
        res = self.process_image(image)

        cv2.imwrite(filename_out,res)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_image(self, image):

        image = tools_CNN_view.normalize(image.astype(numpy.float32))
        res = self.model.predict(numpy.array([image]))

        return res[0]
# ----------------------------------------------------------------------------------------------------------------------
    def learn(self, file_annotations, folder_out, folder_annotation=None, limit=1000000):

        if folder_annotation is not None:
            foldername = folder_annotation
        else:
            foldername = '/'.join(file_annotations.split('/')[:-1]) + '/'

        with open(file_annotations) as f:lines = f.readlines()[1:limit]
        list_filenames = sorted(set([foldername+line.split(' ')[0] for line in lines]))
        true_boxes = tools_YOLO.get_true_boxes(foldername, file_annotations, delim =' ')


        X = []
        Y = []
        for filename,boxes in zip(list_filenames,true_boxes):
            if not os.path.isfile(filename): continue
            image = cv2.imread(filename)
            if image is None: continue
            image_resized = cv2.resize(image,(self.input_shape[1], self.input_shape[0]))
            y = numpy.zeros((self.output_shape[1],self.output_shape[2],3),dtype=numpy.float32)
            for box in boxes:
                row_up, col_left, row_down, col_right = box[1], box[0], box[3], box[2]
                row_up*=y.shape[0]/image.shape[0]
                row_down*=y.shape[0] / image.shape[0]
                col_left *= y.shape[1] / image.shape[1]
                col_right *= y.shape[1] / image.shape[1]

                row_up = int(row_up)
                row_down = int(row_down)
                col_left = int(col_left)
                col_right = int(col_right)

                if abs(row_up - row_down)<1:row_up-=1
                if abs(col_left - col_right) < 1: col_left -= 1
                cv2.rectangle(y,(col_left,row_up), (col_right,row_down), [255,255,255], -1)

            X.append(image_resized)
            Y.append(numpy.expand_dims(y[:,:,0],-1))

        X = numpy.array(X, dtype=numpy.float32)
        Y = numpy.array(Y, dtype=numpy.float32)

        self.model.summary()

        cv2.imwrite(folder_out+'Y.png',Y[0])

        early_stopping_monitor = EarlyStopping(monitor='loss', patience=10)
        self.model.fit(x=X,y=Y,verbose=2,epochs=self.epochs,callbacks=[early_stopping_monitor])
        self.save_model(folder_out + 'A_model.h5')
        return
# ----------------------------------------------------------------------------------------------------------------------