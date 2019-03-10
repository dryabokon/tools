import math
import os
from os import listdir
import fnmatch
import numpy
import math
from keras.models import Model, load_model, Sequential
from keras.layers import BatchNormalization, Input, Dense, Dropout, Flatten, Activation, InputLayer, Conv2DTranspose,UpSampling2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D, Conv3D
from keras.models import load_model
from keras.callbacks import EarlyStopping
import cv2
# --------------------------------------------------------------------------------------------------------------------
#K.set_image_dim_ordering('th')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# --------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_CNN_view
import tools_image
# --------------------------------------------------------------------------------------------------------------------
class classifier_FC_Keras(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = "FC"
        self.model = []
        self.verbose = True
        self.epochs = 200
        self.folder_debug = folder_debug
        self.filename_debug_image = 'data/ex-natural/dog/dog_0000.jpg'
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
    def init_model_1d(self, input_shape, num_classes):

        layer_001 = Input(input_shape)
        layer_002 = BatchNormalization()(layer_001)

        if layer_002.shape.ndims == 2:
            layer_out = Dense(units=num_classes,activation='softmax')(layer_002)
        else:
            layer_out = Dense(units=num_classes,activation='softmax')(Flatten()(layer_002))

        self.model = Model(inputs=layer_001, outputs=layer_out)
        self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        return
# ----------------------------------------------------------------------------------------------------------------
    def init_model_2x2(self, input_shape, num_classes):

        n_filters = 64
        n_features =  1*num_classes*num_classes
        n_features = 64

        scale = int(math.sqrt(n_filters) * math.sqrt(input_shape[0] * input_shape[1] / n_features))
        d1 =  int(math.sqrt(scale))
        d2 =  int(scale/d1)

        self.S1 =  int(math.sqrt(d1))
        self.S2 =  int(d1/self.S1)

        self.S3 =  int(math.sqrt(d2))
        self.S4 =  int(d2/self.S3)


        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=input_shape))

        self.model.add(Conv2D(n_filters, (2, 2), activation='relu', strides=(self.S1, self.S1),padding='same',data_format='channels_last'))
        self.model.add(MaxPooling2D(     (2, 2), strides=(self.S2, self.S2)))

        N = self.model.layers[-1].output.get_shape().as_list()[3]
        self.model.add(Conv2D(N,         (2, 2), activation='relu',strides=(self.S3, self.S3),padding='same',data_format='channels_last'))
        self.model.add(MaxPooling2D(     (2, 2), strides=(self.S4, self.S4)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        N = numpy.prod(self.model.layers[-3].output.get_shape().as_list()[1:])
        self.model.add(Dense(N, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

        tools_CNN_view.visualize_layers(self.model, self.filename_debug_image, self.folder_debug)
        self.model.summary()
        return

    # ----------------------------------------------------------------------------------------------------------------
    def init_model_16_24(self, input_shape, num_classes):
        n_filters = 32*3



        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=input_shape))

        self.model.add(Conv2D(n_filters, (2, 2), activation='relu', strides=(2, 2), padding='same',data_format='channels_last'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        N = self.model.layers[-1].output.get_shape().as_list()[3]
        self.model.add(Conv2D(N, (2, 2), activation='relu', strides=(2, 2), padding='same',data_format='channels_last'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        N = numpy.prod(self.model.layers[-3].output.get_shape().as_list()[1:])
        self.model.add(Dense(int(N/4), activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        tools_CNN_view.visualize_layers(self.model, self.filename_debug_image, self.folder_debug)
        self.model.summary()
        return
    # ----------------------------------------------------------------------------------------------------------------
    def init_model(self, input_shape, num_classes):#_alexNet

        n_filters = 96

        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=input_shape))

        self.model.add(Conv2D(n_filters, (11, 11), activation='relu', strides=(2, 2),padding='same',data_format='channels_last'))
        self.model.add(MaxPooling2D(     (2, 2)))

        N = self.model.layers[-1].output.get_shape().as_list()[3]
        self.model.add(Conv2D(N,         (5, 5), activation='relu',strides=(2, 2),padding='same',data_format='channels_last'))
        self.model.add(MaxPooling2D(     (2, 2),strides=(2, 2)))

        N = self.model.layers[-1].output.get_shape().as_list()[3]
        self.model.add(Conv2D(N,         (3, 3), activation='relu',strides=(1, 1),padding='same',data_format='channels_last'))
        self.model.add(MaxPooling2D(     (2, 2),strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(numpy.prod(self.model.layers[-3].output.get_shape().as_list()[1:]), activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))

        weights = tools_CNN_view.import_weight('data/filter1.png',(11,11,3,96))[:,:,:,:n_filters]
        self.model.layers[1].set_weights([weights, tools_CNN_view.import_bias()])
        self.model.layers[1].trainable = True

        self.model.summary()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        tools_CNN_view.visualize_filters(self.model,self.folder_debug)
        tools_CNN_view.visualize_layers(self.model, self.filename_debug_image, self.folder_debug)

        return

    # ----------------------------------------------------------------------------------------------------------------

    def learn(self,data_train, target_train):

        num_classes = numpy.unique(target_train).shape[0]

        if len(data_train.shape)==4:
            self.init_model(data_train.shape[1:],num_classes)
        else:
            self.init_model_1d(data_train.shape[1:], num_classes)

        data_train=tools_CNN_view.normalize(data_train.astype(numpy.float32))
        target_train = tools_IO.to_categorical(target_train)

        M = data_train.shape[0]
        idx_train = numpy.sort(numpy.random.choice(M, int(1*M/2), replace=False))
        idx_valid = numpy.array([x for x in range(0, M) if x not in idx_train])

        early_stopping_monitor = EarlyStopping(monitor='acc', patience=10)

        hist = self.model.fit(data_train[idx_train].astype(numpy.float32), target_train[idx_train], nb_epoch=self.epochs,
                              verbose=self.verbose,validation_data=(data_train[idx_valid], target_train[idx_valid]), callbacks=[early_stopping_monitor])
        if self.folder_debug is not None:
            self.save_stats(hist)
            self.save_model(self.folder_debug+self.name+'_model.h5')

        acc = hist.history['val_acc'][-1]
        prob = self.predict(data_train[idx_valid])
        label_pred = numpy.argmax(prob, axis=1)
        label_fact = tools_IO.from_categorical(target_train[idx_valid])
        matches = numpy.array([1*(label_fact[i]==label_pred[i]) for i in range(0,label_pred.shape[0])]).astype(int)
        acc2 = float(numpy.average(matches))




        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self,array):
        return self.model.predict(array,verbose=0)
# ----------------------------------------------------------------------------------------------------------------------
    def images_to_features(self,images):
        feature_layer = -2

        if images.shape[1]!=self.model.input_shape[1] or images.shape[2]!=self.model.input_shape[2]:
            images[:] = cv2.resize(images[:], (self.model.input_shape[1], self.model.input_shape[2]))

        features = Model(inputs=self.model.input, outputs=self.model.layers[feature_layer].output).predict(numpy.array(images))
        return features

# ----------------------------------------------------------------------------------------------------------------------
    def generate_features(self, path_input, path_output, limit=1000000, mask='*.png'):

        feature_layer = -2

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
                    image = cv2.resize(image, (self.model.input_shape[1], self.model.input_shape[2]))
                    feature = Model(inputs=self.model.input, outputs=self.model.layers[feature_layer].output).predict(numpy.array([image]))
                    features.append(feature[0])

                features = numpy.array(features)

                mat = numpy.zeros((features.shape[0], features.shape[1] + 1)).astype(numpy.str)
                mat[:, 0] = local_filenames
                mat[:, 1:] = features
                tools_IO.save_mat(mat, feature_filename, fmt='%s', delim='\t')


        return
# ----------------------------------------------------------------------------------------------------------------------
    def feature_to_layers(self, feature,path_output):
        orig_layers = self.model.layers

        invmodel = Sequential()

        invmodel = tools_CNN_view.add_de_conv_layer(invmodel, orig_layers[3],crop=1,input_shape=feature.shape)
        invmodel = tools_CNN_view.add_de_pool_layer(invmodel, orig_layers[2])
        invmodel = tools_CNN_view.add_de_conv_layer(invmodel, orig_layers[1])

        outputs = Model(inputs=invmodel.input, outputs=[layer.output for layer in invmodel.layers]).predict(numpy.array([feature]))

        feature_image = tools_CNN_view.tensor_gray_3D_to_image(feature)
        cv2.imwrite(path_output + 'feature.png', feature_image*255.0/feature_image.max())

        tools_CNN_view.stage_tensors(outputs,path_output)


        return
    # ----------------------------------------------------------------------------------------------------------------------
    def save_stats(self,hist):
        c1 = numpy.array(['accuracy']     + hist.history['acc'])
        c2 = numpy.array(['val_acc'] + hist.history['val_acc'])
        c3 = numpy.array(['loss']    + hist.history['loss'])
        c4 = numpy.array(['val_loss']+ hist.history['val_loss'])

        mat = (numpy.array([c1,c2,c3,c4]).T)

        tools_IO.save_mat(mat,self.folder_debug+self.name+'_learn_rates.txt')
        return
# ----------------------------------------------------------------------------------------------------------------------
