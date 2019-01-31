import numpy
import tensorflow as tf
from sklearn.externals import joblib
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Input, Dense, Dropout, Flatten, Activation
from keras.optimizers import SGD
# --------------------------------------------------------------------------------------------------------------------
#K.set_image_dim_ordering('th')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# --------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_FC(object):
    def __init__(self):
        self.name = "FC"
# ----------------------------------------------------------------------------------------------------------------
    def init_Model0(self, input_shape, num_classes):  # supports fit_generator

        initial_lr = 0.0001
        dropout = 0.5
        dense = 1024 * 10

        layer_002 = Input(input_shape)
        if layer_002.shape.ndims == 2:
            layer_003 = Dense(units=dense, activation='relu')(layer_002)
        else:
            layer_003 = Dense(units=dense, activation='relu')(Flatten()(layer_002))
        layer_004 = BatchNormalization()(layer_003)
        layer_005 = Activation('relu')(layer_004)
        layer_006 = Dropout(dropout)(layer_005)

        layer_007 = Dense(dense)(layer_006)
        layer_008 = BatchNormalization()(layer_007)
        layer_009 = Activation('relu')(layer_008)
        layer_010 = Dropout(dropout)(layer_009)

        layer_010 = Dense(units=num_classes, activation='softmax')(layer_010)
        layer_011 = BatchNormalization()(layer_010)
        layer_012 = Activation('softmax')(layer_011)

        self.model = Model(inputs=layer_002, outputs=layer_012)
        self.model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model

# ----------------------------------------------------------------------------------------------------------------
    def init_Model(self, input_shape, num_classes):  # supports fit_generator

        layer_001 = Input(input_shape)
        layer_002 = BatchNormalization()(layer_001)

        if layer_002.shape.ndims == 2:
            layer_out = Dense(units=num_classes,activation='softmax')(layer_002)
        else:
            layer_out = Dense(units=num_classes,activation='softmax')(Flatten()(layer_002))


        self.model = Model(inputs=layer_001, outputs=layer_out)
        self.model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model

# ----------------------------------------------------------------------------------------------------------------
    def learn_on_arrays(self,data_train, target_train):


        epochs_per_lr = 100

        target_train = IO.to_categorical(target_train)
        M = data_train.shape[0]
        idx_train = numpy.sort(numpy.random.choice(M, int(1*M/2), replace=False))
        idx_valid = numpy.array([x for x in range(0, M) if x not in idx_train])

        self.model = self.init_Model([data_train.shape[1]],target_train.shape[1])


        hist = self.model.fit(data_train[idx_train], target_train[idx_train], nb_epoch=epochs_per_lr, verbose=0,validation_data=(data_train[idx_valid], target_train[idx_valid]))
        acc = hist.history['val_acc'][-1]
        prob = self.predict_probability_of_array(data_train[idx_valid])
        label_pred = numpy.argmax(prob, axis=1)
        label_fact = IO.from_categorical(target_train[idx_valid])
        matches = numpy.array([1*(label_fact[i]==label_pred[i]) for i in range(0,label_pred.shape[0])]).astype(int)
        acc2 = float(numpy.average(matches))
        return acc2
# ----------------------------------------------------------------------------------------------------------------------
    def learn_on_tensors(self, X_train_4d, Y_train_2d, output_path_models=None):
        data_array = X_train_4d.reshape((X_train_4d.shape[0], X_train_4d.shape[1] * X_train_4d.shape[2])).astype('float32')
        target_array = IO.from_categorical(Y_train_2d)
        self.learn_on_arrays(data_array, target_array)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def learn_on_features_file(self, file_train, delimeter='\t', output_path_models=None):
        X_train = (IO.load_mat(file_train, numpy.chararray,delimeter))
        Y_train = X_train[:, 0].astype('float32')
        X_train = (X_train[:, 1:]).astype('float32')
        self.learn_on_arrays(X_train, Y_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_tensor(self, X_test_4d):
        array = X_test_4d.reshape((X_test_4d.shape[0], X_test_4d.shape[1] * X_test_4d.shape[2]))
        return self.predict_probability_of_array(array)
# ----------------------------------------------------------------------------------------------------------------------
    def predict_probability_of_array(self,array):
        return self.model.predict(array,verbose=0)
# ----------------------------------------------------------------------------------------------------------------------
