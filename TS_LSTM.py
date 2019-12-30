import numpy
from keras.models import Model, load_model, Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
# --------------------------------------------------------------------------------------------------------------------
import tools_IO
# --------------------------------------------------------------------------------------------------------------------
class TS_LSTM(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = "TS_LTSM_01"
        self.model = None
        self.verbose = 2
        self.units = 4
        self.epochs = 100
        self.batch_size = 1
        self.folder_debug = folder_debug
        if filename_weights is not None:
            self.load_model(filename_weights)
        return
# ----------------------------------------------------------------------------------------------------------------
    def init_model(self,dims):
        model = Sequential()
        model.add(LSTM(self.units, input_shape=(dims, 1)))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_model(self, filename_weights):
        self.model = load_model(filename_weights)
        return
    # ----------------------------------------------------------------------------------------------------------------
    def save_model(self, filename_output):
        self.model.save(filename_output)
        return
# ----------------------------------------------------------------------------------------------------------------
    def save_stats(self, hist):
        c1 = numpy.array(['accuracy'] + hist.history['acc'])
        c2 = numpy.array(['val_acc'] + hist.history['val_acc'])
        c3 = numpy.array(['loss'] + hist.history['loss'])
        c4 = numpy.array(['val_loss'] + hist.history['val_loss'])

        mat = (numpy.array([c1, c2, c3, c4]).T)

        tools_IO.save_mat(mat, self.folder_debug + self.name + '_learn_rates.txt')
        return

#----------------------------------------------------------------------------------------------------------------------
    def train(self, array_X, array_Y):
        M = array_X.shape[0]
        self.init_model(array_X.shape[1])
        tensor_X = numpy.expand_dims(array_X,2)
        tensor_Y = numpy.expand_dims(array_Y,1)

        early_stopping_monitor = EarlyStopping(monitor='loss', patience=10)

        hist = self.model.fit(tensor_X, tensor_Y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks = [early_stopping_monitor])

        if self.folder_debug is not None:
            #self.save_stats(hist)
            self.save_model(self.folder_debug + self.name + '_model.h5')
        return self.predict(array_X,array_Y)
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array_X,array_Y):
        tensor_X = numpy.expand_dims(array_X, 2)
        res = self.model.predict(tensor_X)
        res = res[:, 0]
        return res
# ----------------------------------------------------------------------------------------------------------------------
