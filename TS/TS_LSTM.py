import numpy
import tensorflow as tf
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
# --------------------------------------------------------------------------------------------------------------------
import tools_IO
# --------------------------------------------------------------------------------------------------------------------
class TS_LSTM(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = "TS_LTSM_01"
        self.model = None
        self.verbose = 2
        self.units = 64
        self.epochs = 100
        self.batch_size = 32
        self.folder_debug = folder_debug
        self.scaler = MinMaxScaler(feature_range=(0, 1))


        if filename_weights is not None:
            self.load_model(filename_weights)
        return
# ----------------------------------------------------------------------------------------------------------------
    def init_model(self):
        # self.model = Sequential()
        # self.model.add(LSTM(self.units, input_shape=(dims, 1)))
        # self.model.add(Dense(1))
        # self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
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
    def train(self, X, Y, lag_perc=0.1):

        self.init_model()
        X_scaled = self.scaler.fit_transform(X)
        self.lag = int(Y.shape[0]*lag_perc)
        tensor_X = numpy.array([X_scaled[i - self.lag:i] for i in range(self.lag, X.shape[0])])
        early_stopping_monitor = EarlyStopping(monitor='loss', patience=10)
        self.model.fit(tensor_X, numpy.expand_dims(Y[self.lag:].flatten(),1), epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks = [early_stopping_monitor])
        train_pred = self.predict(X)
        train_pred = numpy.concatenate(([train_pred[0]]*self.lag,train_pred),axis=0)
        return train_pred
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, X,Y=None,ongoing_retrain=False):
        X_scaled = self.scaler.transform(X)
        tensor_X = numpy.array([X_scaled[i - self.lag:i] for i in range(self.lag, X.shape[0])])

        Y_pred = self.model.predict(tensor_X).flatten()
        return Y_pred
# ----------------------------------------------------------------------------------------------------------------------
