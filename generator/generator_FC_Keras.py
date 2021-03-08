import numpy
from keras.models import Model, load_model, Sequential
from keras.layers import BatchNormalization, Input, Dense, Dropout, Flatten, Activation, InputLayer, Reshape, UpSampling2D, Conv2DTranspose,Conv2D
from keras.models import load_model
from keras.callbacks import EarlyStopping

# --------------------------------------------------------------------------------------------------------------------
#K.set_image_dim_ordering('th')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# --------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_CNN_view
import tools_image
# --------------------------------------------------------------------------------------------------------------------
class generator_FC_Keras(object):
    def __init__(self,folder_debug=None,filename_weights=None):
        self.name = "Gen_FC"
        self.model = []
        self.verbose = True
        self.epochs = 1000
        self.folder_debug = folder_debug
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
    def init_model0(self, input_dim,out_dim,grayscale):

        #loss = 'binary_crossentropy'
        #loss = 'mean_absolute_error'
        loss = 'mean_squared_error'

        #activation = 'relu'
        activation = 'linear'

        #optimizer = 'RMSprop'
        #optimizer = 'Nadam'
        optimizer = 'adam'

        self.model = Sequential()
        self.model.add(Dense((out_dim), activation=activation, input_shape=[input_dim]))
        self.model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
        return
# ----------------------------------------------------------------------------------------------------------------
    def init_model(self, input_dim,out_dim,grayscale):

        self.model = Sequential()
        rows = tools_image.numerical_devisor(int(input_dim))
        cols = int(input_dim/rows)

        self.model.add(Reshape((rows, cols, 1),input_shape=[input_dim]))
        self.model.add(Conv2D(32, (2, 2), activation='linear', padding='same'))
        self.model.add(UpSampling2D((2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense((out_dim),activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
        return
# ----------------------------------------------------------------------------------------------------------------
    def learn(self,features, flat_images,grayscale):

        self.init_model(features.shape[1],flat_images.shape[1],grayscale)

        M = features.shape[0]
        idx_train = numpy.sort(numpy.random.choice(M, int(1*M/2), replace=False))
        idx_valid = numpy.array([x for x in range(0, M) if x not in idx_train])

        hist = self.model.fit(features[idx_train].astype(numpy.float32), flat_images[idx_train].astype(numpy.float32),
                              nb_epoch=self.epochs,verbose=self.verbose,
                              validation_data=(features[idx_valid].astype(numpy.float32), flat_images[idx_valid].astype(numpy.float32)),
                              callbacks=[EarlyStopping(monitor='acc', patience=15)])

        if self.folder_debug is not None:
            self.save_stats(hist)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def generate(self,array,shape):
        flat_array = self.model.predict(array, verbose=0)
        flat_array*= 255.0/flat_array.max()
        return flat_array.reshape((flat_array.shape[0],)+shape)
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
