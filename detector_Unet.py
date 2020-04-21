
import cv2
import os
import sys
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
from skimage.io import imread
from tqdm import tqdm
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.optimizers import RMSprop
# ----------------------------------------------------------------------------------------------------------------------
class detector_Unet(object):
# ----------------------------------------------------------------------------------------------------------------------
    def __init__(self,filename_chk=None):
        self.W = 256
        self.H = 256
        self.n_channels = 3
        self.init_model()
        self.folder_out = './data/output/'
        if filename_chk is not None:
            self.__load_model(filename_chk)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_model(self):
        inputs = Input((self.H, self.W, self.n_channels))
        s = Lambda(lambda x: x / 255)(inputs)

        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        self.model.compile(loss=[self.bce_dice_loss], optimizer=RMSprop(lr=0.0001), metrics=[self.dice_coeff])

        return
# ----------------------------------------------------------------------------------------------------------------------
    def do_patch(self,img, BORDER1=8, BORDER=12,  PATCH_SIZE=128, color=True):

        OUT_PATCH_SIZE = PATCH_SIZE - 2 * BORDER
        img_h, img_w = img.shape[:2]

        top, bottom, left, right = BORDER, BORDER + PATCH_SIZE, BORDER, BORDER + PATCH_SIZE
        ext_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)

        input_patch_im = []

        for y in range(0, img_h, OUT_PATCH_SIZE):
            for x in range(0, img_w, OUT_PATCH_SIZE):
                if not color:
                    input_patch_im.append(numpy.expand_dims(numpy.asarray(ext_img[y:y + PATCH_SIZE, x:x + PATCH_SIZE], dtype='float32'), axis=2))
                else:
                    input_patch_im.append((numpy.asarray(ext_img[y:y + PATCH_SIZE, x:x + PATCH_SIZE], dtype='float32')))

        return numpy.array(input_patch_im)
# ----------------------------------------------------------------------------------------------------------------------
    def do_stitch(self,img_h,img_w,patches, BORDER1, BORDER, PATCH_SIZE, color=True):

        OUT_PATCH_SIZE = PATCH_SIZE - 2 * BORDER
        if color:
            output_img = numpy.zeros((img_h + OUT_PATCH_SIZE, img_w + OUT_PATCH_SIZE, 3), dtype="uint8")
        else:
            output_img = numpy.zeros((img_h + OUT_PATCH_SIZE, img_w + OUT_PATCH_SIZE, 1), dtype="uint8")
        idx = 0
        for y in range(0, img_h, OUT_PATCH_SIZE):
            for x in range(0, img_w, OUT_PATCH_SIZE):
                patch = patches[idx,:,:]
                patch = patch[BORDER : BORDER + OUT_PATCH_SIZE, BORDER : BORDER + OUT_PATCH_SIZE]
                output_img[y: y + OUT_PATCH_SIZE, x: x + OUT_PATCH_SIZE] = patch
                idx = idx + 1

        return output_img[0:img_h,0:img_w]
# ----------------------------------------------------------------------------------------------------------------------
    def dice_coeff(self,y_true, y_pred):

        smooth = 0*1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        return score
# ----------------------------------------------------------------------------------------------------------------------
    def dice_loss(self,y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss
# ----------------------------------------------------------------------------------------------------------------------
    def bce_dice_loss(self,y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss
# ----------------------------------------------------------------------------------------------------------------------
    def do_resize(self,image_data,image_mask):
        image_data = cv2.resize(image_data,(self.W,self.H))
        image_mask = cv2.resize(image_data, (self.W, self.H))
        return image_data,image_mask
# ----------------------------------------------------------------------------------------------------------------------
    def get_data_train(self, folder_in):

        local_filenames_data = tools_IO.get_filenames(folder_in,'*.jpg')
        local_filenames_mask = tools_IO.get_filenames(folder_in,'*.png')

        X_train, Y_train = [],[]

        for file_data,file_mask in zip (local_filenames_data,local_filenames_mask):
            image_data = cv2.imread(folder_in+file_data)
            image_mask = cv2.imread(folder_in + file_mask)

            image_data,image_mask = self.do_resize(image_data,image_mask)

            X_train.append(image_data)
            Y_train.append(numpy.array([image_mask[:,:,0]])/255)

        X_train = numpy.array(X_train)
        Y_train = numpy.array(Y_train)

        return X_train, Y_train
# ----------------------------------------------------------------------------------------------------------------------
    def fit(self, X_train, Y_train):

        earlystopper = EarlyStopping(patience=6, verbose=1)
        checkpointer = ModelCheckpoint(self.folder_out+'chk.h5', verbose=1, save_best_only=True)
        results = self.model.fit(X_train, Y_train, validation_split=0.3, batch_size=16, epochs=10,callbacks=[earlystopper, checkpointer])
        self.model.save(self.folder_out+'trained.h5')
        return results
# ----------------------------------------------------------------------------------------------------------------------
    def __load_model(self,filename_in):
        self.model = load_model(filename_in,custom_objects={'dice_coeff': self.dice_coeff, 'bce_dice_loss': self.bce_dice_loss})
        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self,image, threshold_prob = 0.15):
        arr = self.do_patch(image, 8, 12, 256, color=True)
        preds_arr = self.model.predict(arr, verbose=0)
        preds_arr[preds_arr <= threshold_prob] = 0.
        preds_arr_t = preds_arr * 255
        pred_img = 255*self.do_stitch(image.shape[0],image.shape[1],preds_arr_t, 8, 12, 256)

        return pred_img
# ----------------------------------------------------------------------------------------------------------------------