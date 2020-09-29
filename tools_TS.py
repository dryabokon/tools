import os
import cv2
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# ----------------------------------------------------------------------------------------------------------------------
# https://www.youtube.com/watch?v=5lUUrREboSk
# GRU, Peephole, # https://medium.com/@godricglow/a-deeper-understanding-of-nnets-part-3-lstm-and-gru-e557468acb04
# random walk?
# benchmark with sliding window # https://www.datascience.com/blog/time-series-forecasting-machine-learning-differences
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_plot
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class tools_TS(object):
    def __init__(self,Classifier=None,delim='\t',max_len=1000000):
        self.classifier = Classifier
        self.delim = delim
        self.max_len = max_len
        return
# ---------------------------------------------------------------------------------------------------------------------
    def load_and_normalize(self,filename_input):

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = tools_IO.load_mat_pd(filename_input,delim=self.delim,dtype=numpy.float32)[:self.max_len]
        data = self.scaler.fit_transform(dataset[:,[0,1]])
        self.scaler_dim = data.shape
        return data
# ---------------------------------------------------------------------------------------------------------------------
    def denormalize(self, X):
        empty = numpy.zeros(self.scaler_dim)
        if len(X.shape)==1:
            empty[:X.shape[0],0] = X
            res = self.scaler.inverse_transform(empty)
            res= res[:X.shape[0],0]
        else:
            empty[:X.shape[0],:] = X
            res = self.scaler.inverse_transform(X)
            res = res[:X.shape[0], :]

        return res
# ---------------------------------------------------------------------------------------------------------------------
    def do_train_test(self,X, Y,train_size):
        Y_train_predict = self.classifier.train(X[:train_size], Y[:train_size])
        Y_test_predict = self.classifier.predict(X[train_size:], Y[train_size:])
        return Y_train_predict,Y_test_predict
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_train_test(self, filename_input, path_output, target_column = 0,use_cashed_model=False,verbose=False,ratio=0.5):

        dataset = self.load_and_normalize(filename_input)
        train_size = int(dataset.shape[0] * ratio)
        suffix = ''
        if target_column >= 0: suffix = ('%03d_' % target_column)
        filename_train_fact = path_output + 'train_fact'  + suffix + '.txt'
        filename_test_fact  = path_output + 'test_fact'   + suffix + '.txt'
        filename_train_pred = path_output + 'train_pred_' + suffix + self.classifier.name + '.txt'
        filename_test_pred  = path_output + 'test_pred_'  + suffix + self.classifier.name + '.txt'

        if target_column >= 0:
            X = numpy.delete(dataset, target_column, axis=1)
            Y = dataset[:, target_column]
            Y_train_predict,Y_test_predict = self.do_train_test(X,Y,train_size)
            tools_IO.save_mat(self.denormalize(Y[:train_size].T ), filename_train_fact)
            tools_IO.save_mat(self.denormalize(Y[train_size:].T ), filename_test_fact)
            tools_IO.save_mat(self.denormalize(Y_train_predict.T), filename_train_pred)
            tools_IO.save_mat(self.denormalize(Y_test_predict.T ), filename_test_pred)
        else:
            Y_train_predicts,Y_test_predicts=[],[]
            for target_column in range(0, dataset.shape[1]):
                Y_train_predict, Y_test_predict = self.do_train_test(numpy.delete(dataset, target_column, axis=1),dataset[:, target_column],train_size)
                Y_train_predicts.append(Y_train_predict)
                Y_test_predicts.append(Y_test_predict)
            tools_IO.save_mat(self.denormalize(dataset[:train_size]), filename_train_fact)
            tools_IO.save_mat(self.denormalize(dataset[train_size:]), filename_test_fact)
            tools_IO.save_mat(self.denormalize(numpy.array(Y_train_predicts).T), filename_train_pred)
            tools_IO.save_mat(self.denormalize(numpy.array(Y_test_predicts).T ), filename_test_pred)


        tools_plot.plot_two_series(filename_train_fact, filename_train_pred,caption='train',filename_out=path_output+self.classifier.name+'_train.png')
        tools_plot.plot_two_series(filename_test_fact , filename_test_pred ,caption='test' ,filename_out=path_output+self.classifier.name+'_test.png')


        return
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_fit(self, filename_input, path_output, target_column = 0, use_cashed_model=False, verbose=False):

        dataset = self.load_and_normalize(filename_input)
        suffix = ''
        if target_column >= 0: suffix = ('%03d_' % target_column)
        filename_fact = path_output + 'fact' + suffix + '.txt'
        filename_pred = path_output + 'fit_' + suffix + self.classifier.name + '.txt'

        if target_column >=0:
            X = numpy.delete(dataset, target_column, axis=1)
            Y = dataset[:,target_column]
            Y_predict = self.classifier.train(X, Y)
            tools_IO.save_mat(self.denormalize(Y.T), filename_fact)
            tools_IO.save_mat(self.denormalize(Y_predict.T), filename_pred)
        else:
            Y_predict = []
            for col in range(0,dataset.shape[1]):
                Y_predict.append(self.classifier.train(numpy.delete(dataset, col, axis=1), dataset[:, col]))
            Y_predict = numpy.array(Y_predict).T
            tools_IO.save_mat(self.denormalize(dataset), filename_fact)
            tools_IO.save_mat(self.denormalize(Y_predict), filename_pred)


        #image_signal = tools_draw_numpy.draw_signals_lines([255*Y_predict[:1024],255*Y[:1024]],w=1)
        image_signal = tools_draw_numpy.draw_signals_lines([255 * Y[:1024]], w=3)
        cv2.imwrite(path_output+self.classifier.name+'_fit.png',image_signal)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def calc_SMAPE_for_files(self,filename1,filename2):
        mat1 = tools_IO.load_mat(filename1, dtype=numpy.float32,delim='\t')
        mat2 = tools_IO.load_mat(filename2, dtype=numpy.float32,delim='\t')

        if len(mat1.shape)==1:
            mat1 = numpy.array([mat1])
            mat2 = numpy.array([mat2])

        q=0
        for v in range(0,mat1.shape[1]):
            q+= tools_plot.smape(mat1[:,v],mat2[:,v])

        return q/mat1.shape[1]
# ---------------------------------------------------------------------------------------------------------------------
