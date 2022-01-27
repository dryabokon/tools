import numpy
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
class tools_TS(object):
    def __init__(self,Classifier=None,dark_mode=False,folder_out=None):
        self.classifier = Classifier
        self.Plotter = tools_plot_v2.Plotter(folder_out,dark_mode=dark_mode)
        self.folder_out = folder_out
        return
# ---------------------------------------------------------------------------------------------------------------------
    def do_train_test(self,X, Y,train_size,ongoing_retrain):

        test_size = Y.shape[0]-train_size
        Y_train_predict = self.classifier.train(X[:train_size], Y[:train_size])
        Y_test_predict = self.classifier.predict(X, Y,ongoing_retrain)[-test_size:]

        return Y_train_predict,Y_test_predict
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_train_test(self, df, idx_target, ratio=0.5,ongoing_retrain=False,do_debug=False):

        X, Y = tools_DF.df_to_XY(df,idx_target)
        train_size = int(Y.shape[0]*ratio)

        Y_train_pred,Y_test_pred = self.do_train_test(X,Y,train_size,ongoing_retrain)

        if do_debug:
            df_train = pd.DataFrame({'fact': Y[:train_size], 'pred': Y_train_pred})
            df_test =  pd.DataFrame({'fact': Y[train_size:], 'pred': Y_test_pred})

            self.Plotter.TS_seaborn(df_train, idxs_target=[0, 1], idx_time=None, filename_out='train_%s.png' % self.classifier.name)
            self.Plotter.TS_seaborn(df_test, idxs_target=[0, 1], idx_time=None, filename_out='test_%s.png' % self.classifier.name)

            self.Plotter.plot_fact_predict(Y[:train_size], Y_train_pred,filename_out='train_fact_pred_%s.png'%self.classifier.name)
            self.Plotter.plot_fact_predict(Y[train_size:], Y_test_pred ,filename_out='test_fact_pred_%s.png'%self.classifier.name)

            x_max = 1.2*max(numpy.abs(Y[:train_size] - Y_train_pred).max(),numpy.abs(Y[train_size:] - Y_test_pred).max())

            self.Plotter.plot_hist(Y[:train_size] - Y_train_pred, x_range=[-x_max,+x_max],filename_out='train_err_%s.png'%self.classifier.name)
            self.Plotter.plot_hist(Y[train_size:] - Y_test_pred , x_range=[-x_max,+x_max],filename_out='test_err_%s.png' % self.classifier.name)
            df_train.to_csv(self.folder_out + 'train.csv', index=False, sep='\t')
            df_test.to_csv(self.folder_out + 'test_%s.csv' % self.classifier.name, index=False, sep='\t')

        return
# ---------------------------------------------------------------------------------------------------------------------
    def predict_n_steps_ahead(self,df, idx_target,n_steps,do_debug=False):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        Y_pred_train = self.classifier.train(X, Y)
        Y_pred_ahead,CI_pred_ahead = self.classifier.predict_n_steps_ahead(n_steps)

        df_step = pd.DataFrame( {'predict_ahead' : Y_pred_ahead,
                                'predict_ahead_min': CI_pred_ahead[:,0],
                                'predict_ahead_max': CI_pred_ahead[:,1]})

        if do_debug:
            df_retro = pd.DataFrame({'GT': df.iloc[:, idx_target],
                                     'predict': numpy.full(df.shape[0], numpy.nan),
                                     'predict_ahead': numpy.full(df.shape[0], numpy.nan),
                                     'predict_ahead_min': numpy.full(df.shape[0], numpy.nan),
                                     'predict_ahead_max': numpy.full(df.shape[0], numpy.nan),
                                     })


            df_step['GT'] = numpy.full(n_steps, numpy.nan)
            df_step['predict'] = numpy.full(n_steps, numpy.nan)

            df_retro = df_retro.append(df_step, ignore_index=True)
            x_range = [max(0, df_retro.shape[0] - n_steps * 20), df_retro.shape[0]]
            self.Plotter.TS_matplotlib(df_retro, [0, 2, 1], None, idxs_fill=[3, 4], x_range=x_range,filename_out='pred_ahead_%s.png' % (self.classifier.name))

        return df_step
# ---------------------------------------------------------------------------------------------------------------------