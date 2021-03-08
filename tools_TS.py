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
    def do_train_test(self,X, Y,train_size):
        Y_train_predict = self.classifier.train(X[:train_size], Y[:train_size])
        Y_test_predict = self.classifier.predict(X[train_size:], Y[train_size:])
        return Y_train_predict,Y_test_predict
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_train_test(self, df, idx_target = 0, ratio=0.5):

        X, Y = tools_DF.df_to_XY(df,idx_target)
        train_size = int(Y.shape[0]*ratio)

        Y_train_pred,Y_test_pred = self.do_train_test(X,Y,train_size)

        df_train = pd.DataFrame({'fact': Y[:train_size], 'pred': Y_train_pred})
        df_test =  pd.DataFrame({'fact': Y[train_size:], 'pred': Y_test_pred})
        df_train.to_csv(self.folder_out + 'train.csv', index=False,sep='\t')
        df_test.to_csv(self.folder_out + 'test_%s.csv'%self.classifier.name , index=False,sep='\t')
        self.Plotter.TS_seaborn(df_train, idxs_target=[0, 1], idx_feature=None, filename_out='train_%s.png'%self.classifier.name)
        self.Plotter.TS_seaborn(df_test , idxs_target=[0, 1], idx_feature=None, filename_out='test_%s.png'%self.classifier.name)

        self.Plotter.plot_fact_predict(Y[:train_size], Y_train_pred,filename_out='train_fact_pred_%s.png'%self.classifier.name)
        self.Plotter.plot_fact_predict(Y[train_size:], Y_test_pred ,filename_out='test_fact_pred_%s.png'%self.classifier.name)



        return
# ---------------------------------------------------------------------------------------------------------------------
