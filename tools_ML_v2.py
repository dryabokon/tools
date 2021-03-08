import os
import numpy
from sklearn import metrics
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
class ML(object):
    def __init__(self,Classifier,folder_out=None,dark_mode=False):
        self.classifier = Classifier
        self.P = tools_plot_v2.Plotter(folder_out,dark_mode)
        self.folder_out = folder_out
        if folder_out is not None and (not os.path.exists(folder_out)):
            os.mkdir(folder_out)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def get_accuracy(self,TPR,FPR,nPos,nNeg):
        accuracy=[]
        for tpr,fpr in zip(TPR,FPR):
            TP = nPos*tpr
            FN = nPos-TP
            FP = nNeg*fpr
            TN = nNeg-FP
            accuracy.append((TP+TN)/(nPos+nNeg))
        return numpy.max(numpy.array(accuracy))
# ---------------------------------------------------------------------------------------------------------------------
    def df_to_XY(self,df,idx_target):
        columns = df.columns.to_numpy()
        col_types = numpy.array([str(t) for t in df.dtypes])
        are_categoirical = numpy.array([cc in ['object', 'category', 'bool'] for cc in col_types])
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
        are_categoirical = numpy.delete(are_categoirical, idx_target)
        idx = idx[~are_categoirical]
        X = df.iloc[:, idx].to_numpy()
        Y = df.iloc[:, [idx_target]].to_numpy().flatten()
        return X,Y
# ---------------------------------------------------------------------------------------------------------------------
    def learn_df(self, df, idx_target):
        X,Y = self.df_to_XY(df, idx_target)
        self.classifier.learn(X,Y)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_datagrid_for_2d(self, X, N = 10):

        if (X.shape[1] != 2):return

        minx,maxx = numpy.nanmin(X[:, 0]), numpy.nanmax(X[:, 0])
        miny,maxy = numpy.nanmin(X[:, 1]), numpy.nanmax(X[:, 1])
        xx, yy = numpy.meshgrid(numpy.linspace(minx, maxx, num=N), numpy.linspace(miny, maxy, num=N))

        return  xx, yy
# ----------------------------------------------------------------------------------------------------------------------
    def plot_ROC(self, df, idx_target,caption='',filename_out=None):

        Y = df.iloc[:, [idx_target]].to_numpy().flatten()
        scores = self.predict_df(df, idx_target)
        fpr, tpr, thresholds = metrics.roc_curve(Y, scores)
        auc = metrics.auc(fpr, tpr)
        accuracy = self.get_accuracy(tpr, fpr, nPos=numpy.count_nonzero(Y > 0),nNeg=numpy.count_nonzero(Y <= 0))

        self.P.plot_tp_fp(tpr, fpr, auc, caption=caption, filename_out=filename_out)
        return accuracy, auc
# ----------------------------------------------------------------------------------------------------------------------
    def plot_report(self,df, idx_target):
        X, Y_target = self.df_to_XY(df, idx_target)
        scores = self.predict_df(df, idx_target)
        y_pred = numpy.array([s>0.5 for s in scores])
        print(metrics.classification_report(Y_target, y_pred))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_density_2d(self, df, idx_target,N = 100,filename_out=None,x_range=None,y_range=None,figsize=(3.5,3.5)):

        idx_col = numpy.delete(numpy.arange(0, len(df.columns.to_numpy())), idx_target)
        X = df.iloc[:, idx_col]
        xx, yy = self.get_datagrid_for_2d(X.to_numpy(), N)
        grid_confidence = self.predict_X(numpy.c_[xx.flatten(), yy.flatten()]).reshape((N, N))

        X = df.iloc[:, idx_col].to_numpy()
        Y = df.iloc[:, [idx_target]].to_numpy().flatten()
        X0 = X[Y <= 0]
        X1 = X[Y > 0]

        self.P.plot_contourf(X0,X1, xx, yy, grid_confidence,x_range,y_range,df.columns[1],df.columns[2],figsize,filename_out)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def E2E_train_test_df(self,df,idx_target,do_density=False,do_pca = False,idx_columns=None):

        if idx_columns is not None:
            df = df.iloc[:,[idx_target]+idx_columns]
            idx_target = 0

        df_train, df_test = train_test_split(df.dropna(), test_size=0.5,shuffle=True)
        self.learn_df(df_train,idx_target)

        X, Y = self.df_to_XY(df, idx_target)
        auc_cross = cross_val_score(self.classifier.model, X, Y, scoring='roc_auc' ,cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=10)).mean()
        acc_cross = cross_val_score(self.classifier.model, X, Y, scoring='accuracy',cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=10)).mean()
        acc_train, auc_train = self.plot_ROC(df_train, idx_target, caption='Train',filename_out='ROC_train.png')
        acc_test , auc_test  = self.plot_ROC(df_test , idx_target, caption='Test' ,filename_out='ROC_test.png')
        print('acc_train = %1.3f\tacc_test = %1.3f' % (acc_train, acc_test))
        print('auc_train = %1.3f\tauc_test = %1.3f' % (auc_train, auc_test))
        print('acc_cross = %1.3f' %acc_cross)
        print('auc_cross = %1.3f' %auc_cross)
        print()

        if do_density and df_train.shape[1]==3:
            idx_col = numpy.delete(numpy.arange(0, len(df.columns.to_numpy())), idx_target)
            X = df.iloc[:, idx_col]

            x_range = numpy.array([X.iloc[:,0].min(), X.iloc[:,0].max()])
            y_range = numpy.array([X.iloc[:,1].min(), X.iloc[:,1].max()])
            self.plot_density_2d(df_train, idx_target, filename_out = 'density_train.png',x_range=x_range,y_range=y_range)
            self.plot_density_2d(df_test , idx_target, filename_out = 'density_test.png',x_range=x_range,y_range=y_range)
        # else:
        #     self.plot_density_multi_dim(df_train, idx_target)

        if do_pca and df_train.shape[1]>3:
            self.P.plot_SVD(df_train, idx_target,'dim_SVD.png')
            self.P.plot_tSNE(df_train, idx_target,'dim_tSNE.png')
            self.P.plot_PCA(df_train, idx_target,'dim_PCA.png')
            self.P.plot_LLE(df_train, idx_target,'dim_LLE.png')
            self.P.plot_ISOMAP(df_train, idx_target,'dim_ISOMAP.png')
            #self.P.plot_UMAP(df_train, idx_target,'dim_UMAP.png')


        return
# ---------------------------------------------------------------------------------------------------------------------
    def predict_X(self, X):
        scores = numpy.array([(100 * self.classifier.predict(x)[:, 1]).astype(int) for x in X])
        return scores
# ---------------------------------------------------------------------------------------------------------------------
    def predict_df(self, df,idx_target):
        X, Y = self.df_to_XY(df, idx_target)
        scores = self.predict_X(X)
        return scores
# ---------------------------------------------------------------------------------------------------------------------
