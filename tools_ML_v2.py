import numpy
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot
# ----------------------------------------------------------------------------------------------------------------------
class tools_ML_enhanced(object):
    def __init__(self,Classifier):
        self.classifier = Classifier
        self.folder_out = './data/output/'
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
    def plot_ROC(self, df, idx_target,filename_out):

        Y = df.iloc[:, [idx_target]].to_numpy().flatten()
        scores = self.predict_df(df, idx_target)
        fpr_train, tpr_train, thresholds = metrics.roc_curve(Y, scores)
        auc_train = metrics.auc(fpr_train, tpr_train)
        accuracy = self.get_accuracy(tpr_train, fpr_train, nPos=numpy.count_nonzero(Y > 0),nNeg=numpy.count_nonzero(Y <= 0))
        plt.clf()
        tools_plot.plot_tp_fp(plt, None, tpr_train, fpr_train, auc_train,filename_out=filename_out)
        return accuracy
# ----------------------------------------------------------------------------------------------------------------------
    def plot_density(self, df, idx_target, filename_out):

        N = 100
        idx_col = numpy.delete(numpy.arange(0, len(df.columns.to_numpy())), idx_target)
        X = df.iloc[:, idx_col]
        xx, yy = self.get_datagrid_for_2d(X.to_numpy(), N)
        grid_confidence = self.predict_X(numpy.c_[xx.flatten(), yy.flatten()]).reshape((N, N))

        X = df.iloc[:, idx_col].to_numpy()
        Y = df.iloc[:, [idx_target]].to_numpy().flatten()
        X0 = X[Y <= 0]
        X1 = X[Y > 0]

        plt.clf()
        plt.contourf(xx,yy,grid_confidence, cmap=cm.coolwarm, alpha=.8)
        plt.plot(X0[:, 0], X0[:, 1], 'ro', color='blue', alpha=0.4)
        plt.plot(X1[:, 0], X1[:, 1], 'ro', color='red', alpha=0.4)

        if filename_out is not None:
            plt.savefig(filename_out)

        return
# ----------------------------------------------------------------------------------------------------------------------

    def E2E_train_test_df(self,df,idx_target,idx_columns=None):

        if idx_columns is not None:
            df = df.iloc[:,[idx_target]+idx_columns]
            idx_target = 0

        df_train, df_test = train_test_split(df.dropna(), test_size=0.5,shuffle=True)

        self.learn_df(df_train,idx_target)

        acc_train = self.plot_ROC(df_train, idx_target, self.folder_out + 'ROC_train.png')
        acc_test =  self.plot_ROC(df_test , idx_target, self.folder_out + 'ROC_test.png')
        self.plot_density(df_train, idx_target, self.folder_out + 'density.png')

        print('accuracy_train = %1.3f\taccuracy_test = %1.3f'%(acc_train,acc_test))

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
