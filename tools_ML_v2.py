import pandas as pd
import os
import numpy
from sklearn import metrics
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
class ML(object):
    def __init__(self,Classifier,folder_out=None,dark_mode=True):
        self.classifier = Classifier
        self.P = tools_plot_v2.Plotter(folder_out,dark_mode)
        self.init_plot_params()
        self.folder_out = folder_out
        if folder_out is not None and (not os.path.exists(folder_out)):
            os.mkdir(folder_out)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def init_plot_params(self):
        self.marker_transparency = 0.80
        self.marker_size = 8
        self.figsize = (3.5, 3.5)
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
    def predict_X(self, X):
        #scores = numpy.array([(100 * self.classifier.predict(x)[:, 1]).astype(int) for x in X])
        scores = (100*self.classifier.predict(X)[:, 1]).astype(int)
        return scores
# ---------------------------------------------------------------------------------------------------------------------
    def predict_df(self, df, idx_target):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        scores = self.predict_X(X)
        return scores
# ---------------------------------------------------------------------------------------------------------------------
    def learn_df(self, df, idx_target):
        X,Y = tools_DF.df_to_XY(df, idx_target)
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
    def get_precision_recall_fpr(self, Y_GT, Y_pred):

        n_pos = numpy.sum(Y_GT)
        n_neg = numpy.sum(1-Y_GT)

        n_hit = numpy.sum(numpy.array((Y_GT-Y_pred)==0)*numpy.array((Y_GT > 0)))
        n_pred_pos = numpy.sum((Y_pred==1))
        n_pred_neg = numpy.sum((Y_pred==0))

        precision, recall,fpr = 0,0,0
        if n_pred_pos>0:precision = n_hit/n_pred_pos
        if n_pos>0 :recall = n_hit/n_pos
        if n_neg>0: fpr = n_pred_neg / n_neg

        return precision, recall, fpr
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_metrics(self, df, idx_target,scores,is_train,plot_charts=False,xlim=None,description=''):

        X, Y = tools_DF.df_to_XY(df, idx_target)
        fpr, tpr, thresholds = metrics.roc_curve(Y, scores)
        precisions, recalls, thresholds = metrics.precision_recall_curve(Y, scores)
        idx_th = numpy.argmax([p * r / (p + r + 1e-4) for p, r in zip(precisions, recalls)])

        accuracy = self.get_accuracy(tpr, fpr, nPos=numpy.count_nonzero(Y > 0),nNeg=numpy.count_nonzero(Y <= 0))
        auc = metrics.auc(fpr, tpr)
        mAP = metrics.average_precision_score(Y, scores)
        f1 = 2*precisions[idx_th]*recalls[idx_th]/(precisions[idx_th]+recalls[idx_th])

        keylist = ['train_AUC', 'train_ACC', 'train_mAP','train_P','train_R','train_f1'] if is_train else ['test_AUC', 'test_ACC', 'test_mAP','test_P','test_R','test_f1']
        dct_res = dict(zip(keylist,[auc,accuracy,mAP,precisions[idx_th],recalls[idx_th],f1]))

        if plot_charts:
            suffix = 'Train' if is_train else 'Test'
            caption_auc = 'AUC_'+suffix
            caption_map = 'mAP_'+suffix

            self.P.plot_tp_fp(tpr, fpr, auc, caption=caption_auc, filename_out=description+caption_auc + '.png')
            self.P.plot_PR(precisions,recalls, mAP, caption=caption_map,filename_out=description+caption_map + '.png')
            self.P.plot_1D_features_pos_neg(scores, Y, labels=True, bins=numpy.linspace(0,100,100),xlim=xlim,filename_out=description+'scores_'+suffix+'.png')

            #self.P.plot_TP_FP_PCA_scatter(scores, X,Y,filename_out=description + 'TP_FP_PCA_' + suffix + '.png')
            self.P.plot_TP_FP_rectangles(scores, X,Y,filename_out=description + 'TP_FP_' + suffix + '.png')

            #df_temp = pd.DataFrame({'GT': Y, 'pred': pred}, index=df.index)
            #df_temp = tools_DF.remove_dups(df_temp)
            #self.P.TS_seaborn(df_temp, idxs_target=[0,1], idx_time=None, mode='pointplot', remove_xticks=False, major_step=60, filename_out=caption + '_fact_pred1.png')

        return dct_res
# ----------------------------------------------------------------------------------------------------------------------
    def feature_correlation(self,df0):

        df = tools_DF.hash_categoricals(df0)
        columns = df.columns
        df_Q = df.corr().abs()

        for i in range(df_Q.shape[0]):
            df_Q.iloc[i, i] = 0

        ranks = []
        while len(ranks) < df_Q.shape[1]:
            idx = numpy.argmax(df_Q)
            r, c = numpy.unravel_index(idx, df_Q.shape)
            df_Q.iloc[r, c] = 0
            if r not in ranks:
                ranks.append(r)
            if c not in ranks:
                ranks.append(c)

        ranks = numpy.array(ranks)
        df_Q = abs(df[columns[ranks]].corr())

        for i in range(df_Q.shape[0]):
            df_Q.iloc[i, i] = numpy.nan

        df_Q.to_csv(self.folder_out + 'df_F_corr.csv')
        corrmat_MC = tools_DF.from_multi_column(pd.read_csv(self.folder_out + 'df_F_corr.csv'), idx_time=0).sort_values(by='value', ascending=False)
        corrmat_MC.to_csv(self.folder_out + 'df_F_corr_MC.csv', index=False)

        self.P.plot_feature_correlation(df_Q,self.figsize,filename_out='FC.png')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_report(self,df, idx_target):
        X, Y_target = tools_DF.df_to_XY(df, idx_target)
        scores = self.predict_df(df, idx_target)
        y_pred = numpy.array([s>0.5 for s in scores])
        print(metrics.classification_report(Y_target, y_pred))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_density_2d(self, df, idx_target,N = 100,x_range=None,y_range=None,filename_out=None):

        df = tools_DF.add_noise_smart(df, idx_target)

        idx_col = numpy.delete(numpy.arange(0, df.shape[1]), idx_target)
        X = df.iloc[:, idx_col]
        xx, yy = self.get_datagrid_for_2d(X.to_numpy(), N)
        grid_confidence = self.predict_X(numpy.c_[xx.flatten(), yy.flatten()]).reshape((N, N))

        X = df.iloc[:, idx_col].to_numpy()
        Y = df.iloc[:, [idx_target]].to_numpy().flatten()
        X0 = X[Y <= 0]
        X1 = X[Y > 0]

        self.P.plot_contourf(X0,X1, xx, yy, grid_confidence,x_range,y_range,df.columns[idx_col[0]],df.columns[idx_col[1]],
                             marker_size=self.marker_size,transparency=self.marker_transparency,figsize=self.figsize,filename_out=filename_out)


        return
# ----------------------------------------------------------------------------------------------------------------------
    def combine_metrics(self,dct_metrics_train,dct_metrics_test,dct_metric_crossval=None):

        name = numpy.array(['AUC', 'ACC', 'mAP','P','R','f1'])
        metrics_train = [dct_metrics_train['train_%s'%s]for s in name]
        metrics_test  = [dct_metrics_test['test_%s' %s] for s in name]

        df = pd.DataFrame({'metric':name,'train':metrics_train,'test':metrics_test})

        if dct_metric_crossval is not None:
            metrics_train_cv = numpy.array([dct_metric_crossval['train_AUC'], dct_metric_crossval['train_ACC'], dct_metric_crossval['train_mAP']])
            metrics_test_cv = numpy.array([dct_metric_crossval['test_AUC'], dct_metric_crossval['test_ACC'],dct_metric_crossval['test_mAP']])
            name_cv = numpy.array(['CV AUC', 'CV ACC', 'CV mAP'])
            df2 = pd.DataFrame({'metric': name_cv, 'train': metrics_train_cv, 'test': metrics_test_cv})
            df = df.append(df2, ignore_index=True)

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def cross_validation_train_test(self,df, idx_target):

        def my_custom_MAPE(model, X, Y):return metrics.average_precision_score(Y, model.predict_proba(X)[:,1].reshape((-1,1)))

        if self.classifier.model is None:
            return None

        X, Y = tools_DF.df_to_XY(df, idx_target)
        cross_val = RepeatedStratifiedKFold(n_splits=2, n_repeats=10)
        scoring_type = {'AUC': 'roc_auc', 'ACC': 'accuracy','mAP': my_custom_MAPE}
        cv_results = cross_validate(self.classifier.model, X, Y, cv=cross_val,return_train_score=True, scoring=scoring_type)

        dct_res = {}
        for key in scoring_type.keys():
            dct_res['train_'+key] = cv_results['train_'+key].mean()
            dct_res['test_'+key] = cv_results['test_' + key].mean()

        return dct_res
# ----------------------------------------------------------------------------------------------------------------------
    def E2E_train_test_df(self,df_train,df_test=None,idx_target=0,do_charts=False,do_density=False,do_pca = False,idx_columns=None,description=''):


        if df_test is None:
            df_train, df_test = train_test_split(df_train, test_size=0.5,shuffle=True)

        df_train = tools_DF.remove_dups(df_train.dropna())
        df_test = tools_DF.remove_dups(df_test.dropna())

        if idx_columns is not None:
            df_train = df_train.iloc[:,[idx_target]+idx_columns]
            df_test  = df_test.iloc[:, [idx_target] + idx_columns]
            idx_target = 0

        self.learn_df(df_train.dropna(),idx_target)

        scores_train = self.predict_df(df_train, idx_target)
        scores_test =self.predict_df(df_test, idx_target)
        S = numpy.concatenate((scores_train, scores_test))
        xlim =None# [int(numpy.floor(numpy.quantile(S,0.01))),int(numpy.ceil(1+numpy.quantile(S,0.99)))]

        dct_metrics_train = self.evaluate_metrics(df_train, idx_target,scores=scores_train,is_train=True,plot_charts=do_charts,xlim=xlim,description=description)
        dct_metrics_test  = self.evaluate_metrics(df_test , idx_target,scores=scores_test,is_train=False,plot_charts=do_charts,xlim=xlim,description=description)
        dct_metric_crossval = None#self.cross_validation_train_test(df_train, idx_target)

        df_metrics = self.combine_metrics(dct_metrics_train,dct_metrics_test,dct_metric_crossval)

        if do_density and df_train.shape[1]==3:

            df_train = tools_DF.remove_long_tail(df_train)
            df_test = tools_DF.remove_long_tail(df_test)
            x_range,y_range=None,None
            # X = pd.concat([df_train.iloc[:, idx_col],df_train.iloc[:, idx_col]])
            # pad = 0.5
            # x_range = numpy.array([X.iloc[:,0].min()-pad, X.iloc[:,0].max()+pad])
            # y_range = numpy.array([X.iloc[:,1].min()-pad, X.iloc[:,1].max()+pad])
            self.plot_density_2d(df_train, idx_target, x_range=x_range,y_range=y_range,filename_out = description+'density_train.png')
            self.plot_density_2d(df_test , idx_target, x_range=x_range,y_range=y_range,filename_out = description+'density_test.png')

        if do_pca and df_train.shape[1]>3:
            #self.P.plot_SVD(df_train, idx_target,filename_out=description+'dim_SVD.png')
            self.P.plot_tSNE(df_train, idx_target,filename_out=description+'dim_tSNE.png')
            #self.P.plot_PCA(df_train, idx_target,filename_out=description+'dim_PCA.png')
            #self.P.plot_LLE(df_train, idx_target,filename_out=description+'dim_LLE.png')
            #self.P.plot_ISOMAP(df_train, idx_target,filename_out=description+'dim_ISOMAP.png')
            #self.P.plot_UMAP(df_train, idx_target,filename_out=description+'dim_UMAP.png')

        return df_metrics
# ----------------------------------------------------------------------------------------------------------------------