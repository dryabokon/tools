import pandas as pd
import os
import numpy
import uuid
from sklearn import metrics
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import joblib
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
class ML(object):
    def __init__(self,Classifier,folder_out=None,do_scaling = False, dark_mode=False):
        self.classifier = Classifier
        self.multiclass = None
        self.P = tools_plot_v2.Plotter(folder_out,dark_mode)
        self.init_plot_params()
        self.folder_out = folder_out
        self.do_scaling = do_scaling
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
    def predict_scores(self, X, idx_target=None):
        if isinstance(X,pd.DataFrame):
            X, y = tools_DF.df_to_XY(X, idx_target)

        if self.do_scaling:
            X = self.scaler.transform(X)
        scores = (100*self.classifier.predict(X)).astype(int)
        if not self.multiclass:
            scores = scores[:, 1]

        return scores
# ---------------------------------------------------------------------------------------------------------------------
    def predict_classes(self, X, idx_target=None):
        if X is not None and X.shape[0]>0:
            classes = numpy.argmax(self.predict_scores(X, idx_target),axis=1)
        else:
            classes =[]
        return classes
# ---------------------------------------------------------------------------------------------------------------------

    def learn_df(self, df, idx_target):

        X,Y = tools_DF.df_to_XY(df, idx_target)
        if self.do_scaling:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            X = self.scaler.fit_transform(X)



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
        pos_label = numpy.sort(numpy.unique(Y))[1]
        fpr, tpr, thresholds = metrics.roc_curve(Y, scores,pos_label=pos_label)
        precisions, recalls, thresholds = metrics.precision_recall_curve(Y, scores,pos_label=pos_label)
        idx_th = numpy.argmax([p * r / (p + r + 1e-4) for p, r in zip(precisions, recalls)])

        accuracy = self.get_accuracy(tpr, fpr, nPos=numpy.count_nonzero(Y==pos_label),nNeg=numpy.count_nonzero(Y!=pos_label))
        auc = metrics.auc(fpr, tpr)
        mAP = metrics.average_precision_score(Y, scores,pos_label=pos_label)
        f1 = 2*precisions[idx_th]*recalls[idx_th]/(precisions[idx_th]+recalls[idx_th])

        keylist = ['train_AUC', 'train_ACC', 'train_mAP','train_P','train_R','train_f1'] if is_train else ['test_AUC', 'test_ACC', 'test_mAP','test_P','test_R','test_f1']
        dct_res = dict(zip(keylist,[auc,accuracy,mAP,precisions[idx_th],recalls[idx_th],f1]))

        if plot_charts:
            suffix = 'Train' if is_train else 'Test'
            caption_auc = 'AUC_'+suffix
            caption_map = 'mAP_'+suffix

            self.P.plot_tp_fp(tpr, fpr, auc, caption=caption_auc, filename_out=description+caption_auc + '.png')
            self.P.plot_PR(precisions,recalls, mAP, caption=caption_map,filename_out=description+caption_map + '.png')
            colors = [self.P.get_color(t)[[2, 1, 0]] / 255.0 for t in numpy.unique(Y)]
            self.P.plot_1D_features_pos_neg(scores, Y, labels=True,colors=colors, bins=numpy.linspace(0,100,100),xlim=xlim,filename_out=description+'scores_'+suffix+'.png')

            #self.P.plot_TP_FP_PCA_scatter(scores, X,Y,filename_out=description + 'TP_FP_PCA_' + suffix + '.png')
            #self.P.plot_TP_FP_rectangles(scores, X,Y,filename_out=description + 'TP_FP_' + suffix + '.png')

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

        self.P.plot_feature_correlation(df_Q,filename_out='FC.png')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_report(self,df, idx_target):
        X, Y_target = tools_DF.df_to_XY(df, idx_target)
        scores = self.predict_scores(df, idx_target)
        y_pred = numpy.array([s>0.5 for s in scores])
        print(metrics.classification_report(Y_target, y_pred))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_density_2d(self, df, idx_target,N = 100,x_range=None,y_range=None,filename_out=None):

        df = tools_DF.add_noise_smart(df, idx_target)

        idx_col = numpy.delete(numpy.arange(0, df.shape[1]), idx_target)
        X = df.iloc[:, idx_col]
        xx, yy = self.get_datagrid_for_2d(X.to_numpy(), N)
        grid_confidence = self.predict_scores(numpy.c_[xx.flatten(), yy.flatten()]).reshape((N, N))

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
    def preditions_to_mat(self,labels_fact, labels_pred, patterns):
        mat = confusion_matrix(numpy.array(labels_fact), numpy.array(labels_pred))
        accuracy = [mat[i, i] / numpy.sum(mat[i, :]) for i in range(0, mat.shape[0])]

        idx = numpy.argsort(accuracy,).astype(int)[::-1]

        descriptions = numpy.array([('%s %3d%%' % (patterns[i], 100*accuracy[i])) for i in range(0, mat.shape[0])])

        a_test = numpy.zeros(labels_fact.shape[0])
        a_pred = numpy.zeros(labels_fact.shape[0])

        for i in range(0, a_test.shape[0]):
            a_test[i] = numpy.where(idx==int(labels_fact[i]))[0][0]
            a_pred[i] = numpy.where(idx==int(labels_pred[i]))[0][0]

        mat2 = confusion_matrix(numpy.array(a_test).astype(numpy.int), numpy.array(a_pred).astype(numpy.int))
        ind = numpy.array([('%3d' % i) for i in range(0, idx.shape[0])])

        l = max([len(each) for each in descriptions])
        descriptions = numpy.array([" " * (l - len(each)) + each for each in descriptions]).astype(numpy.chararray)
        descriptions = [ind[i] + ' | ' + descriptions[idx[i]] for i in range(0, idx.shape[0])]

        return mat2, descriptions, patterns[idx]

# ----------------------------------------------------------------------------------------------------------------------
    def print_accuracy(self,labels_fact, labels_pred, patterns, filename=None):

        if (filename != None):
            file = open(filename, 'w')
        else:
            file = None

        mat, descriptions, sorted_labels = self.preditions_to_mat(labels_fact, labels_pred, patterns)
        ind = numpy.array([('%3d' % i) for i in range(0, mat.shape[0])])
        TP = float(numpy.trace(mat))
        import tools_IO
        tools_IO.my_print_int(numpy.array(mat).astype(int), rows=descriptions, cols=ind.astype(int), file=file)

        print("Accuracy = %d/%d = %1.4f" % (TP, float(numpy.sum(mat)), float(TP / numpy.sum(mat))), file=file)
        print("Fails    = %d" % float(numpy.sum(mat) - TP), file=file)
        print(file=file)

        if (filename != None):
            file.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def construct_metrics_2_classes(self,df_train,df_test,df_val,idx_target,do_charts=False,description=''):
        scores_train = self.predict_scores(df_train, idx_target)
        scores_test = self.predict_scores(df_test, idx_target)

        dct_metrics_train = self.evaluate_metrics(df_train, idx_target,scores=scores_train,is_train=True,plot_charts=do_charts,description=description)
        dct_metrics_test  = self.evaluate_metrics(df_test , idx_target,scores=scores_test,is_train=False,plot_charts=do_charts,description=description)
        dct_metric_crossval = None#self.cross_validation_train_test(df_train, idx_target)

        df_metrics = self.combine_metrics(dct_metrics_train,dct_metrics_test,dct_metric_crossval)

        return df_metrics
# ----------------------------------------------------------------------------------------------------------------------
    def construct_metrics_multiclass(self,df_train,df_test,df_val,idx_target,do_charts=False,description=''):

        y_unique = numpy.sort(df_train.dropna().iloc[:, idx_target].unique())
        self.dct_class_id   = dict(zip(y_unique,numpy.arange(0, len(y_unique))))
        self.dct_class_name = dict(zip(numpy.arange(0, len(y_unique)), y_unique))

        scores_train = self.predict_scores(df_train, idx_target)
        scores_test  = self.predict_scores(df_test , idx_target)

        pred_train = self.predict_classes(df_train, idx_target)
        pred_test  = self.predict_classes(df_test, idx_target)
        pred_val   = self.predict_classes(df_val, idx_target)

        X, Y_train = tools_DF.df_to_XY(df_train, idx_target)
        cm_train = confusion_matrix(Y_train, [self.dct_class_name[c] for c in pred_train])

        X, Y_test = tools_DF.df_to_XY(df_test, idx_target)
        cm_test = confusion_matrix(Y_test, [self.dct_class_name[c] for c in pred_test])

        acc_train = numpy.trace(cm_train)/numpy.sum(cm_train)
        acc_test = numpy.trace(cm_test) / numpy.sum(cm_test)

        df_metrics = pd.DataFrame({'metric': ['ACC'], 'train': [acc_train], 'test': [acc_test]})

        if do_charts:
            self.print_accuracy(numpy.array([self.dct_class_id[y] for y in Y_train]), pred_train, y_unique, filename=self.folder_out+'cm_train.txt')
            self.print_accuracy(numpy.array([self.dct_class_id[y] for y in Y_test]) , pred_test , y_unique, filename=self.folder_out+'cm_test.txt')
            colors = [self.P.get_color(t)[[2, 1, 0]] / 255.0 for t in numpy.sort(numpy.unique(y_unique))]
            self.P.plot_1D_features_pos_neg(scores_train, Y_train, labels=True, colors=colors, bins=numpy.linspace(0, 100, 100), filename_out='scores_train.png')
            self.P.plot_1D_features_pos_neg(scores_test , Y_test , labels=True, colors=colors, bins=numpy.linspace(0, 100, 100), filename_out='scores_test.png')


        return df_metrics
# ----------------------------------------------------------------------------------------------------------------------
    def E2E_train_test_df(self,df_train,df_test=None,df_val=None,idx_target=0,do_charts=False,do_density=False,do_pca = False,idx_columns=None,description=''):

        if df_test is None:
            df_train, df_test = train_test_split(df_train, test_size=0.5,shuffle=True)

        df_train = tools_DF.remove_dups(df_train.dropna())
        df_test = tools_DF.remove_dups(df_test.dropna())

        if idx_columns is not None:
            df_train = df_train.iloc[:,[idx_target]+idx_columns]
            df_test  = df_test.iloc[:, [idx_target]+idx_columns]
            idx_target = 0

        self.multiclass = len(set(df_train.iloc[:, idx_target].unique()).union(set(df_test.iloc[:, idx_target].unique())))>2
        self.learn_df(df_train.dropna(),idx_target)

        self.P.set_color(0, self.P.color_red)
        self.P.set_color(1, self.P.color_blue)

        if self.multiclass:
            df_metrics = self.construct_metrics_multiclass(df_train, df_test, df_val, idx_target, do_charts, description)
        else:
            df_metrics = self.construct_metrics_2_classes(df_train, df_test, df_val, idx_target, do_charts, description)

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
            self.P.plot_PCA(df_train, idx_target,filename_out=description+'PCA.png')

        return df_metrics
# ----------------------------------------------------------------------------------------------------------------------
    def do_export(self):
        ext = '.pkl'
        uuid4_out = uuid.uuid4().hex

        if self.do_scaling:
            joblib.dump(self.scaler, self.folder_out+uuid4_out+'_s'+ext)

        joblib.dump(self.classifier.model,self.folder_out+uuid4_out+'_c'+ext)

        return uuid4_out
# ----------------------------------------------------------------------------------------------------------------------
    def do_import(self,filename):
        ext = '.pkl'
        if os.path.isfile(filename+'_s'+ext):
            self.do_scaling = True
            self.scaler = joblib.load(filename+'_s'+ext)

        if os.path.isfile(filename + '_c' + ext):
            self.classifier.model = joblib.load(filename+'_c'+ext)

        return
# ----------------------------------------------------------------------------------------------------------------------