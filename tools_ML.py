import cv2
import os
import numpy
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
#import tools_CNN_view
# ----------------------------------------------------------------------------------------------------------------------
class tools_ML(object):
    def __init__(self,Classifier):
        self.classifier = Classifier
        return
# ---------------------------------------------------------------------------------------------------------------------
    def predict(self, m_class, X_test, cutoff_best=0.0, cutoff_challangers=0.20):

        if (cutoff_best > cutoff_challangers):
            cutoff_best = cutoff_challangers

        n = X_test.shape[0]
        m = numpy.array(list(set(m_class))).shape[0]
        challangers = numpy.empty((n, m), dtype=numpy.chararray)
        challangers[:] = "--"

        prob = numpy.array(self.classifier.predict(X_test))

        label_prob = numpy.max(prob, axis=1)
        label_pred = m_class[numpy.argmax(prob, axis=1)]

        for i in range(0, n):
            for j in range(0, m):
                challangers[i, j] = m_class[j]

        return label_pred, numpy.array(label_prob), challangers, prob
# ---------------------------------------------------------------------------------------------------------------------
    def prepare_arrays_from_feature_files(self, path_input, patterns=numpy.array(['0', '1']),feature_mask='.txt', limit=1000000,has_header=True,has_labels_first_col=True):

        x = tools_IO.load_mat(path_input + ('%s%s' % (patterns[0], feature_mask)), numpy.chararray, delim='\t')

        if has_header:x = x[1:,:]

        X = numpy.full(x.shape[1], '-').astype(numpy.chararray)
        Y = numpy.array(patterns[0])

        for i in range(0,patterns.shape[0]):
            x = tools_IO.load_mat(path_input + ('%s%s' % (patterns[i], feature_mask)), numpy.chararray, delim='\t')
            if has_header: x = x[1:, :]

            if (limit != 1000000) and (x.shape[0] > limit):
                idx_limit = numpy.sort(numpy.random.choice(x.shape[0], int(limit), replace=False))
                x = x[idx_limit]

            X = numpy.vstack((X, x))
            a = numpy.full(x.shape[0], i)
            Y = numpy.hstack((Y, a))

        X = X[1:]
        Y = Y[1:]
        filenames = X[:, 0].astype(numpy.str)
        X = X[:, 1:].astype(numpy.float32)

        return (X, Y.astype(numpy.int32), filenames)
# ---------------------------------------------------------------------------------------------------------------------
    def prepare_tensors_from_image_folders(self,path_input, patterns, mask = '*.png', limit=1000000,resize_W=8, resize_H =8,grayscaled = False):

        #X = numpy.zeros((1,resize_H, resize_W,3), dtype=numpy.uint8)
        #Y, all_filenames=[0],['-']
        for i in range(0, patterns.shape[0]):
            images, labels, filenames = tools_IO.load_aligned_images_from_folder(path_input+patterns[i]+'/', i, mask=mask, limit=limit,resize_W=resize_W, resize_H =resize_H,grayscaled = grayscaled)
            if i==0:
                X=images
                Y = labels
                all_filenames = filenames
            else:
                X = numpy.concatenate((X,images))
                Y = numpy.concatenate((Y,labels))
                all_filenames = numpy.concatenate((all_filenames,filenames))

        return (X, Y, all_filenames)
# ---------------------------------------------------------------------------------------------------------------------
    def score_feature_file(self, file_test, filename_scrs=None, delimeter='\t', append=0, rand_sel=None,has_header=True,has_labels_first_col=False):

        if not os.path.isfile(file_test):return
        data_test = tools_IO.load_mat(file_test, numpy.chararray, delimeter)
        header, Y_test, X_test= self.preprocess_header(data_test,has_header=has_header,has_labels_first_col=has_labels_first_col)

        Y_test = numpy.array(Y_test, dtype=numpy.float)
        Y_test = numpy.array(Y_test, dtype=numpy.int)

        if rand_sel is not None:
            X_test = X_test[rand_sel]
            Y_test = Y_test[rand_sel]

        score = self.classifier.predict(X_test)
        score = (100 * score[:, 1]).astype(int)

        if (filename_scrs != None):
            tools_IO.save_labels(filename_scrs, Y_test, score, append, delim=delimeter)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def train_test(self, X, Y, idx_train, idx_test,path_output=None):

        self.classifier.learn(X[idx_train], Y[idx_train])
        (labels_test_pred, labels_test_prob, challangers_test,challangers_test_prob) = self.predict(numpy.unique(Y), X[idx_test])

        self.classifier.learn(X[idx_test], Y[idx_test])
        (labels_train_pred, labels_train_prob, challangers_train,challangers_train_prob) = self.predict(numpy.unique(Y), X[idx_train])

        return (labels_train_pred, labels_train_prob, challangers_train,challangers_train_prob,labels_test_pred, labels_test_prob, challangers_test,challangers_test_prob)
# ---------------------------------------------------------------------------------------------------------------------
    def learn_on_pos_neg_files(self, file_train_pos, file_train_neg, delimeter='\t', rand_pos=None, rand_neg=None,has_header=True,has_labels_first_col=False):

        X_train_pos = (tools_IO.load_mat(file_train_pos, numpy.chararray, delimeter)).astype(numpy.str)
        X_train_neg = (tools_IO.load_mat(file_train_neg, numpy.chararray, delimeter)).astype(numpy.str)

        if has_header:
            X_train_pos = X_train_pos[1:, :]
            X_train_neg = X_train_neg[1:, :]

        if has_labels_first_col:
            X_train_pos = X_train_pos[:, 1:]
            X_train_neg = X_train_neg[:, 1:]

        X_train_neg = X_train_neg.astype('float32')
        X_train_pos = X_train_pos.astype('float32')



        if rand_pos != []:
            X_train_pos = X_train_pos[rand_pos]

        if rand_neg != []:
            X_train_neg = X_train_neg[rand_neg]

        X_train = numpy.vstack((X_train_pos, X_train_neg))
        Y_train = numpy.hstack((numpy.full(X_train_pos.shape[0], +1), numpy.full(X_train_neg.shape[0], -1)))

        self.classifier.learn(X_train, Y_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def generate_data_grid(self,filename_data_train, filename_data_test, filename_data_grid,has_header=True,has_labels_first_col=True):

        data = tools_IO.load_mat(filename_data_train, numpy.chararray, '\t')
        if has_header:data = data[1:,:]
        if has_labels_first_col:data = data[:, 1:]
        X_train = data.astype('float32')

        data = tools_IO.load_mat(filename_data_test, numpy.chararray, '\t')
        if has_header:data = data[1:,:]
        if has_labels_first_col:data = data[:, 1:]
        X_test = data.astype('float32')

        X = numpy.vstack((X_train, X_test))

        if (X.shape[1] != 2):
            return

        minx = numpy.min(X[:, 0])
        maxx = numpy.max(X[:, 0])
        miny = numpy.min(X[:, 1])
        maxy = numpy.max(X[:, 1])
        step = 10

        data_grid = numpy.zeros((step * step, 2))
        target_grid = numpy.zeros(step * step)

        for s1 in range(0, step):
            for s2 in range(0, step):
                data_grid[s1 + (step - 1 - s2) * step, 0] = minx + s1 * (maxx - minx) / (step - 1)
                data_grid[s1 + (step - 1 - s2) * step, 1] = miny + s2 * (maxy - miny) / (step - 1)

        tools_IO.save_data_to_feature_file_float(filename_data_grid, data_grid, target_grid)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def get_th_pos_neg(self, filename_scores_pos, filename_scores_neg, delim='\t'):

        data = tools_IO.load_mat(filename_scores_pos, numpy.chararray, delim)[1:, :]
        scores1 = (data[:, 1:]).astype('float32')
        labels1 = numpy.full(len(scores1), 1)

        data = tools_IO.load_mat(filename_scores_neg, numpy.chararray, delim)[1:, :]
        scores0 = (data[:, 1:]).astype('float32')
        labels0 = numpy.full(len(scores0), 1)

        labels = numpy.hstack((labels1, labels0)).astype(int)
        scores = numpy.vstack((scores1, scores0))

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        v = numpy.argmax(tpr + (1 - fpr))
        th = thresholds[v]

        return th
# ---------------------------------------------------------------------------------------------------------------------
    def get_th_train(self, filename_scores_train,delim='\t',has_header=True):
        X = tools_IO.load_mat(filename_scores_train, numpy.chararray, '\t')

        if has_header:
            X = X[1:, :]
        else:
            X = X[:, :]

        labels = (X[:, 0]).astype('float32')
        scores = X[:, 1:].astype('float32')

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        v = numpy.argmax(tpr + (1 - fpr))
        th = thresholds[v]
        return th
    # ---------------------------------------------------------------------------------------------------------------------
    def stage_train_stats(self, path_output, labels_fact, labels_train_pred, labels_test_pred, labels_train_prob, labels_test_prob, patterns,X=None,Y=None,verbose=False):

        labels_pred   = numpy.hstack((labels_train_pred, labels_test_pred))
        labels_prob   = numpy.hstack((labels_train_prob, labels_test_prob))

        predictions = numpy.array([patterns[labels_fact],patterns[labels_pred], labels_prob]).T
        tools_IO.save_mat(predictions, path_output + self.classifier.name + '_predictions.txt')
        tools_IO.print_accuracy    (labels_fact, labels_pred, patterns)
        tools_IO.print_accuracy    (labels_fact, labels_pred, patterns, filename=path_output + self.classifier.name + '_confusion_mat.txt')
        tools_IO.print_top_fails   (labels_fact, labels_pred, patterns, filename=path_output + self.classifier.name + '_errors.txt')
        tools_IO.print_reject_rate (labels_fact, labels_pred, labels_prob, filename=path_output + self.classifier.name + '_accuracy.txt')

        if verbose == True:
            verbose_PCA = True if (X is not None) and (Y is not None) else False
            #verbose_PCA = False

            if verbose_PCA:
                print('Extracting features for PCA')
                features = self.classifier.images_to_features(X)

            fig = plt.figure(figsize=(12, 6))
            fig.subplots_adjust(hspace=0.01)
            if verbose_PCA:
                tools_IO.plot_features_PCA   (plt.subplot(1, 3, 1),features,Y,patterns)
                tools_IO.plot_learning_rates1(plt.subplot(1, 3, 2), fig,filename_mat=path_output + self.classifier.name + '_learn_rates.txt')
                tools_IO.plot_confusion_mat  (plt.subplot(1, 3, 3), fig, filename_mat=path_output + self.classifier.name + '_predictions.txt',caption=self.classifier.name)
            else:
                tools_IO.plot_learning_rates1(plt.subplot(1, 2, 1), fig,filename_mat=path_output + self.classifier.name + '_learn_rates.txt')
                tools_IO.plot_confusion_mat(  plt.subplot(1, 2, 2), fig,filename_mat=path_output + self.classifier.name + '_predictions.txt',caption=self.classifier.name)

            plt.tight_layout()
            plt.show()


        return
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_images(self,path_input, path_output, mask = '*.png', limit_classes=1000000,limit_instances=1000000,resize_W=8, resize_H =8,grayscaled = False,verbose=True):
        print('E2E train-test on images: classifier=%s\n\n' % (self.classifier.name))
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))[:limit_classes]



        (X, Y, filenames) = self.prepare_tensors_from_image_folders(path_input, patterns,mask= mask,limit=limit_instances,resize_W=resize_W, resize_H =resize_H,grayscaled = grayscaled)
        #X = tools_CNN_view.normalize(X.astype(numpy.float32))
        idx_train = numpy.sort(numpy.random.choice(X.shape[0], int(X.shape[0] / 2), replace=False))
        idx_test  = numpy.array([x for x in range(0, X.shape[0]) if x not in idx_train])

        (labels_train_pred, labels_train_prob, challangers_train, challangers_train_prob, labels_test_pred,labels_test_prob, challangers_test, challangers_test_prob) = \
            self.train_test(X, Y, idx_train, idx_test,path_output)

        labels_fact = numpy.hstack((Y[idx_train], Y[idx_test]))

        self.stage_train_stats(path_output, labels_fact, labels_train_pred, labels_test_pred, labels_train_prob, labels_test_prob, patterns,X=X,Y=Y,verbose=verbose)



        return
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_features(self, path_input, path_output,mask = '.txt', limit_classes=1000000,limit_instances=1000000,has_header=True,has_labels_first_col=True):

        print('E2E train-test on features: classifier=%s\n\n' % (self.classifier.name))
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        patterns = numpy.unique(numpy.array([f.path[len(path_input):].split(mask)[0] for f in os.scandir(path_input) if f.is_file()]))[:limit_classes]

        (X, Y, filenames) = self.prepare_arrays_from_feature_files(path_input, patterns,feature_mask=mask,limit=limit_instances,has_header=has_header,has_labels_first_col=has_labels_first_col)
        idx_train = numpy.sort(numpy.random.choice(X.shape[0], int(X.shape[0] / 2), replace=False))
        idx_test  = numpy.array([x for x in range(0, X.shape[0]) if x not in idx_train])

        if has_header:
            header = tools_IO.load_mat(path_input + ('%s%s' % (patterns[0], mask)), numpy.chararray, delim='\t')
            header = header[0]
        else:
            header = None

        X = normalize(X)
        min = numpy.min(X)
        X -= min
        max = numpy.max(X)
        X*=255.0/(max)

        (labels_train_pred, labels_train_prob, challangers_train, challangers_train_prob, labels_test_pred,labels_test_prob, challangers_test, challangers_test_prob) = \
            self.train_test(X, Y, idx_train, idx_test)

        labels_fact = numpy.hstack((Y[idx_train], Y[idx_test]))

        self.stage_train_stats(path_output, labels_fact, labels_train_pred, labels_test_pred, labels_train_prob, labels_test_prob, patterns)

        N = 3
        if not has_header:N-=1

        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(hspace=0.01)
        tools_IO.plot_features_PCA(plt.subplot(1, N, 1), X, Y, patterns)
        tools_IO.plot_confusion_mat(plt.subplot(1, N, 2),fig,path_output + self.classifier.name + '_predictions.txt',self.classifier.name)
        if has_header:
            tools_IO.plot_feature_importance(plt.subplot(1, N, 3), fig, X, Y, header)
        plt.tight_layout()
        plt.savefig(path_output + 'fig_roc.png')

        return
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_features_2_classes_dim_2(self, folder_out, filename_data_pos, filename_data_neg,has_header=True,has_labels_first_col=True):

        tools_IO.remove_files(folder_out)

        filename_scrs_pos = folder_out + 'scores_pos_'+ self.classifier.name + '.txt'
        filename_scrs_neg = folder_out + 'scores_neg_'+ self.classifier.name + '.txt'

        filename_data_grid = folder_out + 'data_grid.txt'
        filename_scores_grid = folder_out + 'scores_grid.txt'

        Pos = (tools_IO.load_mat(filename_data_pos, numpy.chararray, '\t')).shape[0]
        Neg = (tools_IO.load_mat(filename_data_neg, numpy.chararray, '\t')).shape[0]

        if has_header:
            Pos-=1
            Neg-=1

        numpy.random.seed(125)
        idx_pos_train = numpy.random.choice(Pos, int(Pos / 2), replace=False)
        idx_neg_train = numpy.random.choice(Neg, int(Neg / 2), replace=False)
        idx_pos_test = [x for x in range(0, Pos) if x not in idx_pos_train]
        idx_neg_test = [x for x in range(0, Neg) if x not in idx_neg_train]

        self.generate_data_grid(filename_data_pos, filename_data_neg, filename_data_grid)

        self.learn_on_pos_neg_files(filename_data_pos,filename_data_neg, delimeter='\t', rand_pos=idx_pos_train,rand_neg=idx_neg_train,has_header=has_header,has_labels_first_col=has_labels_first_col)
        self.score_feature_file(filename_data_pos, filename_scrs  =filename_scrs_pos,delimeter='\t', append=0, rand_sel=idx_pos_test,has_header=has_header,has_labels_first_col=has_labels_first_col)
        self.score_feature_file(filename_data_neg, filename_scrs  =filename_scrs_neg,delimeter='\t', append=0, rand_sel=idx_neg_test,has_header=has_header,has_labels_first_col=has_labels_first_col)

        self.learn_on_pos_neg_files(filename_data_pos,filename_data_neg, '\t', idx_pos_test,idx_neg_test,has_header=has_header,has_labels_first_col=has_labels_first_col)
        self.score_feature_file(filename_data_pos,  filename_scrs = filename_scrs_pos,delimeter='\t',append= 1,rand_sel=idx_pos_train,has_header=has_header,has_labels_first_col=has_labels_first_col)
        self.score_feature_file(filename_data_neg,  filename_scrs = filename_scrs_neg,delimeter='\t',append= 1,rand_sel=idx_neg_train,has_header=has_header,has_labels_first_col=has_labels_first_col)

        self.score_feature_file(filename_data_grid, filename_scrs=filename_scores_grid,has_header=has_header,has_labels_first_col=has_labels_first_col)

        tpr, fpr, auc = tools_IO.get_roc_data_from_scores_file_v2(filename_scrs_pos, filename_scrs_neg, delim='\t')

        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(hspace=0.01)

        th = self.get_th_pos_neg(filename_scrs_pos, filename_scrs_neg, delim='\t')
        tools_IO.plot_2D_scores(plt.subplot(1, 3, 1), fig, filename_data_pos, filename_data_neg,filename_data_grid, filename_scores_grid, th, noice_needed=1,caption=self.classifier.name + ' %1.2f' % auc)
        tools_IO.display_distributions(plt.subplot(1, 3, 2), fig, filename_scrs_pos, filename_scrs_neg, delim='\t')
        tools_IO.display_roc_curve_from_descriptions(plt.subplot(1, 3, 3), fig, filename_scrs_pos, filename_scrs_neg,delim='\t')
        plt.tight_layout()
        plt.savefig(folder_out + 'fig_roc.png')

        return tpr, fpr, auc
# ---------------------------------------------------------------------------------------------------------------------
    def check_corr(self,X,Y,header):
        N = X.shape[1]
        C = numpy.zeros(N)
        for i in range(N):C[i] = math.fabs(numpy.corrcoef(X[:, i], Y)[0,1])
        idx = numpy.argsort(-C)

        H = numpy.array(header,dtype=numpy.str)

        for i in range(N):
            print('%1.2f %s'%(C[idx[i]],str(H[idx[i]])))
        return
# ---------------------------------------------------------------------------------------------------------------------
    def preprocess_header(self,X,has_header,has_labels_first_col):

        header, first_col = None,None

        if has_header:
            header = numpy.array(X[0,:],dtype=numpy.str)
            result  = X[1:,:]
        else:
            header = None
            result = X[:,:]

        if has_labels_first_col:
            first_col = result[:, 0]
            result = result[:, 1:]
            if header is not None:
                header = header [1:]

        result = numpy.array(result,dtype=numpy.float)

        return header, first_col, result
# ---------------------------------------------------------------------------------------------------------------------
    def plot_results_pos_neg(self, x_pos, x_neg, header, filename_scrs_pos, filename_scrs_neg, filename_out):


        X = numpy.vstack((x_pos, x_neg)).astype(numpy.float)
        Y = numpy.hstack((numpy.full(x_pos.shape[0], 1), numpy.full(x_neg.shape[0], 0)))

        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(hspace=0.01)
        tools_IO.display_roc_curve_from_descriptions(plt.subplot(2, 2, 3), fig, filename_scrs_pos, filename_scrs_neg, delim='\t')
        tools_IO.display_distributions(plt.subplot(2, 2, 2), fig, filename_scrs_pos, filename_scrs_neg, delim='\t')
        #tools_IO.plot_features_PCA(plt.subplot(2, 2, 1),X,Y,['pos','neg'])
        if header is not None:
            tools_IO.plot_feature_importance(plt.subplot(2,2,4),fig,X,Y,header)
        plt.tight_layout()
        plt.savefig(filename_out)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def plot_results_train_test(self,filename_scores_train,filename_scores_test,filename_out,has_header=False):

        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(hspace=0.01)

        tpr, fpr, auc = tools_IO.get_roc_data_from_scores_file(filename_scores_train,has_header=has_header)
        tools_IO.plot_tp_fp(plt.subplot(1, 2, 1),fig,tpr,fpr,auc,caption='Train',filename_out=None)

        tpr, fpr, auc = tools_IO.get_roc_data_from_scores_file(filename_scores_test, has_header=has_header)
        tools_IO.plot_tp_fp(plt.subplot(1, 2, 2), fig, tpr, fpr, auc, caption='Test', filename_out=None)

        plt.tight_layout()
        plt.savefig(filename_out)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def draw_GT_pred(self,filename_scrs, filename_out, th, has_header=True):


        X = tools_IO.load_mat(filename_scrs, numpy.chararray, '\t')

        if has_header:
            X = X[1:, :]
        else:
            X = X[:, :]

        labels = (X[:, 0]).astype('float32')
        scores = X[:, 1:].astype('float32')

        H = 100
        color_GT = (64, 128, 0)
        color_pred  =(190, 128, 0)

        image = numpy.full((H, X.shape[0], 3), 0xC0)
        for c,label in enumerate(labels):
            if label<=0:
                image[70:75,c]= color_GT
            else:
                image[20:25,c] = color_GT


        for c,score in enumerate(scores):
            if score<=th:
                image[76:81,c]=color_pred
            else:
                image[14:19,c] = color_pred

        #image = cv2.flip(image,0)

        cv2.imwrite(filename_out,image)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_features_2_classes_multi_dim(self, folder_out, filename_data_pos, filename_data_neg,has_header=True,has_labels_first_col=True):

        tools_IO.remove_files(folder_out)

        filename_scrs_pos = folder_out + 'scores_pos_'+ self.classifier.name + '.txt'
        filename_scrs_neg = folder_out + 'scores_neg_'+ self.classifier.name + '.txt'

        data_pos = tools_IO.load_mat(filename_data_pos, numpy.chararray, '\t')
        data_neg = tools_IO.load_mat(filename_data_neg, numpy.chararray, '\t')

        header,first_col, x_pos = self.preprocess_header(data_pos, has_header, has_labels_first_col)
        header,first_col, x_neg = self.preprocess_header(data_neg, has_header, has_labels_first_col)

        numpy.random.seed(125)
        idx_pos_train = numpy.random.choice(x_pos.shape[0], int(x_pos.shape[0]/2),replace=False)
        idx_neg_train = numpy.random.choice(x_neg.shape[0], int(x_neg.shape[0]/2),replace=False)
        idx_pos_test = [x for x in range(0, x_pos.shape[0]) if x not in idx_pos_train]
        idx_neg_test = [x for x in range(0, x_neg.shape[0]) if x not in idx_neg_train]

        self.learn_on_pos_neg_files(filename_data_pos,filename_data_neg, delimeter='\t', rand_pos=idx_pos_train,rand_neg=idx_neg_train,has_header=has_header,has_labels_first_col=has_labels_first_col)
        self.score_feature_file(filename_data_pos, filename_scrs  =filename_scrs_pos,delimeter='\t', append=0, rand_sel=idx_pos_test,has_header=has_header,has_labels_first_col=has_labels_first_col)
        self.score_feature_file(filename_data_neg, filename_scrs  =filename_scrs_neg,delimeter='\t', append=0, rand_sel=idx_neg_test,has_header=has_header,has_labels_first_col=has_labels_first_col)

        self.learn_on_pos_neg_files(filename_data_pos,filename_data_neg, '\t', idx_pos_test,idx_neg_test,has_header=has_header,has_labels_first_col=has_labels_first_col)
        self.score_feature_file(filename_data_pos,  filename_scrs = filename_scrs_pos,delimeter='\t',append= 1,rand_sel=idx_pos_train,has_header=has_header,has_labels_first_col=has_labels_first_col)
        self.score_feature_file(filename_data_neg,  filename_scrs = filename_scrs_neg,delimeter='\t',append= 1,rand_sel=idx_neg_train,has_header=has_header,has_labels_first_col=has_labels_first_col)

        self.plot_results_pos_neg(x_pos, x_neg, header, filename_scrs_pos, filename_scrs_neg, folder_out + 'fig_roc.png')

        return# tpr, fpr, auc
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_features_2_classes_multi_dim_train_test(self,folder_out, filename_train, filename_test,has_header=True,has_labels_first_col=True):
        tools_IO.remove_files(folder_out)

        data_train = tools_IO.load_mat(filename_train, numpy.chararray, '\t')
        data_test  = tools_IO.load_mat(filename_test, numpy.chararray, '\t')

        header, Y_train, X_train = self.preprocess_header(data_train, has_header, has_labels_first_col=False)
        header, Y_test , X_test  = self.preprocess_header(data_test, has_header, has_labels_first_col=False)

        data_pos, data_neg = [],[]
        idx_pos_train, idx_neg_train, idx_pos_test, idx_neg_test = [], [], [], []
        cnt_pos_train, cnt_neg_train, cnt_pos_test, cnt_neg_test = 0,0,0,0


        for x in X_train:
            if x[0]>0:
                data_pos.append(x)
                idx_pos_train.append(cnt_pos_train)
                cnt_pos_train+=1
            else:
                data_neg.append(x)
                idx_neg_train.append(cnt_neg_train)
                cnt_neg_train+=1

        offset_pos = len(data_pos)
        offset_neg = len(data_neg)

        for x in X_test:
            if x[0]>0:
                data_pos.append(x)
                idx_pos_test.append(offset_pos+cnt_pos_test)
                cnt_pos_test+=1
            else:
                data_neg.append(x)
                idx_neg_test.append(offset_neg+cnt_neg_test)
                cnt_neg_test+=1

        data_pos = numpy.array(data_pos)
        data_neg = numpy.array(data_neg)
        idx_pos_train,idx_neg_train,idx_pos_test,idx_neg_test = numpy.array(idx_pos_train),numpy.array(idx_pos_train),numpy.array(idx_pos_test),numpy.array(idx_neg_test)

        filename_data_pos = folder_out + 'temp_pos.txt'
        filename_data_neg = folder_out + 'temp_neg.txt'
        tools_IO.save_mat(data_pos, filename_data_pos)
        tools_IO.save_mat(data_neg, filename_data_neg)

        self.learn_on_pos_neg_files(filename_data_pos,filename_data_neg, delimeter='\t', rand_pos=idx_pos_train,rand_neg=idx_neg_train,has_header=has_header,has_labels_first_col=has_labels_first_col)

        filename_scrs_train = folder_out + 'scores_train_' + self.classifier.name + '.txt'
        filename_scrs_test  = folder_out + 'scores_test_' + self.classifier.name + '.txt'
        self.score_feature_file(filename_train, filename_scrs=filename_scrs_train, delimeter='\t', append=0,has_header=has_header, has_labels_first_col=has_labels_first_col)
        self.score_feature_file(filename_test , filename_scrs=filename_scrs_test , delimeter='\t', append=0,has_header=has_header, has_labels_first_col=has_labels_first_col)

        self.plot_results_train_test(filename_scrs_train,filename_scrs_test,folder_out + 'fig_roc_train.png',has_header=True)

        th = self.get_th_train(filename_scrs_train,has_header=True)

        self.draw_GT_pred(filename_scrs_train,folder_out + 'GT_train.png',th,has_header=True)
        self.draw_GT_pred(filename_scrs_test, folder_out + 'GT_test.png', th, has_header=True)

        return
# ---------------------------------------------------------------------------------------------------------------------
