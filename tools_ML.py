import os
import numpy
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import normalize
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_CNN_view
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
    def prepare_arrays_from_feature_files(self, path_input, patterns=numpy.array(['0', '1']),feature_mask='.txt', limit=1000000):

        x = tools_IO.load_mat(path_input + ('%s%s' % (patterns[0], feature_mask)), numpy.chararray, delim='\t')

        X = numpy.full(x.shape[1], '-').astype(numpy.chararray)
        Y = numpy.array(patterns[0])

        for i in range(0,patterns.shape[0]):
            x = tools_IO.load_mat(path_input + ('%s%s' % (patterns[i], feature_mask)), numpy.chararray, delim='\t')
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
    def score_feature_file(self, file_test, filename_scrs=None, delimeter='\t', append=0, rand_sel=[]):

        if not os.path.isfile(file_test):return

        data_test = tools_IO.load_mat(file_test, numpy.chararray, delimeter)
        data_test = data_test[:, :]

        labels_test = (data_test[:, 0]).astype(numpy.str)
        data_test = data_test[:, 1:]

        if data_test[0, -1] == b'':
            data_test = data_test[:, :-1]

        data_test = data_test.astype('float32')

        if rand_sel != []:
            data_test = data_test[rand_sel]
            labels_test = labels_test[rand_sel]

        score = self.classifier.predict(data_test)
        score = (100 * score[:, 1]).astype(int)

        if (filename_scrs != None):
            tools_IO.save_labels(filename_scrs, labels_test, score, append, delim=' ')
        return
# ---------------------------------------------------------------------------------------------------------------------
    def train_test(self, X, Y, idx_train, idx_test,path_output=None):

        self.classifier.learn(X[idx_train], Y[idx_train])
        (labels_test_pred, labels_test_prob, challangers_test,challangers_test_prob) = self.predict(numpy.unique(Y), X[idx_test])

        self.classifier.learn(X[idx_test], Y[idx_test])
        (labels_train_pred, labels_train_prob, challangers_train,challangers_train_prob) = self.predict(numpy.unique(Y), X[idx_train])

        return (labels_train_pred, labels_train_prob, challangers_train,challangers_train_prob,labels_test_pred, labels_test_prob, challangers_test,challangers_test_prob)
# ---------------------------------------------------------------------------------------------------------------------
    def learn_on_pos_neg_files(self, file_train_pos, file_train_neg, delimeter='\t', rand_pos=None, rand_neg=None):

        X_train_pos = (tools_IO.load_mat(file_train_pos, numpy.chararray, delimeter)).astype(numpy.str)
        X_train_neg = (tools_IO.load_mat(file_train_neg, numpy.chararray, delimeter)).astype(numpy.str)

        X_train_pos = (X_train_pos[:, 1:])

        X_train_pos = X_train_pos.astype('float32')
        X_train_neg = X_train_neg[:, 1:].astype('float32')

        if rand_pos != []:
            X_train_pos = X_train_pos[rand_pos]

        if rand_neg != []:
            X_train_neg = X_train_neg[rand_neg]

        X_train = numpy.vstack((X_train_pos, X_train_neg))
        Y_train = numpy.hstack((numpy.full(X_train_pos.shape[0], +1), numpy.full(X_train_neg.shape[0], -1)))

        self.classifier.learn(X_train, Y_train)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def generate_data_grid(self,filename_data_train, filename_data_test, filename_data_grid):

        data = tools_IO.load_mat(filename_data_train, numpy.chararray, '\t')
        X_train = data[:, 1:].astype('float32')

        data = tools_IO.load_mat(filename_data_test, numpy.chararray, '\t')
        X_test = data[:, 1:].astype('float32')

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
    def get_th(self,filename_scores_pos, filename_scores_neg,delim='\t'):

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
        X = tools_CNN_view.normalize(X.astype(numpy.float32))
        idx_train = numpy.sort(numpy.random.choice(X.shape[0], int(X.shape[0] / 2), replace=False))
        idx_test  = numpy.array([x for x in range(0, X.shape[0]) if x not in idx_train])

        (labels_train_pred, labels_train_prob, challangers_train, challangers_train_prob, labels_test_pred,labels_test_prob, challangers_test, challangers_test_prob) = \
            self.train_test(X, Y, idx_train, idx_test,path_output)

        labels_fact = numpy.hstack((Y[idx_train], Y[idx_test]))

        self.stage_train_stats(path_output, labels_fact, labels_train_pred, labels_test_pred, labels_train_prob, labels_test_prob, patterns,X=X,Y=Y,verbose=verbose)



        return
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_features(self, path_input, path_output,mask = '.txt', limit_classes=1000000,limit_instances=1000000):

        print('E2E train-test on features: classifier=%s\n\n' % (self.classifier.name))
        if not os.path.exists(path_output):
            os.makedirs(path_output)



        patterns = numpy.unique(numpy.array([f.path[len(path_input):].split(mask)[0] for f in os.scandir(path_input) if f.is_file()]))[:limit_classes]

        (X, Y, filenames) = self.prepare_arrays_from_feature_files(path_input, patterns,feature_mask=mask,limit=limit_instances)
        idx_train = numpy.sort(numpy.random.choice(X.shape[0], int(X.shape[0] / 2), replace=False))
        idx_test  = numpy.array([x for x in range(0, X.shape[0]) if x not in idx_train])

        X = normalize(X)
        min = numpy.min(X)
        X -= min
        max = numpy.max(X)
        X*=255.0/(max)

        (labels_train_pred, labels_train_prob, challangers_train, challangers_train_prob, labels_test_pred,labels_test_prob, challangers_test, challangers_test_prob) = \
            self.train_test(X, Y, idx_train, idx_test)

        labels_fact = numpy.hstack((Y[idx_train], Y[idx_test]))

        self.stage_train_stats(path_output, labels_fact, labels_train_pred, labels_test_pred, labels_train_prob, labels_test_prob, patterns)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_features_2_classes(self, folder_out, filename_data_pos, filename_data_neg, filename_scrs_pos=None, filename_scrs_neg=None, fig=None):

        tools_IO.remove_files(folder_out)

        filename_data_grid = folder_out+'data_grid.txt'
        filename_scores_grid = folder_out+'scores_grid.txt'


        Pos = (tools_IO.load_mat(filename_data_pos, numpy.chararray, '\t')).shape[0]
        Neg = (tools_IO.load_mat(filename_data_neg, numpy.chararray, '\t')).shape[0]

        numpy.random.seed(125)
        idx_pos_train = numpy.random.choice(Pos, int(Pos/2),replace=False)
        idx_neg_train = numpy.random.choice(Neg, int(Neg/2),replace=False)
        idx_pos_test = [x for x in range(0, Pos) if x not in idx_pos_train]
        idx_neg_test = [x for x in range(0, Neg) if x not in idx_neg_train]

        self.generate_data_grid(filename_data_pos, filename_data_neg, filename_data_grid)

        self.learn_on_pos_neg_files(filename_data_pos,filename_data_neg, delimeter='\t', rand_pos=idx_pos_train,rand_neg=idx_neg_train)
        self.score_feature_file(filename_data_pos, filename_scrs  =filename_scrs_pos,delimeter='\t', append=0, rand_sel=idx_pos_test)
        self.score_feature_file(filename_data_neg, filename_scrs  =filename_scrs_neg,delimeter='\t', append=0, rand_sel=idx_neg_test)
        self.score_feature_file(filename_data_grid, filename_scrs =filename_scores_grid)

        self.learn_on_pos_neg_files(filename_data_pos,filename_data_neg, '\t', idx_pos_test,idx_neg_test)
        self.score_feature_file(filename_data_pos,  filename_scrs = filename_scrs_pos,delimeter='\t',append= 1,rand_sel=idx_pos_train)
        self.score_feature_file(filename_data_neg,  filename_scrs = filename_scrs_neg,delimeter='\t',append= 1,rand_sel=idx_neg_train)

        tpr, fpr, auc = tools_IO.get_roc_data_from_scores_file_v2(filename_scrs_pos, filename_scrs_neg,delim=' ')


        if(fig!=None):
            th = self.get_th(filename_scrs_pos, filename_scrs_neg,delim=' ')
            tools_IO.display_roc_curve_from_descriptions(plt.subplot(1, 3, 3), fig, filename_scrs_pos, filename_scrs_neg, delim=' ')
            tools_IO.display_distributions(plt.subplot(1, 3, 2), fig, filename_scrs_pos, filename_scrs_neg, delim=' ')
            tools_IO.plot_2D_scores(plt.subplot(1, 3, 1), fig, filename_data_pos, filename_data_neg, filename_data_grid, filename_scores_grid, th, noice_needed=1, caption=self.classifier.name + ' %1.2f' % auc)
            plt.tight_layout()
            plt.show()

        return tpr, fpr, auc
# ---------------------------------------------------------------------------------------------------------------------
