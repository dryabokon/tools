import os
from os import listdir
import numpy
import fnmatch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import normalize
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
class tools_ML(object):
    def __init__(self,Classifier):
        self.classifier = Classifier
        return
# ---------------------------------------------------------------------------------------------------------------------
    def predict_arrays_all_all(self, m_class, X_test, cutoff_best=0.0, cutoff_challangers=0.20):

        if (cutoff_best > cutoff_challangers):
            cutoff_best = cutoff_challangers

        n = X_test.shape[0]
        m = numpy.array(list(set(m_class))).shape[0]
        challangers = numpy.empty((n, m), dtype=numpy.chararray)
        challangers[:] = "--"

        prob = numpy.array(self.classifier.predict_probability_of_array(X_test))

        label_prob = numpy.max(prob, axis=1)
        label_pred = m_class[numpy.argmax(prob, axis=1)]

        idx = numpy.array([i for i, v in enumerate(label_prob) if ((v < cutoff_best))])
        if idx.shape[0] > 0:
            label_pred[idx] = "-"

        for i in range(0, n):
            for j in range(0, m):
                challangers[i, j] = m_class[j]

        return label_pred, numpy.array(label_prob), challangers, prob
# ---------------------------------------------------------------------------------------------------------------------
    def prepare_arrays_from_feature_files(self, path_input, patterns=numpy.array(['0', '1']),feature_mask='_features.txt', limit=1000000):

        x = tools_IO.load_mat(path_input + ('%s%s' % (patterns[0], feature_mask)), numpy.chararray, delim='\t')

        X = numpy.full(x.shape[1], '-').astype(numpy.chararray)
        Y = numpy.array(patterns[0])
        filenames = []
        i = 0
        for each in patterns:
            print('.', end='', flush=True)
            x = tools_IO.load_mat(path_input + ('%s%s' % (each, feature_mask)), numpy.chararray, delim='\t')
            if (limit != 1000000) and (x.shape[0] > limit):
                idx_limit = numpy.sort(numpy.random.choice(x.shape[0], int(limit), replace=False))
                x = x[idx_limit]

            X = numpy.vstack((X, x))
            a = numpy.full(x.shape[0], i)
            Y = numpy.hstack((Y, a))
            i = i + 1

        X = X[1:]
        Y = Y[1:]
        filenames = X[:, 0].astype(numpy.str)
        X = X[:, 1:].astype(numpy.float32)

        M = X.shape[0]
        # numpy.random.seed(124)
        idx_train = numpy.sort(numpy.random.choice(M, int(M / 2), replace=False))
        idx_test = numpy.array([x for x in range(0, M) if x not in idx_train])

        print('\n')
        return (X, Y.astype(numpy.int32), idx_train, idx_test, filenames)

# ---------------------------------------------------------------------------------------------------------------------
    def score_feature_file(self, file_test, filename_scrs=None, delimeter='\t', append=0, rand_sel=[]):

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

        score = self.classifier.predict_probability_of_array(data_test)
        score = (100 * score[:, 1]).astype(int)

        if (filename_scrs != None):
            tools_IO.save_labels(filename_scrs, labels_test, score, append, delim=delimeter)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def train_test_on_array(self, X, Y, idx_train, idx_test, path_output):

        self.classifier.learn_on_arrays(X[idx_train], Y[idx_train])
        (labels_test_pred, labels_test_prob, challangers_test,challangers_test_prob) = self.predict_arrays_all_all(numpy.unique(Y), X[idx_test])
        self.classifier.learn_on_arrays(X[idx_test], Y[idx_test])
        (labels_train_pred, labels_train_prob, challangers_train,challangers_train_prob) = self.predict_arrays_all_all(numpy.unique(Y), X[idx_train])

        return (labels_train_pred, labels_train_prob, challangers_train,challangers_train_prob,labels_test_pred, labels_test_prob, challangers_test,challangers_test_prob)
# ---------------------------------------------------------------------------------------------------------------------
    def learn_on_pos_neg_files(self, file_train_pos, file_train_neg, delimeter='\t', rand_pos=[], rand_neg=[]):

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

        model = self.classifier.learn_on_arrays(X_train, Y_train)
        return model
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
    def get_th(self,filename_scores_pos, filename_scores_neg):

        data = tools_IO.load_mat(filename_scores_pos, numpy.chararray, '\t')[1:, :]
        labels1 = (data[:, 0]).astype('float32')
        scores1 = (data[:, 1:]).astype('float32')

        data = tools_IO.load_mat(filename_scores_neg, numpy.chararray, '\t')[1:, :]
        labels2 = (data[:, 0]).astype('float32')
        scores2 = (data[:, 1:]).astype('float32')

        labels = numpy.hstack((labels1, labels2)).astype(int)
        scores = numpy.vstack((scores1, scores2))

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        v = numpy.argmax(tpr + (1 - fpr))
        th = thresholds[v]

        return th
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_features(self, path_input, path_output, limit_classes=1000000,limit_instances=1000000):

        print('E2E train-test on features: classifier=%s\n\n' % (self.classifier.name))
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        patterns = numpy.unique(numpy.array([f.path[len(path_input):].split('_features')[0] for f in os.scandir(path_input) if f.is_file()]))[:limit_classes]

        feature_mask = '_features.txt'
        (X, Y, idx_train, idx_test, filenames) = self.prepare_arrays_from_feature_files(path_input, patterns,feature_mask=feature_mask,limit=limit_instances)
        X = normalize(X)
        min = numpy.min(X)
        X += min
        max = numpy.max(X)
        X*=255/(max)

        (labels_train_pred, labels_train_prob, challangers_train, challangers_train_prob, labels_test_pred,labels_test_prob, challangers_test, challangers_test_prob) = \
            self.train_test_on_array(X, Y, idx_train, idx_test, path_output)

        labels_fact   = numpy.hstack((Y[idx_train], Y[idx_test]))
        labels_pred   = numpy.hstack((labels_train_pred, labels_test_pred))
        labels_prob   = numpy.hstack((labels_train_prob, labels_test_prob))

        predictions = numpy.array([patterns[labels_fact],patterns[labels_pred], labels_prob]).T
        tools_IO.save_mat(predictions, path_output + self.classifier.name + '_predictions.txt')
        tools_IO.print_accuracy    (labels_fact, labels_pred, patterns)
        tools_IO.print_accuracy    (labels_fact, labels_pred, patterns, filename=path_output + self.classifier.name + '_confusion_mat.txt')
        tools_IO.print_top_fails   (labels_fact, labels_pred, patterns, filename=path_output + self.classifier.name + '_errors.txt')
        tools_IO.print_reject_rate (labels_fact, labels_pred, labels_prob, filename=path_output + self.classifier.name + '_accuracy.txt')

        return
# ---------------------------------------------------------------------------------------------------------------------
    def E2E_features_2_classes(self, folder_out, filename_data_pos, filename_data_neg, filename_scrs_pos=None, filename_scrs_neg=None, fig=None):
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

        model = self.learn_on_pos_neg_files(filename_data_pos,filename_data_neg, '\t', idx_pos_train,idx_neg_train)
        self.score_feature_file(filename_data_pos, filename_scrs  =filename_scrs_pos,delimeter='\t', append=0, rand_sel=idx_pos_test)
        self.score_feature_file(filename_data_neg, filename_scrs  =filename_scrs_neg,delimeter='\t', append=0, rand_sel=idx_neg_test)
        self.score_feature_file(filename_data_grid, filename_scrs =filename_scores_grid)

        model = self.learn_on_pos_neg_files(filename_data_pos,filename_data_neg, '\t', idx_pos_test,idx_neg_test)
        self.score_feature_file(filename_data_pos,  filename_scrs = filename_scrs_pos,delimeter='\t',append= 1,rand_sel=idx_pos_train)
        self.score_feature_file(filename_data_neg,  filename_scrs = filename_scrs_neg,delimeter='\t',append= 1,rand_sel=idx_neg_train)

        tpr, fpr, auc = tools_IO.get_roc_data_from_scores_file_v2(filename_scrs_pos, filename_scrs_neg)


        if(fig!=None):
            th = self.get_th(filename_scrs_pos, filename_scrs_neg)
            tools_IO.display_roc_curve_from_descriptions(plt.subplot(1, 3, 3), fig, filename_scrs_pos, filename_scrs_neg, delim='\t')
            tools_IO.display_distributions(plt.subplot(1, 3, 2), fig, filename_scrs_pos, filename_scrs_neg, delim='\t')
            tools_IO.plot_2D_scores(plt.subplot(1, 3, 1), fig, filename_data_pos, filename_data_neg, filename_data_grid, filename_scores_grid, th, noice_needed=1, caption=self.classifier.name + ' %1.2f' % auc)
            plt.tight_layout()
            plt.show()

        return tpr, fpr, auc
# ---------------------------------------------------------------------------------------------------------------------
