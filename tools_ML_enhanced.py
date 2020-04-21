import cv2
import numpy
import matplotlib.pyplot as plt
import progressbar
from sklearn import metrics
# ----------------------------------------------------------------------------------------------------------------------
import classifier_Gauss
import classifier_Gauss_indep
import classifier_RF
import classifier_XGBoost
import classifier_XGBoost2
import classifier_DTree
import classifier_Bayes
import classifier_KNN
import classifier_SVM
import classifier_LM
import classifier_Hash
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_plot
# ----------------------------------------------------------------------------------------------------------------------
class tools_ML_enhanced(object):
    def __init__(self,Classifier):
        self.classifier = Classifier
        self.folder_out = './data/output/'

        self.filename_scores_train = self.folder_out+'scores_train.txt'
        self.filename_roc_train = self.folder_out+'roc_train.png'
        self.filename_scores_test= self.folder_out + 'scores_test.txt'
        self.filename_roc_test= self.folder_out + 'roc_test.png'
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
        return numpy.array(accuracy).max()
# ---------------------------------------------------------------------------------------------------------------------
    def learn_file(self, filename_train, has_header=True, verbose=False):
        N = tools_IO.count_lines(filename_train)
        data = numpy.array(tools_IO.get_lines(filename_train, start=has_header * 1, end=has_header * 1 + N))
        self.classifier.learn(data[:, 1:], data[:, 0])
        #self.classifier.learn_file(filename_train,has_header=has_header)
        tools_IO.write_cache(self.folder_out+self.classifier.name+'.dat',self.classifier)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def inference(self,filename_in,has_header=False,has_labels_first_col=False):

        self.classifier,success = tools_IO.load_if_exists(self.folder_out + self.classifier.name + '.dat')
        if success:
            self.predict_file(filename_in, self.filename_scores_test, has_header=has_header,has_labels_first_col=has_labels_first_col)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def E2E_train_test(self,filename_train,filename_test,has_header,has_labels_first_col):

        self.learn_file(filename_train, has_header=has_header,verbose=True)
        self.predict_file(filename_train,self.filename_scores_train,has_header=has_header,has_labels_first_col=has_labels_first_col)
        self.predict_file(filename_test, self.filename_scores_test ,has_header=has_header,has_labels_first_col=has_labels_first_col)
        Y_train = numpy.array(tools_IO.get_columns(filename_train,start=0,end=1),dtype=numpy.int)
        Y_test  = numpy.array(tools_IO.get_columns(filename_test ,start=0,end=1),dtype=numpy.int)

        fig = plt.figure()
        tpr, fpr, auc = tools_IO.get_roc_data_from_scores_file(self.filename_scores_train, has_header=has_header)
        accuracy_train = self.get_accuracy(tpr,fpr,nPos=numpy.count_nonzero(Y_train> 0),nNeg=numpy.count_nonzero(Y_train<=0))
        tools_plot.plot_tp_fp(plt, fig, tpr, fpr, auc, caption='Train', filename_out=self.filename_roc_train)

        plt.clf()
        tpr, fpr, auc = tools_IO.get_roc_data_from_scores_file(self.filename_scores_test, has_header=has_header)
        accuracy_test = self.get_accuracy(tpr, fpr, nPos=numpy.count_nonzero(Y_test > 0),nNeg=numpy.count_nonzero(Y_test <= 0))
        tools_plot.plot_tp_fp(plt, fig, tpr, fpr, auc, caption='Test', filename_out=self.filename_roc_test)

        M = tools_IO.load_mat(self.filename_scores_train,dtype=numpy.float,delim='\t')
        loss_train = ((M[:, 0] - M[:, 1] / 100) ** 2).mean()
        M = tools_IO.load_mat(self.filename_scores_test, dtype=numpy.float, delim='\t')
        loss_test = ((M[:, 0] - M[:, 1] / 100) ** 2).mean()


        print('\naccuracy_train,accuracy_test,loss_train,loss_test')
        print('%d\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t'%(len(Y_train)+len(Y_test),accuracy_train,accuracy_test,loss_train,loss_test))

        return
# ---------------------------------------------------------------------------------------------------------------------
    def split_train_test(self,filename_in,filename_train,filename_test,ratio=0.75,has_header=False,random_split=True,max_line=None):

        g1 = open(filename_train,'w')
        g2 = open(filename_test, 'w')

        N = tools_IO.count_lines(filename_in)
        if max_line is not None:
            N = min(N,max_line)

        with open(filename_in, 'r') as f:
            for i,line in enumerate(f):
                if has_header and i==0:
                    g1.write("%s\n" % line)
                    g2.write("%s\n" % line)
                    continue

                if random_split:
                    if numpy.random.rand(1) <ratio:
                        g1.write("%s" % line)
                    else:
                        g2.write("%s" % line)
                else:
                    if float(i/N) < ratio:
                        g1.write("%s" % line)
                    else:
                        g2.write("%s" % line)

                if (max_line is not None) and (i>=max_line-1):
                    break

        g1.close()
        g2.close()

        return
# ---------------------------------------------------------------------------------------------------------------------
    def predict_file(self, filename_in,filename_out,delim='\t',has_header=True,has_labels_first_col=True):

        scores = []
        if has_header:
            scores.append(-1)

        N = tools_IO.count_lines(filename_in)

        bar = progressbar.ProgressBar(max_value=N)
        with open(filename_in, 'r') as f:
            for i,line in enumerate(f):
                bar.update(i)
                if has_header and i==0:continue
                line = line.strip()
                if len(line) > 0 :
                    X = numpy.array(line.split(delim))
                    if has_labels_first_col:
                        X=X[1:]

                    score = self.classifier.predict(X)
                    score = (100 * score[:, 1]).astype(int)
                    scores.append(score)


        if not has_labels_first_col:
            tools_IO.save_mat(scores, filename_out,fmt='%s',delim='\t')
        else:
            Y_train = tools_IO.get_columns(filename_in, start=0, end=1)
            M = numpy.hstack((Y_train, numpy.array(scores)))
            tools_IO.save_mat(M, filename_out, fmt='%s', delim='\t')


        return
# ---------------------------------------------------------------------------------------------------------------------
    def reduce_features(self,filename_in,filename_out,list_ID,has_header=False,has_labels_first_col=True,delim='\t'):

        g = open(filename_out, 'w')
        f = open(filename_in, 'r')

        for i,line in enumerate(f):

            if has_header and i==0:
                g.write("%s\n" % line)
                continue

            line = line.strip()
            if len(line) > 0 :
                X = numpy.array(line.split(delim),dtype=numpy.int)
                result = []
                if has_labels_first_col:
                    result.append(X[0])
                    X=X[1:]

                for idx in list_ID:
                    result.append(X[idx])

                for each in result:
                    g.write("%d\t" % each)
                g.write("\n")

        f.close()
        g.close()

        return
# ---------------------------------------------------------------------------------------------------------------------
    def pairplot(self,filename_in,folder_out,has_header=False):

        data = numpy.array(tools_IO.get_lines(filename_in, start=has_header*1),dtype=numpy.float)
        Y = data[:, 0]
        X = data[:, 1:]
        #noice = 0*(numpy.random.random(X.shape)-0.5)/X.max()
        tools_plot.pairplot(X,Y,folder_out)

        filenames = tools_IO.get_filenames(folder_out,'plt_*.png')
        N = int((1+numpy.sqrt(1+8*len(filenames)))//2)-1
        small_size=(240,320)
        large_image = numpy.full((small_size[0]*N,small_size[1]*N,3),255,dtype=numpy.uint8)

        row,col = 0,0
        for i,filename in enumerate(filenames):
            image = cv2.resize(cv2.imread(folder_out + filename),(small_size[1],small_size[0]))
            large_image[row*small_size[0]:row*small_size[0]+small_size[0],col*small_size[1]:col*small_size[1]+small_size[1]]=image
            large_image[col*small_size[0]:col*small_size[0]+small_size[0],row*small_size[1]:row*small_size[1]+small_size[1]]=image
            col+=1
            if col>=N:
                row+=1
                col=row

        cv2.imwrite(folder_out+'large.png',large_image)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def engineer_feature_v1(self, X_train, Y_train,X_val, Y_val):

        best_feature_train, best_feature_val,best_acc, best_classifier = None,None,None,None

        C1 = classifier_DTree.classifier_DT()

        #C1 = classifier_Gauss.classifier_Gauss()
        #C2 = classifier_LM.classifier_LM()
        #C3 = classifier_SVM.classifier_SVM()
        #C4 = classifier_Bayes.classifier_Bayes()
        #C6= classifier_RF.classifier_RF()
        #C7= classifier_XGBoost.classifier_XGBoost()
        #C8= classifier_XGBoost2.classifier_XGBoost2()
        #C9= classifier_DTree.classifier_DT()
        #C10= classifier_KNN.classifier_KNN()

        classifiers = [C1]


        for classifier in classifiers:
            classifier.learn(X_train, Y_train)
            features_train = 100*classifier.predict(X_train)[:, 1]
            features_val   = 100*classifier.predict(X_val)[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(Y_val, features_val)
            acc = self.get_accuracy(tpr, fpr, nPos=numpy.count_nonzero(Y_val> 0), nNeg=numpy.count_nonzero(Y_val<= 0))
            #print('%1.3f %s' % (acc, classifier.name))

            if (best_acc is None) or (acc>best_acc):
                best_acc  = acc
                best_feature_train = features_train
                best_feature_val = features_val
                best_classifier = classifier.name

        return best_feature_train, best_feature_val,best_acc, best_classifier
# ---------------------------------------------------------------------------------------------------------------------
    def engineer_feature_v2(self, X_train, Y_train, X_val, Y_val):

        best_feature_train, best_feature_val, best_acc, best_classifier = None, None, 0, None

        classifier = classifier_DTree.classifier_DT()

        classifier.learn(X_train, Y_train)
        features_train = 100 * classifier.predict(X_train)[:, 1]
        features_val = 100 * classifier.predict(X_val)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(Y_val, features_val)
        acc = self.get_accuracy(tpr, fpr, nPos=numpy.count_nonzero(Y_val > 0),nNeg=numpy.count_nonzero(Y_val <= 0))


        #classifier = classifier_Bayes.classifier_Bayes()
        classifier = classifier_Hash.classifier_Hash()

        X_train_sum = numpy.vstack((numpy.sum(X_train, axis=1),(numpy.zeros(X_train.shape[0])))).T
        X_val_sum   = numpy.vstack((numpy.sum(X_val  , axis=1),(numpy.zeros(X_val.shape[0])))).T
        classifier.learn(X_train_sum, Y_train)
        features_train_sum = 100 * classifier.predict(X_train_sum)[:, 1]
        features_val_sum = 100 * classifier.predict(X_val_sum)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(Y_val, features_val_sum)
        acc_sum= self.get_accuracy(tpr, fpr, nPos=numpy.count_nonzero(Y_val > 0),nNeg=numpy.count_nonzero(Y_val <= 0))

        best_acc = acc_sum
        best_feature_train = features_train_sum
        best_feature_val = features_val_sum

        if acc_sum>acc:
            best_classifier = 'eng'
        else:
            best_classifier = '   '
            best_acc = 0

        return best_feature_train, best_feature_val, float(acc_sum/acc), acc, best_classifier
# ---------------------------------------------------------------------------------------------------------------------
    def extend_file_with_features(self, filename_in, filename_out, features, has_header=False,has_labels_first_col=True, delim='\t'):

        g = open(filename_out, 'w')
        f = open(filename_in, 'r')

        for i, line in enumerate(f):

            if has_header and i == 0:
                g.write("%s\n" % line)
                continue

            line = line.strip()
            if len(line) > 0:
                X = numpy.array(line.split(delim), dtype=numpy.int)
                X = numpy.hstack((X, features[i]))

                for each in X:
                    g.write("%d\t" % each)
                g.write("\n")

        f.close()
        g.close()

        return
# ---------------------------------------------------------------------------------------------------------------------
    def enrich(self,filename_train,filename_val):

        data = numpy.array(tools_IO.get_lines(filename_train), dtype=numpy.float)
        X_train = data[:, 1:].copy()
        Y_train = data[:, 0].copy()

        data = numpy.array(tools_IO.get_lines(filename_val), dtype=numpy.float)
        X_val= data[:, 1:].copy()
        Y_val= data[:, 0].copy()

        features_train, features_val, accs, names = [],[],[],[]

        for i in range(X_train.shape[1] - 2):
            for j in range(i + 1, X_train.shape[1]-1):
                for k in range(j + 1, X_train.shape[1]):

                    feature_train,feature_val, acc,acc_temp,name = self.engineer_feature_v2(X_train[:,[i, j, k]],Y_train,X_val[:,[i, j,k ]],Y_val)

                    features_train.append(feature_train)
                    features_val.append(feature_val)
                    accs.append(acc)
                    names.append(name)

                    print('%02d %02d %02d %1.3f %1.3f %s'%(i,j,k,acc,acc_temp,name))

        accs = numpy.array(accs)
        idx = numpy.argsort(-accs)
        features_train = numpy.array(features_train).T
        features_val = numpy.array(features_val).T

        N = 10

        aa = accs[idx[:N]]
        xx = features_train[:,idx[:N]]

        self.extend_file_with_features(filename_train, self.folder_out+'train_enriched.txt', features_train[:,idx[:N]])
        self.extend_file_with_features(filename_val  , self.folder_out+'val_enriched.txt'  , features_val[:,idx[:N]])

        return
# ---------------------------------------------------------------------------------------------------------------------
