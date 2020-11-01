import numpy
import os
from os import listdir
import fnmatch
from sklearn import metrics, datasets
from sklearn.metrics import confusion_matrix, auc
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from xgboost import XGBClassifier
from matplotlib.colors import LinearSegmentedColormap
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
def plot_tp_fp(plt,fig,tpr,fpr,roc_auc,caption='',filename_out=None):

    #ax = fig.gca()
    #ax.set_xticks(numpy.arange(0, 1.1, 0.1))
    #ax.set_yticks(numpy.arange(0, 1.1, 0.1))

    lw = 2
    plt.plot(fpr, tpr, color='darkgreen', lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1.05], [0, 1.05], color='lightgray', lw=lw, linestyle='--')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.set_title(caption + ('AUC = %0.4f' % roc_auc))
    plt.legend(loc="lower right")
    plt.grid(which='major', color='lightgray', linestyle='--')
    #fig.canvas.set_window_title(caption + ('AUC = %0.4f' % roc_auc))
    #plt.set_title(caption)

    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_multiple_tp_fp(tpr,fpr,roc_auc,desc,caption=''):

    fig = plt.figure()
    fig.canvas.set_window_title(caption)
    ax = fig.gca()
    ax.set_xticks(numpy.arange(0, 1.1, 0.1))
    ax.set_yticks(numpy.arange(0, 1.1, 0.1))

    lw = 2
    plt.plot([0, 1.1], [0, 1.1], color='lightgray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()

    colors_list = list(('lightgray','red','blue','purple','green','orange','cyan','darkblue','darkred','darkgreen','darkcyan'))
    lbl = []
    lbl.append('')
    L = len(colors_list)

    for i in range(0,len(roc_auc)):
        plt.plot(fpr[i], tpr[i],  lw=lw, color=colors_list[(i+1)%L],alpha = 0.5)
        lbl.append('%0.2f %s' % (roc_auc[i],desc[i]))

    plt.legend(lbl, loc=4)

    leg = plt.gca().get_legend()
    for i in range(0, len(roc_auc)):
        leg.legendHandles[i].set_color(colors_list[i%L])

    leg.legendHandles[0].set_visible(False)
    return
# ----------------------------------------------------------------------------------------------------------------------
def smape(A, F):
    v_norm = (numpy.abs(A) + numpy.abs(F))
    v_diff = numpy.abs(F - A)
    idx = numpy.where(v_norm!=0)
    res = 2*100/A.shape[0]*numpy.sum(v_diff[idx]/v_norm[idx])
    return res
# ----------------------------------------------------------------------------------------------------------------------
def calc_series_SMAPE(S):

    if len(S.shape)==1:
        return [0]
    SMAPE = numpy.zeros(S.shape[1])
    for i in range(1,S.shape[1]):
        SMAPE[i] = smape(S[:,0], S[:,i])

    return SMAPE
# ----------------------------------------------------------------------------------------------------------------------
def plot_hystogram(plt,H,label=None,SMAPE=None,xmin=None,xmax=None,ymax=None,filename_out=None):

    plt.hist(H, bins='auto')
    #plt.axis((xmin, xmax, 0, ymax))
    if (SMAPE is not None) and (label is not None):
        plt.legend([label[0] + ' SMAPE = %1.2f%%'%SMAPE[0]])
    plt.grid()
    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_series(S,labels=None,SMAPE=None,filename_out = None):

    colors_list = tools_draw_numpy.get_colors(S.shape[1], colormap = 'viridis')/255
    colors_list = colors_list[:,[2,1,0]]

    plt.clf()

    x = numpy.arange(0,S.shape[0],1)

    #for i in range(0,S.shape[1]):plt.plot(x,S[:,i], lw=1, color=colors_list[i], alpha=0.75)
    for i in range(1, S.shape[1]):
        plt.fill_between(x,S[:,i],S[:,i-1],color = colors_list[i])

    plt.fill_between(x, 0, S[:, 0], color=colors_list[0])

    plt.tight_layout()
    plt.grid()

    if labels is not None:
        patches = [mpatches.Patch(color=colors_list[i], label=labels[i]) for i in reversed(range(0, S.shape[1]))]
        plt.legend(handles=patches,loc='upper left')


    if filename_out is not None:
        plt.savefig(filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_two_series(filename1, filename2, caption='', filename_out=None):

    mat1 = tools_IO.load_mat(filename1, dtype=numpy.float32, delim='\t')
    mat2 = tools_IO.load_mat(filename2, dtype=numpy.float32, delim='\t')

    fig = plt.figure(figsize=(12, 6))

    if len(mat1.shape)==1:
        SMAPE = calc_series_SMAPE(numpy.array([mat1, mat2]).T)
        labels = [filename1.split('/')[-1], filename2.split('/')[-1]]
        plot_series(numpy.vstack((mat1,mat2)).T, labels=labels,SMAPE=SMAPE)

    else:
        labels = [filename1.split('/')[-1], filename2.split('/')[-1]]
        S = numpy.minimum(mat1.shape[1],5)
        for i in range(0,S):
            SMAPE = calc_series_SMAPE(numpy.array([mat1[:,i], mat2[:,i]]).T)
            plot_series(numpy.vstack((mat1[:,i], mat2[:,i])).T, labels=labels, SMAPE=SMAPE)
        plt.tight_layout()

    if filename_out is not None:
        plt.savefig(filename_out)

    return
# ---------------------------------------------------------------------------------------------------------------------
def plot_multiple_series(filename_fact, list_filenames, target_column=0,caption='',filename_out=None):

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace =0.01)
    fig.canvas.set_window_title(caption)

    S = 1+len(list_filenames)
    Labels_train = numpy.array(['fact']+[each.split('/')[-1] for each in list_filenames])


    Series = []
    mat = tools_IO.load_mat(filename_fact, dtype=numpy.float32, delim='\t')
    if len(mat.shape) == 1:
        Series.append(mat)
    else:
        Series.append(mat[:, target_column])

    for filename in list_filenames:
        mat = tools_IO.load_mat(filename,dtype=numpy.float32,delim='\t')
        if len(mat.shape)==1:
            Series.append(mat)
        else:
            Series.append(mat[:, target_column])

    Series=numpy.array(Series).T

    SMAPE = calc_series_SMAPE(Series)

    for i in range(1, S):
        plot_series(Series[:,[0,i]],labels=Labels_train[[0,i]],SMAPE=SMAPE[[0,i]])

    plt.tight_layout()
    if filename_out is not None:
        plt.savefig(filename_out)

    return
# ---------------------------------------------------------------------------------------------------------------------
def display_roc_curve_from_file(plt,fig,path_scores,caption=''):

    data = tools_IO.load_mat(path_scores, dtype = numpy.chararray, delim='\t')
    labels = (data [:, 0]).astype('float32')
    scores = data[:, 1:].astype('float32')

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plot_tp_fp(plt,fig,tpr,fpr,roc_auc,caption)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_2D_samples_from_folder(foldername,caption='',add_noice=0):

    local_filenames = fnmatch.filter(listdir(foldername), '*.*')
    fig = plt.figure()
    fig.canvas.set_window_title(caption)
    colors_list = list(('red', 'blue', 'green', 'orange', 'cyan', 'purple', 'black', 'gray', 'pink', 'darkblue'))

    i=0
    for filename in local_filenames:
        data = tools_IO.load_mat(foldername+filename,dtype=numpy.chararray, delim='\t')
        labels = (data[:, 0]).astype('float32')
        X = data[:, 1:].astype('float32')

        if add_noice>0:
            noice = 0.05-0.1*numpy.random.random_sample(X.shape)
            X+= noice

        plt.plot(X[:, 0], X[:, 1], 'ro', color=colors_list[i%len(colors_list)], alpha=0.4)
        i+=1

    plt.grid()

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_2D_features_multi_Y(plt, X, Y, labels=None, filename_out=None):

    colors_list = list(('red', 'blue', 'green', 'orange', 'cyan', 'purple','black','gray','pink','darkblue'))
    patches = []

    for i,each in enumerate(numpy.unique(Y)):
        idx = numpy.where(Y==each)
        plt.plot(X[idx, 0], X[idx, 1], 'ro', color=colors_list[i%len(colors_list)], alpha=0.4,markeredgewidth=0)
        if labels is not None:
            patches.append(mpatches.Patch(color=colors_list[i%len(colors_list)],label=labels[i]))

    plt.grid()

    if labels is not None:
        plt.legend(handles=patches)

    plt.tight_layout()
    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_2D_features_pos_neg(X, Y, filename_out=None):

    dict_pos,dict_neg ={},{}
    for x in X[Y>0]:
        if tuple(x) not in dict_pos:
            dict_pos[tuple(x)]=1
        else:
            dict_pos[tuple(x)]+=1

    for x in X[Y<=0]:
        if tuple(x) not in dict_neg:
            dict_neg[tuple(x)]=1
        else:
            dict_neg[tuple(x)]+=1

    col_neg = (0, 0.5, 1)
    col_pos = (1, 0.5, 0)
    col_gray = (0.5, 0.5, 0.5)

    min_size = 4
    max_size = 20
    norm = max(tools_IO.max_element_by_value(dict_pos)[1],tools_IO.max_element_by_value(dict_neg)[1])/max_size

    for x in dict_pos:
        if tuple(x) not in dict_neg:
            sz = dict_pos[tuple(x)] / norm
            plt.plot(x[0], x[1], 'ro', color=col_pos, markeredgewidth=0,markersize=max(min_size,sz))
        else:
            sz = (dict_pos[tuple(x)]+dict_neg[tuple(x)]) / norm
            plt.plot(x[0], x[1], 'ro', color=col_gray, markeredgewidth=0, markersize=max(min_size, sz))

            if dict_pos[tuple(x)]<dict_neg[tuple(x)]:
                sz = (dict_neg[tuple(x)]-dict_pos[tuple(x)]) / norm
                plt.plot(x[0], x[1], 'ro', color=col_neg, markeredgewidth=0, markersize=max(min_size, sz))
            else:
                sz = (-dict_neg[tuple(x)]+dict_pos[tuple(x)]) / norm
                plt.plot(x[0], x[1], 'ro', color=col_pos, markeredgewidth=0, markersize=max(min_size, sz))

    for x in dict_neg:
        if tuple(x) not in dict_pos:
            sz = dict_neg[tuple(x)] / norm
            plt.plot(x[0], x[1], 'ro', color=col_neg, markeredgewidth=0,markersize=max(min_size,sz))

    plt.grid()

    plt.tight_layout()
    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_1D_features_pos_neg(plt, X, Y, labels=None, filename_out=None):

    patches = []

    plt.xlim([-1, X.max()+1])
    bins = numpy.arange(-0.5,X.max()+0.5,0.25)

    for i,y in enumerate(numpy.unique(Y)):
        if int(y)<=0:
            col = (0, 0.5, 1)
        else:
            col = (1, 0.5, 0)
        plt.hist(X[Y==y], bins=bins,color=col,alpha=0.5,width=0.25,align='mid')

        if labels is not None:
            patches.append(mpatches.Patch(color=col,label=y))

    plt.grid()

    if labels is not None:
        plt.legend(handles=patches)

    plt.tight_layout()
    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_2D_scores(plt,fig,filename_data_pos,filename_data_neg,filename_data_grid,filename_scores_grid,th,noice_needed=0,caption='',filename_out=None):
    if not os.path.isfile(filename_data_pos): return
    if not os.path.isfile(filename_data_neg): return
    if not os.path.isfile(filename_data_grid): return

    data = tools_IO.load_mat(filename_scores_grid, dtype=numpy.chararray, delim='\t')
    data = data[1:,:]
    grid_scores = data[:, 1:].astype('float32')

    data = tools_IO.load_mat(filename_data_grid, dtype=numpy.chararray, delim='\t')
    data_grid = data[:,1:].astype('float32')

    data = tools_IO.load_mat(filename_data_pos, dtype=numpy.chararray, delim='\t')
    l1 = (data[:, 0]).astype('float32')
    x1 = data[:,1:].astype('float32')

    data = tools_IO.load_mat(filename_data_neg, dtype=numpy.chararray, delim='\t')
    l2 = (data[:, 0]).astype('float32')
    x2 = data[:,1:].astype('float32')

    X = numpy.vstack((x1,x2))
    labels = numpy.hstack((l1, l2)).astype(int)

    X1 = X[labels >  0]
    X0 = X[labels <= 0]

    #'''
    max = numpy.max(grid_scores)
    min = numpy.min(grid_scores)
    for i in range(0,grid_scores.shape[0]):
        if(grid_scores[i]>th):
            grid_scores[i]=(grid_scores[i]-th)/(max-th)
        else:
            grid_scores[i] = (grid_scores[i] - th) / (th-min)
    #'''

    S=int(math.sqrt(grid_scores.shape[0]))
    grid_scores=numpy.reshape(grid_scores,(S,S))

    minx=numpy.min(data_grid[:, 0])
    maxx=numpy.max(data_grid[:, 0])
    miny=numpy.min(data_grid[:, 1])
    maxy=numpy.max(data_grid[:, 1])


    if noice_needed>0:
        noice1 = 0.05-0.2*numpy.random.random_sample(X1.shape)
        noice0 = 0.05-0.2*numpy.random.random_sample(X0.shape)
        X1+=noice1
        X0+=noice0

    plt.set_title(caption)

    xx, yy = numpy.meshgrid(numpy.linspace(minx, maxx, num=S), numpy.linspace(miny, maxy,num=S))

    plt.contourf(xx, yy, numpy.flip(grid_scores,0), cmap=cm.coolwarm, alpha=.8)
    #plt.imshow(grid_scores, interpolation='bicubic',cmap=cm.coolwarm,extent=[minx,maxx,miny,maxy],aspect='auto')

    plt.plot(X0[:, 0], X0[:, 1], 'ro', color='blue', alpha=0.4)
    plt.plot(X1[: ,0], X1[:, 1], 'ro' ,color='red' , alpha=0.4)
    plt.grid()
    plt.set_xticks(())
    plt.set_yticks(())
    #fig.subplots_adjust(hspace=0.001,wspace =0.001)
    if filename_out is not None:
        plt.savefig(filename_out)

    return

# ----------------------------------------------------------------------------------------------------------------------
def display_roc_curve_from_descriptions(plt,figure,filename_scores_pos, filename_scores_neg,delim=' ',caption='',inverse_score=0,filename_out=None):

    scores_pos = []
    scores_neg = []
    files_pos = []
    files_neg = []

    with open(filename_scores_pos) as f:
        lines = f.read().splitlines()
    for each in lines:
        filename = each.split(delim)[0]
        value = each.split(delim)[1]
        if ((value[0]=='+')or(value[0]=='-')):
            if ((value[1]>='0') and (value[1]<='9')):
                scores_pos.append(value)
                files_pos.append(filename)
        else:
            if ((value[0] >= '0') and (value[0] <= '9')):
                scores_pos.append(value)
                files_pos.append(filename)


    with open(filename_scores_neg) as f:
        lines = f.read().splitlines()
    for each in lines:
        filename = each.split(delim)[0]
        value = each.split(delim)[1]
        if ((value[0]=='+')or(value[0]=='-')):
            if ((value[1]>='0') and (value[1]<='9')):
                scores_neg.append(value)
                files_neg.append(filename)
        else:
            if ((value[0] >= '0') and (value[0] <= '9')):
                scores_neg.append(value)
                files_neg.append(filename)

    scores_pos = numpy.array(scores_pos)
    scores_neg = numpy.array(scores_neg)

    for i in range(0,scores_pos.shape[0]):
        scores_pos[i]=scores_pos[i].split('x')[0]

    for i in range(0,scores_neg.shape[0]):
        scores_neg[i]=scores_neg[i].split('x')[0]

    scores_pos = scores_pos.astype(numpy.float32)
    scores_neg = scores_neg.astype(numpy.float32)

    if(inverse_score==1):
        scores_neg[:] = 1.0 -scores_neg[:]

    labels = numpy.hstack((numpy.full(len(scores_pos), 1), numpy.full(len(scores_neg), 0)))
    scores = numpy.hstack((scores_pos, scores_neg))




    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)



    if(roc_auc>0.5):
        plot_tp_fp(plt,figure,tpr, fpr,roc_auc,caption,filename_out)
    else:
        plot_tp_fp(plt, figure, fpr, tpr, 1-roc_auc,caption,filename_out)


# ----------------------------------------------------------------------------------------------------------------------
def display_distributions(plt,fig,path_scores1, path_scores2,delim=' ',inverse_score=0,filename_out=None):
    scores1 = []
    scores2 = []


    with open(path_scores1) as f:
        lines = f.read().splitlines()
    for each in lines:
        value = each.split(delim)[1]
        if ((value[0] == '+') or (value[0] == '-')):
            if ((value[1] >= '0') and (value[1] <= '9')):
                scores1.append(value)
        else:
            if ((value[0] >= '0') and (value[0] <= '9')):
                scores1.append(value)


    with open(path_scores2) as f:
        lines = f.read().splitlines()
    for each in lines:
        value = each.split(delim)[1]
        if ((value[0] == '+') or (value[0] == '-')):
            if ((value[1] >= '0') and (value[1] <= '9')):
                scores2.append(value)
        else:
            if ((value[0] >= '0') and (value[0] <= '9')):
                scores2.append(value)

    scores1 = numpy.array(scores1)
    scores2 = numpy.array(scores2)

    for i in range(0,scores1.shape[0]):
        scores1[i] = scores1[i].split('x')[0]

    for i in range(0,scores2.shape[0]):
        scores2[i] = scores2[i].split('x')[0]


    if(numpy.max(numpy.array(scores1).astype(float))<=1):
        scores1=100*numpy.array(scores1).astype(float)

    if (numpy.max(numpy.array(scores2).astype(float)) <= 1):
        scores2 = 100 * numpy.array(scores2).astype(float)

    if (inverse_score == 1):
        scores2 = 100-scores2

    scores1= scores1.astype(numpy.float32)
    scores2= scores2.astype(numpy.float32)

    m1 = numpy.min(scores1)
    m2 = numpy.min(scores2)
    min = numpy.minimum(m1,m2)
    scores1+= -min
    scores2+= -min


    freq1=numpy.bincount(numpy.array(scores1).astype(int))/len(scores1)
    freq2=numpy.bincount(numpy.array(scores2).astype(int))/len(scores2)


    x_max1=numpy.max(numpy.array(scores1).astype(float))
    x_max2=numpy.max(numpy.array(scores2).astype(float))
    y_max1=numpy.max(numpy.array(freq1).astype(float))
    y_max2=numpy.max(numpy.array(freq2).astype(float))

    x_max=max(x_max1,x_max1)*1.1
    y_max=max(y_max1,y_max2)*1.1

    #major_ticks = numpy.arange(0, x_max, 10)
    #minor_ticks = numpy.arange(0, x_max, 1)
    #plt.xlim([0.0, x_max])
    #plt.ylim([0.0, y_max])

    plt.grid(which='major',axis='both', color='lightgray',linestyle='--')
    #plt.minorticks_on()
    #plt.grid(which='minor', color='r')
    #plt.grid(which='both')


    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='on', labelleft='off', labeltop='off',labelright='off', labelbottom='off')

    plt.plot(freq1, color='red', lw=2)
    plt.plot(freq2, color='gray', lw=2)

    if filename_out is not None:
        plt.savefig(filename_out)

    return
#----------------------------------------------------------------------------------------------------------------------
def plot_learning_rates1(plt,fig,filename_mat):
    if not os.path.isfile(filename_mat):
        return

    mat = tools_IO.load_mat(filename_mat, dtype=numpy.chararray, delim='\t')
    dsc1 = mat[0,0].decode("utf-8")

    mat = mat[1:,:].astype(numpy.float32)
    x = numpy.arange(0,mat.shape[0])

    plt.plot(x, mat[:,0])
    #plt.plot(x, mat[:,1])
    plt.grid(which='major', color='lightgray', linestyle='--')
    ax = fig.gca()
    ax.set_title(dsc1)


    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_learning_rates2(plt,fig,filename_mat):
    if not os.path.isfile(filename_mat):
        return

    mat = tools_IO.load_mat(filename_mat, dtype=numpy.chararray, delim='\t')
    dsc = mat[0,2].decode("utf-8")

    mat = mat[1:,:].astype(numpy.float32)
    x = numpy.arange(0,mat.shape[0])

    plt.plot(x, mat[:,2])
    plt.plot(x, mat[:,3])
    plt.grid(which='major', color='lightgray', linestyle='--')
    ax = fig.gca()
    ax.set_title(dsc)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_features_PCA(plt,features,Y,patterns,filename_out=None):

    X_TSNE = TSNE(n_components=2).fit_transform(features)
    plot_2D_features_multi_Y(plt, X_TSNE, Y, labels=patterns)

    if filename_out is not None:
        plt.savefig(filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_feature_importance(plt,fig,X,Y,header,filename_out=None):

    model = XGBClassifier()
    model.fit(X, Y)

    keys, values = [],[]

    feature_importances = model.get_booster().get_score()
    for k, v in feature_importances.items():
        keys.append(k)
        values.append(v)



    values = numpy.array(values)
    idx = numpy.argsort(-values)
    keys = numpy.array(keys)[idx]
    values = values[idx]
    header = header[idx]

    N=5
    ax = fig.gca()
    ax.pie(values[:N],  labels=header[:N], autopct='%1.1f%%',shadow=False, startangle=90)
    #plt.set_title('Feature importance')
    if filename_out is not None:
        plt.savefig(filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_corelation(plt,fig,X,Y,header):

    N = X.shape[1]
    mat = numpy.zeros((N,N))

    for i in range(N):
        for j in range(N):
            mat[i,j] = numpy.correlate(X[:,i],X[:,j])

    plt.imshow(mat,cmap='jet')

    ax = fig.gca()
    ax.set_xticks(numpy.arange(mat.shape[1]))
    ax.set_yticks(numpy.arange(mat.shape[0]))

    ax.set_yticklabels(header)

    return
#----------------------------------------------------------------------------------------------------------------------
def plot_confusion_mat(plt,fig,filename_mat,caption=''):

    confusion_mat = tools_IO.load_mat(filename_mat,dtype=numpy.chararray,delim='\t')[:,:2]
    patterns = numpy.unique(confusion_mat[:, 0])

    labels_fact = numpy.array([tools_IO.smart_index(patterns, each) for each in confusion_mat[:, 0]])
    labels_pred = numpy.array([tools_IO.smart_index(patterns, each) for each in confusion_mat[:, 1]])


    mat, descriptions,sorted_labels = tools_IO.preditions_to_mat(labels_fact, labels_pred, numpy.unique(confusion_mat[:,0]))
    ind = numpy.array([('%3d' % i) for i in range(0, mat.shape[0])])
    TP = float(numpy.trace(mat))

    plt.imshow(mat,cmap='jet')

    ax = fig.gca()
    ax.set_xticks(numpy.arange(mat.shape[1]))
    ax.set_yticks(numpy.arange(mat.shape[0]))


    ax.set_yticklabels(sorted_labels.astype(numpy.str))


    for i in range(mat.shape[1]):
        for j in range(mat.shape[0]):
            ax.text(j, i, '%1.0f'% mat[i,j],ha="center", va="center", color='white')

    TP = numpy.trace(mat)
    ax.set_title(caption + " %1.2f" % (float(TP/numpy.sum(mat))))

    return
#----------------------------------------------------------------------------------------------------------------------
def plot_histo(dict_H, filename_out=None, colors=None):
    minw = tools_IO.min_element_by_key(dict_H)[0]
    maxw = tools_IO.max_element_by_key(dict_H)[0]

    norm = sum(dict_H.values())

    xticks = numpy.array([x for x in dict_H.keys()],dtype=numpy.int32)
    xticks = numpy.sort(xticks)
    Y = numpy.array([100 * dict_H[x] / norm for x in xticks])

    #fig = plt.figure(figsize=(6, 12))
    barlist = plt.bar(xticks, Y,width=5)
    if colors is not None and len(colors)==len(dict_H):
        for i in range(len(colors)):
            barlist[i].set_color((colors[i,2]/255,colors[i,1]/255,colors[i,0]/255))
    plt.xticks(xticks)
    plt.xlim(left=minw-1, right=maxw+1)
    plt.ylim(bottom=0, top=40)
    plt.grid()
    plt.tight_layout()
    if filename_out is not None:
        plt.savefig(filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_gradient_hack(p0, p1, npts=20, cmap=None, **kw):

    x_1, y_1 = p0
    x_2, y_2 = p1

    X = numpy.linspace(x_1, x_2, npts)
    Xs = X[:-1]
    Xf = X[1:]
    Xpairs = zip(Xs, Xf)

    Y = numpy.linspace(y_1, y_2, npts)
    Ys = Y[:-1]
    Yf = Y[1:]
    Ypairs = zip(Ys, Yf)

    C = numpy.linspace(0, 1, npts)
    cmap = plt.get_cmap(cmap)
    # the simplest way of doing this is to just do the following:
    for x, y, c in zip(Xpairs, Ypairs, C):
        plt.plot(x, y, '-', c=cmap(c), **kw)


    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_gradient_rbg_pairs(p0, p1, rgb0, rgb1, **kw):
    cmap = LinearSegmentedColormap.from_list('tmp', (rgb0, rgb1))
    plot_gradient_hack(p0, p1, cmap=cmap, **kw)
    return
# ----------------------------------------------------------------------------------------------------------------------
def fig2data(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
    buf.shape = (w, h, 4)
    buf = numpy.roll(buf, 3, axis=2)
    return buf
# ----------------------------------------------------------------------------------------------------------------------
def pairplot(X,Y,folder_out):

    for i in range(X.shape[1]):
        plt.clf()
        plot_1D_features_pos_neg(plt, X[:, i], Y, labels=Y, filename_out=folder_out + 'plt_%02d_%02d.png' % (i, i))


    for i in range(X.shape[1]-1):
        for j in range(i+1,X.shape[1]):
            plt.clf()
            #plot_2D_features_pos_neg(X[:, [i, j]], Y, filename_out=folder_out + 'plt_%02d_%02d.png' % (i, j))


    return
# ----------------------------------------------------------------------------------------------------------------------