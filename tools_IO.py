import numpy
import os
from os import listdir
import fnmatch
from shutil import copyfile
import shutil
import random
from sklearn import metrics, datasets
from sklearn.metrics import confusion_matrix, auc
import cv2
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
#from PIL import Image
from xgboost import XGBClassifier
#from xgboost import plot_importance
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
def find_nearest(array, value):
    return array[(numpy.abs(array - value)).argmin()]
# ----------------------------------------------------------------------------------------------------------------------
def smart_index(array, value):
    return numpy.array([i for i, v in enumerate(array) if (v == value)])

# ----------------------------------------------------------------------------------------------------------------------
def remove_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)
# ----------------------------------------------------------------------------------------------------------------------
def remove_files(path,create=False):


    if not os.path.exists(path):
        if create:
            os.mkdir(path)
        return

    filelist = [f for f in os.listdir(path)]
    for f in filelist:
        if os.path.isdir(path + f):
            # shutil.rmtree(path + f)
            continue
        else:
            os.remove(path + f)
    return
# ----------------------------------------------------------------------------------------------------------------------
def remove_folders(path):

    if (path==None):
        return

    if not os.path.exists(path):
        return

    filelist = [f for f in os.listdir(path)]
    for f in filelist:
        if os.path.isdir(path + f):
            shutil.rmtree(path + f)
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_filenames(path_input,list_of_masks):
    local_filenames = []
    for mask in list_of_masks.split(','):
        local_filenames += fnmatch.filter(listdir(path_input), mask)

    return local_filenames
# ----------------------------------------------------------------------------------------------------------------------
def get_next_folder_out(base_folder_out):
    sub_folders = get_sub_folder_from_folder(base_folder_out)
    if len(sub_folders) > 0:
        sub_folders = numpy.array(sub_folders, dtype=numpy.int)
        sub_folders = numpy.sort(sub_folders)
        sub_folder_out = str(sub_folders[-1] + 1)
    else:
        sub_folder_out = '0'
    return base_folder_out + sub_folder_out + '/'
# ----------------------------------------------------------------------------------------------------------------------
def save_flatarrays_as_images(path, cols, rows, array, labels=None, filenames=None, descriptions=None):

    if not os.path.exists(path):
        os.makedirs(path)

    if descriptions is not None:
        f_handle = open(path+"descript.ion", "a+")

    if(array.ndim ==2):
        N=array.shape[0]
    else:
        N=1

    for i in range(0,N):

        if(array.ndim == 2):
            arr = array[i]
            if descriptions is not None:
                description = descriptions[i]


            if filenames is not None:
                short_name = filenames[i]
            else:
                short_name = "%s_%05d.bmp" % (labels[i], i)

        else:
            arr = array[i]
            if descriptions is not None:
                description = descriptions[i]

            if filenames is not None:
                short_name = filenames[i]
            else:
                short_name = "%s_%05d.bmp" % (labels, i)


        #img= toimage(arr.reshape(rows, cols).astype(int)).convert('RGB')
        #img.save(path + short_name)
        arr = arr.reshape(rows, cols)
        arr/=numpy.max(arr)/255.0
        cv2.imwrite(path + short_name,arr)


        if descriptions is not None:
            f_handle.write("%s %s\n" % (short_name, description))

    if descriptions is not None:
        f_handle.close()

    return

# ----------------------------------------------------------------------------------------------------------------------
def save_raw_vec(vec, filename,mode=(os.O_RDWR|os.O_APPEND),fmt='%d',delim=' '):

    if not os.path.isfile(filename):
        mode = os.O_RDWR | os.O_CREAT

    f_handle = os.open(filename,mode)

    s = ""
    for i in range(0, vec.shape[0]-1):
        value = ((fmt+delim) % vec[i]).encode()
        os.write(f_handle,value)

    value = ((fmt+'\n') % vec[vec.shape[0]-1]).encode()
    os.write(f_handle, value)
    os.close(f_handle)

    return
# ----------------------------------------------------------------------------------------------------------------------
def save_mat(mat, filename,fmt='%s',delim='\t'):
    numpy.savetxt(filename, mat,fmt=fmt,delimiter=delim)
    return
# ----------------------------------------------------------------------------------------------------------------------
def save_data_to_feature_file_float(filename,array,target):

    m = numpy.array(array).astype('float32')
    v = numpy.matrix(target).astype('float32')
    mat = numpy.concatenate((v.T,m),axis=1)
    #print(mat)
    numpy.savetxt(filename, mat, fmt='%+2.2f',delimiter='\t')
    return

# ----------------------------------------------------------------------------------------------------------------------
def count_lines(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    f.close()

    return lines
# ----------------------------------------------------------------------------------------------------------------------
def load_mat(filename, dtype=numpy.chararray, delim='\t', lines=None):
    mat  = numpy.genfromtxt(filename, dtype=dtype, delimiter=delim)
    return mat
# ----------------------------------------------------------------------------------------------------------------------
def load_mat_var_size(filename,dtype=numpy.int,delim='\t'):
    l=[]
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                l.append(line.split(delim))
    return l
# ----------------------------------------------------------------------------------------------------------------------
def get_column(list_of_list,col):
    res=[]
    for i in range (0,len(list_of_list)):
        if (col<len(list_of_list[i])):
            res.append(list_of_list[i][col])
        else:
            res.append('-')
    return res
# ----------------------------------------------------------------------------------------------------------------------
def remove_column(list_of_list,col):
    res=[]
    for i in range (0,len(list_of_list)):
        lst = list_of_list[i][0:col] + list_of_list[i][col+1::]
        res.append(lst)
    return res
# ----------------------------------------------------------------------------------------------------------------------
def my_print_sting(strng, space=[]):
    if (strng.ndim != 1):
        return

    fm = "%s"

    for j in range(0, strng.shape[0]):
        s = (fm) % strng[j]
        print(space + s, end=' ')
    print()

    return
# ----------------------------------------------------------------------------------------------------------------------
def my_print_vector(mat):
    if (mat.ndim != 1):
        return

    mx = mat.max()

    fm = "%d"
    if mx >= 10:
        fm = "%2d"
    if mx >= 100:
        fm = "%3d"
    if mx >= 1000:
        fm = "%4d"
    if mx >= 10000:
        fm = "%5d"

    mat.astype(int)

    for j in range(0, mat.shape[0]):
        s = (fm) % mat[j]
        print(s, end=" ")
    print()

    return
# ----------------------------------------------------------------------------------------------------------------------
def my_print_int(mat, rows=None, cols=None,file = None):

    if (mat.ndim == 1):
        return my_print_vector(mat)

    if (rows is not None):
        l = numpy.array([len(each) for each in rows]).max()
        desc_r = numpy.array([" " * (l - len(each)) + each for each in rows]).astype(numpy.chararray)

    mx = mat.max()

    if (cols is not None):
        mx = max(mx,cols.max())

    fm = "%1d"
    if mx >= 10:
        fm = "%2d"
    if mx >= 100:
        fm = "%3d"
    if mx >= 1000:
        fm = "%4d"
    if mx >= 10000:
        fm = "%5d"


    if (cols is not None):
        if (rows is not None):
            print(" " * len(desc_r[0]) + ' |', end="",file=file)

        for each in cols:
            print((fm) % each, end=" ",file=file)

        print(file=file)
        print("-" * (len(desc_r[0])+2), end="",file=file)
        print("-" * cols.shape[0]*(int(fm[1])+1),file=file)



    mat.astype(int)
    for i in range(0, mat.shape[0]):
        if (rows is not None):
            print(desc_r[i] + ' |', end="",file=file)

        for j in range(0, mat.shape[1]):
            print(((fm) % mat[i, j]), end=" ",file=file)
        print(file=file)

    if (cols is not None):
        print("-" * (len(desc_r[0]) + 2), end="", file=file)
        print("-" * cols.shape[0] * (int(fm[1]) + 1), file=file)

    return

# ----------------------------------------------------------------------------------------------------------------------
def get_sub_folder_from_folder(path):
    subfolders = [f.path for f in os.scandir(path) if f.is_dir() ]
    subfolders = [subfolders[i].split(path)[1] for i in range(0,len(subfolders))]

    return subfolders
# ----------------------------------------------------------------------------------------------------------------------
def count_images_from_folder(path, mask="*.bmp"):

    filenames = []
    i=0

    for image_name in fnmatch.filter(listdir(path), mask) :
        try:
            img = cv2.imread(path + image_name)
            filenames.append(image_name)
        except OSError:
            i=i

    return len(filenames)
# ----------------------------------------------------------------------------------------------------------------------
def load_aligned_images_from_folder(path, label, mask="*.bmp", exclusion_folder=None, limit=None, resize_W=None,resize_H=None,grayscaled=False):
    exclusions = []
    if (exclusion_folder != None) and (os.path.exists(exclusion_folder)):

        for each in os.listdir(exclusion_folder):
            exclusions.append(each[:len(each) - 6] + ".bmp")

    images = []
    filenames = []
    i = 0
    img = 0
    for image_name in fnmatch.filter(listdir(path), mask) :
        if ((limit == None) or (i < limit)) and (image_name not in exclusions):
            
            img = cv2.imread(path + image_name) if grayscaled==False else cv2.imread(path + image_name,0)
            if img is None:continue
            if ((resize_W is not None) and (resize_H is not None)):
                img = cv2.resize(img,(resize_W,resize_H))

            images.append(img)
            filenames.append(image_name)
            i += 1

    images = numpy.array(images)
    filenames = numpy.array(filenames)

    labels = numpy.full(images.shape[0], label)

    return images, labels, filenames
# ----------------------------------------------------------------------------------------------------------------------
def preditions_to_mat(labels_fact, labels_pred,patterns):
    mat = confusion_matrix(numpy.array(labels_fact).astype(numpy.int), numpy.array(labels_pred).astype(numpy.int))
    accuracy = [100-100*mat[i,i]/numpy.sum(mat[i,:]) for i in range (0,mat.shape[0])]

    idx = numpy.argsort(accuracy).astype(int)

    descriptions = numpy.array([('%s %3d%%' % (patterns[i], 100-accuracy[i])) for i in range (0,mat.shape[0])])

    a_test = numpy.zeros(labels_fact.shape[0])
    a_pred = numpy.zeros(labels_fact.shape[0])

    for i in range(0,a_test.shape[0]):
        a_test[i] = smart_index(idx ,int(labels_fact[i]))[0]
        a_pred[i] = smart_index(idx, int(labels_pred[i]))[0]

    mat2 = confusion_matrix(numpy.array(a_test).astype(numpy.int), numpy.array(a_pred).astype(numpy.int))
    ind = numpy.array([('%3d' % i) for i in range(0, idx.shape[0])])

    l = numpy.array([len(each) for each in descriptions]).max()
    descriptions = numpy.array([" " * (l - len(each)) + each for each in descriptions]).astype(numpy.chararray)
    descriptions = [ind[i] + ' | ' + descriptions[idx[i]] for i in range(0, idx.shape[0])]

    return mat2,descriptions, patterns[idx]
# ----------------------------------------------------------------------------------------------------------------------
def print_accuracy(labels_fact, labels_pred,patterns,filename = None):

    if (filename!=None):
        file = open(filename, 'w')
    else:
        file = None

    mat,descriptions,sorted_labels = preditions_to_mat(labels_fact, labels_pred,patterns)
    ind = numpy.array([('%3d' % i) for i in range(0, mat.shape[0])])
    TP = float(numpy.trace(mat))

    my_print_int(numpy.array(mat).astype(int),rows=descriptions,cols=ind.astype(int),file = file)

    print("Accuracy = %d/%d = %1.4f" % (TP, float(numpy.sum(mat)), float(TP/numpy.sum(mat))),file=file)
    print("Fails    = %d" % float(numpy.sum(mat) - TP),file=file)
    print(file=file)

    if (filename != None):
        file.close()
    return
# ----------------------------------------------------------------------------------------------------------------------
def print_reject_rate(labels_fact, labels_pred, labels_prob,filename=None):

    hit = numpy.array([labels_fact[i] == labels_pred[i] for i in range(0, labels_fact.shape[0])]).astype(int)
    mat = numpy.vstack((labels_prob,hit))
    mat = mat.T
    idx = numpy.argsort(mat[:,0]).astype(int)
    mat2 = numpy.vstack((mat[:,0][idx],mat[:,1][idx]))
    mat2 = mat2.T


    if (filename!=None):
        file = open(filename, 'w')
    else:
        file = None

    decisions,accuracy,th  = [],[],[]

    for i in range(0,mat2.shape[0]):
        dec = mat2.shape[0]-i
        hits = numpy.sum(mat2[i:,1])
        decisions.append(float(dec/mat2.shape[0]))
        #accuracy.append(float(hits/dec))
        accuracy.append(int(100*hits/dec)/100)
        th.append(mat2[i,0])

    decisions2, accuracy2, th2 = [], [], []

    if (filename == None):
        print()
        print()

    print('Dcsns\tAccrcy\tCnfdnc', file=file)
    for each in numpy.unique(accuracy):
        idx = smart_index(accuracy,each)[0]
        print('%1.2f\t%1.2f\t%1.2f' % (decisions[idx],accuracy[idx],th[idx]),file=file)

    if (filename!=None):
        file.close()


    return
# ----------------------------------------------------------------------------------------------------------------------

def print_top_fails(labels_fact, labels_pred, patterns,filename = None):

    if (filename!=None):
        file = open(filename, 'w')
    else:
        file = None

    mat = confusion_matrix(numpy.array(labels_fact).astype(numpy.int), numpy.array(labels_pred).astype(numpy.int))

    error, class1, class2 = [], [], []
    for i in range (0,mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if(i != j):
                error.append(mat[i,j])
                class1.append(i)
                class2.append(j)

    error = numpy.array(error)
    idx = numpy.argsort(-error).astype(int)

    if (filename == None):
        print()
        print('Typical fails:')


    for i in range(0,error.shape[0]):
        if(error[idx[i]]>0):
            print('%3d %s %s' % (error[idx[i]],patterns[class1[idx[i]]],patterns[class2[idx[i]]]),file=file)

    if (filename != None):
        file.close()

    return
# ----------------------------------------------------------------------------------------------------------------------

def split_annotation_file(folder_annotation,file_input,file_part1,file_part2, ratio=0.5,delim=' ',limit=1000000):

    with open(file_input) as f: lines = f.readlines()
    header = lines[0]
    lines = lines[1:]

    if limit<len(lines):
        idx = numpy.random.choice(len(lines),limit)
        lines = numpy.array(lines)[idx]



    part1, part2 = [],[]
    part1.append(header.split('\n')[0])
    part2.append(header.split('\n')[0])

    for each in lines:
        each = each.split('\n')[0]
        split = each.split(delim)
        #if not os.path.isfile(folder_annotation + split[0]):
        #    print('ERROR: ' + folder_annotation + split[0])
        #    continue

        #image = Image.open(folder_annotation + split[0])
        #if image is None:
        #    print('ERROR: ' + folder_annotation + split[0])
        #    continue

        if (random.random() > ratio):
            part1.append(each)
        else:
            part2.append(each)


    save_mat(part1, file_part1, delim=delim)
    save_mat(part2, file_part2, delim=delim)

    return
# ----------------------------------------------------------------------------------------------------------------------

def split_samples(input_folder, folder_part1, folder_part2, ratio=0.5):
    print("Split samples..")
    if not os.path.exists(folder_part1):
        os.makedirs(folder_part1)
    else:
        remove_files(folder_part1)
        remove_folders(folder_part1)

    if not os.path.exists(folder_part2):
        os.makedirs(folder_part2)
    else:
        remove_files(folder_part2)
        remove_folders(folder_part2)

    folder_list = [f for f in os.listdir(input_folder)]

    for f in folder_list:
        folder_name = input_folder + f
        if os.path.isdir(folder_name):
            os.makedirs(folder_part1 + f)
            os.makedirs(folder_part2 + f)
            print(f)
            file_list = [f for f in os.listdir(folder_name)]
            for file_name in file_list:
                if (random.random() > ratio):
                    copyfile(folder_name + '/' + file_name, folder_part1 + f + '/' + file_name)
                else:
                    copyfile(folder_name + '/' + file_name, folder_part2 + f + '/' + file_name)
    return


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def save_labels(out_filename, filenames, labels,append=0,delim='\t'):
    if filenames.shape[0]!= labels.shape[0]:
        return

    if(append== 0):
        f_handle = open(out_filename, "w")
        f_handle.write("header1%cheader2xx\n" % delim)
    else:
        f_handle = open(out_filename, "a")

    for i in range(0,labels.shape[0]):
        f_handle.write("%s%c%03d\n" % (filenames[i],delim,labels[i]))
        #f_handle.write("%s%c%s\n" % (filenames[i],delim,labels[i]))

    f_handle.close()

    #if append>0:
    #sort_labels(out_filename,delim)

    return


# ----------------------------------------------------------------------------------------------------------------------
def intersection(x11, x12, y11, y12, x21, x22, y21, y22):
    if ((y12 < y21) or (y22 < y11) or (x22 < x11) or (x12 < x21)):
        return 0
    else:
        dx = min(x12, x22) - max(x11, x21)
        dy = min(y12, y22) - max(y11, y21)
        return dx * dy
    return

# ----------------------------------------------------------------------------------------------------------------------
def list_to_chararray(input_list):
    bufer = numpy.array(list(input_list)).astype(numpy.chararray)
    bufer = ''.join(bufer)
    return bufer


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
    #plt.legend(loc="lower right")
    plt.grid(which='major', color='lightgray', linestyle='--')
    #fig.canvas.set_window_title(caption + ('AUC = %0.4f' % roc_auc))
    plt.set_title(caption)

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
def plot_hystogram(plt,H,label,SMAPE,xmin=None,xmax=None,ymax=None):

    plt.hist(H, bins='auto')
    plt.axis((xmin, xmax, 0, ymax))
    plt.legend([label[0] + ' SMAPE = %1.2f%%'%SMAPE[0]])
    plt.grid()
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_series(plt,S,labels=None,SMAPE=None):

    colors_list = list(('blue','gray', 'red', 'purple', 'green', 'orange', 'cyan', 'black','pink','darkblue','darkred','darkgreen', 'darkcyan'))
    patches = []

    x=numpy.arange(0,S.shape[0],1)

    if SMAPE is None:SMAPE = [0]*S.shape[1]
    if labels is None: labels = [''] * S.shape[1]

    for i in range(0,S.shape[1]):
        color = colors_list[i%len(colors_list)]
        plt.plot(x,S[:,i], lw=1, color=color, alpha=0.75)
        if S.shape[1]==1:
            patches.append(mpatches.Patch(color=color, label=labels[i]))
        else:
            patches.append(mpatches.Patch(color=color, label=labels[i]+' %1.2f%%'% SMAPE[i]))

    plt.grid()

    if labels is not None:
        plt.legend(handles=patches[1:],loc='upper left')

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_two_series(filename1, filename2, caption=''):

    mat1 = load_mat(filename1, dtype=numpy.float32, delim='\t')
    mat2 = load_mat(filename2, dtype=numpy.float32, delim='\t')

    if len(mat1.shape)==1:
        SMAPE = calc_series_SMAPE(numpy.array([mat1, mat2]).T)
        labels = [filename1.split('/')[-1], filename2.split('/')[-1]]
        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(hspace=0.01)
        fig.canvas.set_window_title(caption)
        plot_series(plt.subplot(1, 1, 1), numpy.vstack((mat1,mat2)).T, labels=labels,SMAPE=SMAPE)
        plt.tight_layout()
    else:
        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(hspace=0.01)
        fig.canvas.set_window_title(caption)
        labels = [filename1.split('/')[-1], filename2.split('/')[-1]]
        S = numpy.minimum(mat1.shape[1],5)
        for i in range(0,S):
            SMAPE = calc_series_SMAPE(numpy.array([mat1[:,i], mat2[:,i]]).T)
            plot_series(plt.subplot(S, 1, i+1), numpy.vstack((mat1[:,i], mat2[:,i])).T, labels=labels, SMAPE=SMAPE)
        plt.tight_layout()

    return
# ---------------------------------------------------------------------------------------------------------------------
def plot_multiple_series(filename_fact, list_filenames, target_column=0,caption=''):

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace =0.01)
    fig.canvas.set_window_title(caption)

    S = 1+len(list_filenames)
    Labels_train = numpy.array(['fact']+[each.split('/')[-1] for each in list_filenames])


    Series = []
    mat = load_mat(filename_fact, dtype=numpy.float32, delim='\t')
    if len(mat.shape) == 1:
        Series.append(mat)
    else:
        Series.append(mat[:, target_column])

    for filename in list_filenames:
        mat = load_mat(filename,dtype=numpy.float32,delim='\t')
        if len(mat.shape)==1:
            Series.append(mat)
        else:
            Series.append(mat[:, target_column])

    Series=numpy.array(Series).T

    SMAPE = calc_series_SMAPE(Series)

    for i in range(1, S):
        plot_series(plt.subplot(S-1,1,i), Series[:,[0,i]],labels=Labels_train[[0,i]],SMAPE=SMAPE[[0,i]])


    plt.tight_layout()


    return
# ---------------------------------------------------------------------------------------------------------------------
def get_roc_data_from_scores_file(path_scores):

    data = load_mat(path_scores, numpy.chararray, ' ')
    labels = (data [:, 0]).astype('float32')
    scores = data[:, 1:].astype('float32')

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return tpr,fpr,roc_auc
# ----------------------------------------------------------------------------------------------------------------------
def get_roc_data_from_scores_file_v2(path_scores_pos,path_scores_neg,delim='\t'):

    data = load_mat(path_scores_pos, numpy.chararray, delim)
    s1= data[1:, 1:].astype('float32')
    l1 = numpy.full(len(s1),1)

    data = load_mat(path_scores_neg, numpy.chararray, delim)
    s0= data[1:, 1:].astype('float32')
    l0 = numpy.full(len(s0), 0)

    labels = numpy.hstack((l0, l1))
    scores = numpy.vstack((s0, s1))

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return tpr,fpr,roc_auc
# ----------------------------------------------------------------------------------------------------------------------
def display_roc_curve_from_file(plt,fig,path_scores,caption=''):

    data = load_mat(path_scores, dtype = numpy.chararray, delim='\t')
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
        data = load_mat(foldername+filename,dtype=numpy.chararray, delim='\t')
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
def plot_2D_scores_multi_Y(plt,X,Y,labels=None):

    colors_list = list(('red', 'blue', 'green', 'orange', 'cyan', 'purple','black','gray','pink','darkblue'))
    patches = []

    i=0
    for each in numpy.unique(Y):
        idx = numpy.where(Y==each)
        plt.plot(X[idx, 0], X[idx, 1], 'ro', color=colors_list[i%len(colors_list)], alpha=0.4)
        if labels is not None:
            patches.append(mpatches.Patch(color=colors_list[i%len(colors_list)],label=labels[i]))
        i+=1

    plt.grid()

    if labels is not None:
        plt.legend(handles=patches)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_2D_scores(plt,fig,filename_data_pos,filename_data_neg,filename_data_grid,filename_scores_grid,th,noice_needed=0,caption='',filename_out=None):
    if not os.path.isfile(filename_data_pos): return
    if not os.path.isfile(filename_data_neg): return
    if not os.path.isfile(filename_data_grid): return

    data = load_mat(filename_scores_grid, dtype=numpy.chararray, delim='\t')
    data = data[1:,:]
    grid_scores = data[:, 1:].astype('float32')

    data = load_mat(filename_data_grid, dtype=numpy.chararray, delim='\t')
    data_grid = data[:,1:].astype('float32')

    data = load_mat(filename_data_pos, dtype=numpy.chararray, delim='\t')
    l1 = (data[:, 0]).astype('float32')
    x1 = data[:,1:].astype('float32')

    data = load_mat(filename_data_neg, dtype=numpy.chararray, delim='\t')
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

    mat = load_mat(filename_mat, dtype=numpy.chararray, delim='\t')
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

    mat = load_mat(filename_mat, dtype=numpy.chararray, delim='\t')
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
def plot_features_PCA(plt,features,Y,patterns):

    X_TSNE = TSNE(n_components=2).fit_transform(features)
    plot_2D_scores_multi_Y(plt, X_TSNE, Y, labels=patterns)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_feature_importance(plt,fig,X,Y,header):

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
    plt.set_title('Feature importance')

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

    confusion_mat = load_mat(filename_mat,dtype=numpy.chararray,delim='\t')[:,:2]
    patterns = numpy.unique(confusion_mat[:, 0])

    labels_fact = numpy.array([smart_index(patterns, each) for each in confusion_mat[:, 0]])
    labels_pred = numpy.array([smart_index(patterns, each) for each in confusion_mat[:, 1]])


    mat, descriptions,sorted_labels = preditions_to_mat(labels_fact, labels_pred, numpy.unique(confusion_mat[:,0]))
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
def from_categorical(Y_2d):
    u = numpy.unique(Y_2d)
    Y = numpy.zeros(Y_2d.shape[0]).astype(int)
    for i in range(0, Y.shape[0]):
        #debug = Y_2d[i]
        index = smart_index(Y_2d[i],1)[0]
        Y[i]=index

    return Y
# ----------------------------------------------------------------------------------------------------------------------
def to_categorical(Y):
    u = numpy.unique(Y)
    Y_2d = numpy.zeros((Y.shape[0],u.shape[0])).astype(int)
    for i in range(0, Y.shape[0]):
        index = smart_index(u,Y[i])
        Y_2d[i,index]=1
    return Y_2d

# --------------------------------------------------------------------------------------------------------------------
def save_MNIST_digits(out_path,limit=200):

    digits = datasets.load_digits()
    data = digits.images.reshape((len(digits.images), -1))

    remove_files(out_path, create=True)
    remove_folders(out_path)

    for i in range(0, 10):
        os.makedirs(out_path + '/' + ('%d' % i))

    idx = []
    filenames=[]
    cntr = numpy.zeros(10)


    for i in range(0,data.shape[0]):
        key = digits.target[i].astype(int)
        name = key.astype(numpy.str)
        if cntr[key]<limit:
            idx.append(i)
            filenames.append(name + '/' + name + ('_%03d' % cntr[key]) + '.png')
            cntr[key] += 1

    save_flatarrays_as_images(out_path, 8,8, data[idx], labels=digits.target[idx], filenames=filenames)

    return
# ----------------------------------------------------------------------------------------------------------------------
def numerical_devisor(n):

    for i in numpy.arange(int(math.sqrt(n))+1,1,-1):
        if n%i==0:
            return i

    return n
#--------------------------------------------------------------------------------------------------------------------------
