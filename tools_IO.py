import numpy
import os
from os import listdir
import fnmatch
import json
from shutil import copyfile
import shutil
import random
from sklearn import metrics, datasets
from sklearn.metrics import confusion_matrix, auc
import cv2
import math
import pickle
import pandas as pd
import operator
from itertools import groupby
import hashlib
import difflib
# ----------------------------------------------------------------------------------------------------------------------
def find_nearest(array, value):
    return array[(numpy.abs(array - value)).argmin()]
# ----------------------------------------------------------------------------------------------------------------------
def smart_index(array, value):
    return numpy.array([i for i, v in enumerate(array) if (v == value)])

# ----------------------------------------------------------------------------------------------------------------------
def flatten_list(A):
    res = [i for I in A
             for i in I]
    return res
# ----------------------------------------------------------------------------------------------------------------------
def rank(A):
    rank = numpy.arange(0, A.shape[0])
    idx = numpy.argsort(-A)
    rank[idx] = 1 + numpy.arange(0, A.shape[0])
    return rank
# ----------------------------------------------------------------------------------------------------------------------
def remove_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)
# ----------------------------------------------------------------------------------------------------------------------
def remove_files(path,list_of_masks='*.*',create=False):

    if not os.path.exists(path):
        if create:
            os.mkdir(path)
        return

    filenames = get_filenames(path, list_of_masks)
    for f in filenames:
        if os.path.isdir(path + f):
            #shutil.rmtree(path + f)
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
def copy_folder(folder_in,folder_out):

    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    filenames = get_filenames(folder_in, '*.*')
    for filename in filenames:
        if os.path.isfile(folder_in + filename):
            copyfile(folder_in + filename, folder_out + filename)


    return
# ----------------------------------------------------------------------------------------------------------------------
def compare_files(filename1,filename2):

    with open(filename1) as f:lines1 = f.readlines()
    with open(filename2) as f:lines2 = f.readlines()
    lines1 = [l.split('\n')[0] for l in lines1]
    lines2 = [l.split('\n')[0] for l in lines2]

    d = difflib.Differ()
    diff = list(d.compare(lines1, lines2))
    diff = [d for d in diff if ('+ ' in d[:2] or '- 'in d[:2])]

    return diff
# ----------------------------------------------------------------------------------------------------------------------
def compare_folders(folder1, folder2):
    filenames1 = get_filenames(folder1,'*.*')
    filenames2 = get_filenames(folder2,'*.*')

    d = difflib.Differ()
    diff = list(d.compare(filenames1, filenames2))
    same = [d[2:] for d in diff if d[:2]=='  ']
    diff = [d for d in diff if ('+ ' in d or '- ' in d)]

    for filename in same:
        if filename == 'Tax Area.csv':
            ii=0
        x = compare_files(folder1+filename, folder2+filename)
        if len(x)>0:
            diff.append('* '+filename)

    idx = numpy.argsort([d[2:] for d in diff])
    diff = numpy.array(diff)[idx]

    return diff
# ----------------------------------------------------------------------------------------------------------------------
def append_CSV(csv_record, filename_out, csv_header=None, delim ='\t'):

    mode = 'a' if os.path.exists(filename_out) else 'w'

    with open(filename_out, mode, encoding='utf-8') as f:
        if mode=='w' and csv_header is not None:
            for i, each in enumerate(csv_header):
                f.write(str(each))
                if i != len(csv_header) - 1:
                    f.write(delim)
            f.write('\n')

        if csv_record is not None and len(csv_record)>0:
            for i,each in enumerate(csv_record):
                if type(each) is str:
                    f.write(each)
                else:
                    f.write('%2.1f'%float(each))

                if i!=len(csv_record)-1:
                    f.write(delim)

            f.write('\n')

    return
# ----------------------------------------------------------------------------------------------------------------------
def get_filenames(path_input,list_of_masks):
    local_filenames = []
    for mask in list_of_masks.split(','):
        res = listdir(path_input)
        if mask != '*.*':
            res = fnmatch.filter(res, mask)
        local_filenames += res

    return numpy.sort(numpy.array(local_filenames))
# ----------------------------------------------------------------------------------------------------------------------
def get_next_folder_out(base_folder_out):
    sub_folders = get_sub_folder_from_folder(base_folder_out)
    if len(sub_folders) > 0:
        sub_folders = numpy.array(sub_folders, dtype=numpy.int)
        sub_folders = numpy.sort(sub_folders)
        sub_folder_out = '%03d'%(sub_folders[-1] + 1)
    else:
        sub_folder_out = '%03d'%(0)
    return base_folder_out + sub_folder_out + '/'
# ----------------------------------------------------------------------------------------------------------------------
def save_raw_vec(vec, filename,mode=(os.O_RDWR|os.O_APPEND),fmt='%d',delim=' '):

    if not os.path.isfile(filename):
        mode = os.O_RDWR | os.O_CREAT

    f_handle = os.open(filename,mode)

    L = len(vec)
    s = ""
    for i in range(L-1):
        value = ((fmt+delim) % vec[i]).encode()
        os.write(f_handle,value)

    value = ((fmt+'\n') % vec[L-1]).encode()
    os.write(f_handle, value)
    os.close(f_handle)

    return
# ----------------------------------------------------------------------------------------------------------------------
def save_mat(mat, filename,fmt='%s',delim='\t'):
    numpy.savetxt(filename, mat,fmt=fmt,delimiter=delim)
    return
# ----------------------------------------------------------------------------------------------------------------------
def save_dict(dct,filename_out):
    with open(filename_out, 'w') as f:
        json.dump(dct, f, indent=' ')
    return
# ----------------------------------------------------------------------------------------------------------------------
def load_dict(filename_in):
    with open(filename_in, 'r') as f:
        dct = json.load(f)
    return dct
# ----------------------------------------------------------------------------------------------------------------------
def save_data_to_feature_file_float(filename,array,target):

    m = numpy.array(array).astype('float32')
    v = numpy.matrix(target).astype('float32')
    mat = numpy.concatenate((v.T,m),axis=1)
    #print(mat)
    numpy.savetxt(filename, mat, fmt='%+2.2f',delimiter='\t')
    return

# ----------------------------------------------------------------------------------------------------------------------
def count_columns(filename,delim='\t'):

    C = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                line = line.split(delim)
                C.append(len(line))

    C = numpy.array(C)
    return C.max()
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
def get_lines(filename,delim='\t',start=None,end=None):

    lines = []
    with open(filename, 'r') as f:
        for i,line in enumerate(f):
            line = line.strip()
            if len(line) > 0:
                if (start is not None and i < start):
                    continue
                if  (end is not None and i>=end):
                    continue
                lines.append(line.split(delim))

    return lines
# ----------------------------------------------------------------------------------------------------------------------
def get_columns(filename,delim='\t',start=None,end=None):

    columns = []
    with open(filename, 'r') as f:
        for i,line in enumerate(f):
            line = line.strip()
            if len(line) > 0 :
                columns.append(line.split(delim)[start:end])

    columns = numpy.array(columns)

    return columns
# ----------------------------------------------------------------------------------------------------------------------
def load_mat(filename, dtype=numpy.chararray, delim='\t'):
    N = count_lines(filename)
    mat = []

    if N==0:
        return mat
    #elif N==1:
    #    mat  = numpy.array([numpy.genfrpickleomtxt(filename, dtype=dtype, delimiter=delim)])
    else:
        mat = numpy.genfromtxt(filename, dtype=dtype, delimiter=delim)
    return mat
# ----------------------------------------------------------------------------------------------------------------------
def load_mat_pd(filename,delim='\t'):
    return pd.read_csv(filename, sep=delim).values
# ----------------------------------------------------------------------------------------------------------------------
def load_mat_var_size(filename,delim='\t'):
    l=[]
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                l.append(line.split(delim))
    return l
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

# ---------------------------------------------------------------------------------------------------------------------
def get_roc_data_from_scores_file(path_scores,has_header=False):

    X = load_mat(path_scores, numpy.chararray, '\t')

    if has_header:
        header = numpy.array(X[0,:],dtype=numpy.str)
        X  = X[1:,:]
    else:
        header = None
        X = X[:,:]

    labels = (X[:, 0]).astype('float32')
    scores = X[:, 1:].astype('float32')

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

# ----------------------------------------------------------------------------------------------------------------------
def numerical_devisor(n):

    for i in numpy.arange(int(math.sqrt(n))+1,1,-1):
        if n%i==0:
            return i

    return n
#--------------------------------------------------------------------------------------------------------------------------
def load_if_exists(folder_in,suffix,name,use_cache=False):
    X, success = None, False

    if (folder_in is None) or (suffix is None) or (name is None):return X,success

    filename_in = folder_in+suffix+name

    if os.path.isfile(filename_in)==False: return X, success

    with open(filename_in, "rb") as fp:
        X = pickle.load(fp)
        success = True

    return X,success
# ---------------------------------------------------------------------------------------------------------------------
def write_cache(folder_out,suffix,name,X):
    if (folder_out is None) or (suffix is None) or (name is None):
        return

    filename_out = folder_out + suffix + name

    with open(filename_out, "wb") as fp:
        pickle.dump(X, fp)
    return
# ---------------------------------------------------------------------------------------------------------------------
def min_element_by_value(dct):
    return min(dct.items(), key=operator.itemgetter(1))
# ---------------------------------------------------------------------------------------------------------------------
def min_element_by_key(dct):
    return min(dct.items(), key=operator.itemgetter(0))
# ---------------------------------------------------------------------------------------------------------------------
def max_element_by_value(dct):
    return max(dct.items(), key=operator.itemgetter(1))
# ---------------------------------------------------------------------------------------------------------------------
def max_element_by_key(dct):
    return max(dct.items(), key=operator.itemgetter(0))
# ---------------------------------------------------------------------------------------------------------------------
def sorted_elements_by_value(dct,descending=False):
    if descending:
        listofTuples = sorted(dct.items(), key=lambda x: -x[1])
    else:
        listofTuples = sorted(dct.items(), key=lambda x:  x[1])
    return listofTuples
# ---------------------------------------------------------------------------------------------------------------------
def switch_comumns(filename_in,filename_out,idx,has_header=False,delim='\t',max_line=None):
    g = open(filename_out, 'w')

    with open(filename_in, 'r') as f:
        for i, line in enumerate(f):

            if has_header and i == 0:
                g.write("%s\n" % line)
                continue

            line = line.strip()
            if len(line) > 0:
                X = numpy.array(line.split(delim),dtype=numpy.int)
                X=X[idx]

                for x in X:
                    g.write("%d\t" % x)

                g.write("\n")
                if (max_line is not None) and (i>=max_line-1):
                    break

    g.close()

    return
# ----------------------------------------------------------------------------------------------------------------------
def get_longest_run_position_len(L):
    if numpy.all(L<=0):return -1,-1
    xx = max(((lambda y: (y[0][0], len(y)))(list(g)) for k, g in groupby(enumerate(L), lambda x: x[1]) if k),key=lambda z: z[1])
    return xx
# ----------------------------------------------------------------------------------------------------------------------
def gen_hash(text):
    res = hashlib.sha224(text.encode("utf-8")).hexdigest()
    return res
# ----------------------------------------------------------------------------------------------------------------------
def validate_str(value):

    if pd.DataFrame([value]).isna()[0][0]:
        res = 'NA'
    else:
        res = str(value)
        res = res.replace('/', '')
        res = res.replace('*', '_total_')

    return res
# ----------------------------------------------------------------------------------------------------------------------
