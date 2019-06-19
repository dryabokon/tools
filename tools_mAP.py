import numpy
import os
import matplotlib.pyplot as plt
import tools_IO
import tools_image
import cv2
import tools_YOLO
import progressbar
from PIL import Image
# ----------------------------------------------------------------------------------------------------------------------
def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
# ----------------------------------------------------------------------------------------------------------------------
def calc_hits_stats(lines_true,lines_pred,class_ID,delim,folder_annotation=None,iuo_th=0.5):
    file_true, file_pred = [], []
    coord_true, coord_pred, = [], []
    conf_true, conf_pred = [], []
    hit_true, hit_pred = [], []

    for line in lines_true:
        split = line.split(delim)
        if int(split[5]) == class_ID:
            x_min,y_min,x_max,y_max = int(split[1]),int(split[2]),int(split[3]),int(split[4])
            if folder_annotation is not None:
                if not os.path.isfile(folder_annotation + split[0]):continue
                image = Image.open(folder_annotation + split[0])
                if image is None: continue
                width, height = image.size
                x_min, y_min = tools_image.smart_resize_point(x_min, y_min, width,height, 416, 416)
                x_max, y_max = tools_image.smart_resize_point(x_max, y_max, width,height, 416, 416)

            file_true.append(split[0].split('/')[-1])
            coord_true.append([x_min,y_min,x_max,y_max])
            conf_true.append(float(0))
            hit_true.append(0)

    for line in lines_pred:
        split = line.split(delim)
        if int(split[5]) == class_ID:
            x_min,y_min,x_max,y_max = int(split[1]),int(split[2]),int(split[3]),int(split[4])
            if folder_annotation is not None:
                if not os.path.isfile(folder_annotation + split[0]):continue
                image = Image.open(folder_annotation + split[0])
                if image is None: continue
                width, height = image.size
                x_min, y_min = tools_image.smart_resize_point(x_min, y_min, width, height,416, 416)
                x_max, y_max = tools_image.smart_resize_point(x_max, y_max, width, height,416, 416)
            file_pred.append(split[0].split('/')[-1])
            coord_pred.append([x_min, y_min, x_max, y_max])
            conf_pred.append(float(split[6]))
            hit_pred.append(0)

    for j, filename_true in enumerate(file_true):
        best_i, best_iou, best_conf = None, -1, None
        for i, filename_pred in enumerate(file_pred):
            if filename_true == filename_pred:
                iuo_value = iou(coord_true[j], coord_pred[i])
                if iuo_value >= iuo_th and iuo_value > best_iou:
                    best_i, best_iou, best_conf = i, iuo_value, conf_pred[i]
        if best_i is not None:
            hit_true[j], conf_true[j] = 1, best_conf
            hit_pred[best_i], conf_pred[best_i] = 1, best_conf
        else:
            hit_true[j], conf_true[j] = 0, float(-1)


    conf_true, conf_pred = numpy.array(conf_true), numpy.array(conf_pred)
    hit_true , hit_pred  = numpy.array(hit_true ), numpy.array(hit_pred)
    file_true, file_pred = numpy.array(file_true ), numpy.array(file_pred)
    coord_true, coord_pred=numpy.array(coord_true ), numpy.array(coord_pred)

    return file_true, file_pred, coord_true, coord_pred, conf_true, conf_pred, hit_true, hit_pred
# ----------------------------------------------------------------------------------------------------------------------
def get_precsion_recall_data_from_markups(file_markup_true, file_markup_pred,iuo_th,delim=' '):

    dict_classes={}
    with open(file_markup_true) as f:lines_true = f.readlines()[1:]
    with open(file_markup_pred) as f:lines_pred = f.readlines()[1:]
    for line in lines_true: dict_classes[int(line.split(delim)[5])] = 0
    for line in lines_pred: dict_classes[int(line.split(delim)[5])] = 0

    precisions, recalls, confidences, class_IDs  = [],[],[],[]

    for class_ID in sorted(set(dict_classes.keys())):

        file_true, file_pred, coord_true, coord_pred, conf_true, conf_pred, hit_true, hit_pred = calc_hits_stats(lines_true,lines_pred,class_ID,delim,iuo_th)
        if len(file_true)==0:
            continue

        relevant_file = numpy.array([each in file_true for each in file_pred])

        ths={}
        for each in sorted(conf_true):
            if each>=iuo_th:ths[each]=0
        precision, recall, conf = [],[],[]

        for cnt,th in enumerate(ths):

            iii = numpy.where(conf_true>=th)
            recall_TP = numpy.sum(hit_true[iii])
            FN = len(hit_true)-recall_TP
            recall.append(float(recall_TP/(recall_TP+FN)))

            prec_TP = numpy.sum(hit_pred[conf_pred>=th])
            idx = numpy.where(relevant_file==True)
            FP = len(hit_pred[idx][conf_pred[idx]>=th])-prec_TP
            if prec_TP + FP>0:
                precision.append(float(prec_TP / (prec_TP + FP)))
            else:
                precision.append(0)
            conf.append(th)


        idx = numpy.argsort(-numpy.array(conf))
        precisions.append(numpy.array(precision)[idx])
        recalls.append(numpy.array(recall)[idx])
        confidences.append(numpy.array(conf)[idx])
        class_IDs.append(class_ID)

    return precisions,recalls,confidences,class_IDs
#--------------------------------------------------------------------------------------------------------------------------
def get_confidence_by_size(file_markup_true, file_markup_pred,iuo_th,delim=' '):

    dict_classes = {}
    with open(file_markup_true) as f:lines_true = f.readlines()[1:]
    with open(file_markup_pred) as f:lines_pred = f.readlines()[1:]
    for line in lines_true: dict_classes[int(line.split(delim)[5])] = 0
    for line in lines_pred: dict_classes[int(line.split(delim)[5])] = 0

    size, confidences, class_IDs = [], [], []

    for class_ID in sorted(set(dict_classes.keys())):
        file_true, file_pred, coord_true, coord_pred, conf_true, conf_pred, hit_true, hit_pred = calc_hits_stats(lines_true, lines_pred, class_ID, delim, iuo_th)

        ths = {}
        for each in sorted(conf_true):
            if each >= iuo_th: ths[each] = 0

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_precision_recall(plt,figure,precision,recall,caption=''):

    AP = numpy.trapz(precision, x=recall)

    lw = 2
    plt.plot(recall,precision, color='darkgreen', lw=lw, label='AP = %0.2f' % AP)
    plt.plot([0, 1.0], [1.0, 0], color='lightgray', lw=lw, linestyle='--')
    plt.set_title(caption + (' %0.4f' % AP))
    plt.grid(which='major', color='lightgray', linestyle='--')
    plt.minorticks_on()
    plt.grid(which='minor', axis='both', color='lightgray', linestyle='--')

    ax = figure.gca()
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')

    return
# ----------------------------------------------------------------------------------------------------------------------
def write_precision_recall(filename_out,precision,recall,caption='',color=(128,128,128)):

    figure = plt.figure()

    AP = numpy.trapz(precision, x=recall)

    lw = 2
    plt.plot(recall,precision, color=color, lw=lw, label='AP = %0.2f' % AP)
    plt.plot([0, 1.0], [1.0, 0], color='lightgray', lw=lw, linestyle='--')
    plt.title(caption + (' %0.4f' % AP))

    plt.grid(which='major', color='lightgray', linestyle='--')
    plt.minorticks_on()
    plt.grid(which='minor', axis='both', color='lightgray', linestyle='--')

    ax = figure.gca()
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')

    plt.savefig(filename_out)
    return AP
# ----------------------------------------------------------------------------------------------------------------------
def write_boxes_distribution(filename_out,true_boxes):

    figure = plt.figure()

    w, h = [], []

    for boxes in true_boxes:
        for box in boxes:
            if (box[2] - box[0]) * (box[3] - box[1]) > 0:
                w.append(box[2] - box[0])
                h.append(box[3] - box[1])

    plt.scatter(w,h,alpha=0.5,s=10,lw = 0)
    plt.grid(which='major', color='lightgray', linestyle='-')
    ax = figure.gca()
    ax.set_xlabel('width')
    ax.set_ylabel('height')

    plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def analyze_markups_mAP(file_markup_true, file_markup_pred,filename_meta, folder_out,out_prefix='',delim=' '):

    input_image_size, class_names, anchors, anchor_mask,obj_threshold, nms_threshold = tools_YOLO.load_metadata(filename_meta)
    colors = tools_YOLO.generate_colors(len(class_names))

    iuo_ths = [0.5,0.3,0.1,0.01]
    results = []
    for iuo_th in iuo_ths:

        out_dir = folder_out + '%02d/' % int(iuo_th * 100)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        precisions,recalls,confidences,class_IDs = get_precsion_recall_data_from_markups(file_markup_true, file_markup_pred,iuo_th, delim=' ')
        mAP = 0

        for i,class_ID in enumerate(class_IDs):
            if len(precisions[i])>1:
                filename_out = out_dir + out_prefix + 'AP_%02d_%s.png' % (class_ID, class_names[class_ID])
                AP = write_precision_recall(filename_out,precisions[i], recalls[i],caption=class_names[class_ID],color=(colors[class_ID][2]/255.0,colors[class_ID][1]/255.0,colors[class_ID][0]/255.0))
                mAP +=AP

        results.append(mAP/len(class_IDs))
        print(out_prefix,iuo_th,mAP/len(class_IDs))

    print()

    return results[0]
# ----------------------------------------------------------------------------------------------------------------------
def analyze_markups_draw_boxes(class_ID,file_markup_true, file_markup_pred,path_out, delim=' ',metric='recall',iou_th=0.1):

    tools_IO.remove_files(path_out,create=True)

    foldername = '/'.join(file_markup_true.split('/')[:-1]) + '/'
    with open(file_markup_true) as f:lines_true = f.readlines()[1:]
    with open(file_markup_pred) as f:lines_pred = f.readlines()[1:]

    file_true, file_pred, coord_true, coord_pred, conf_true, conf_pred, hit_true, hit_pred = calc_hits_stats(lines_true,lines_pred,class_ID,delim,iuo_th=iou_th)

    red=(0,32,255)
    amber=(0,192,255)
    green=(0,192,0)
    marine =(64,192,0)
    hit_colors_true = [red  , green]
    hit_colors_pred = [amber, marine]
    descript_ion = []

    bar = progressbar.ProgressBar(max_value=len(set(file_true)))

    for b,filename in enumerate(set(file_true)):
        bar.update(b)
        image = cv2.imread(foldername + filename)
        if image is None:
            continue
        image = tools_image.desaturate(image)
        is_hit=1
        is_FP=1
        idx = numpy.where(file_true==filename)
        for coord, hit in zip(coord_true[idx],hit_true[idx]):
            cv2.rectangle(image,(coord[0], coord[1]),(coord[2], coord[3]),hit_colors_true[hit],thickness=2)
            is_hit = min(is_hit,hit)
        idx = numpy.where(file_pred == filename)
        for coord, hit in zip(coord_pred[idx],hit_pred[idx]):
            cv2.rectangle(image,(coord[0], coord[1]),(coord[2], coord[3]),hit_colors_pred[hit],thickness=1)
            is_FP = min(is_FP, hit)

        if metric=='recall':
            descript_ion.append([filename, is_hit])
        else:
            descript_ion.append([filename, is_FP])

        cv2.imwrite(path_out+filename,image)
        tools_IO.save_mat(descript_ion,path_out+'descript.ion',delim=' ')

    return
# ----------------------------------------------------------------------------------------------------------------------
def analyze_hit_miss_sizes(class_ID,folder_annotation,file_markup_true, file_markup_pred,path_out, delim=' ',iou_th=0.1):


    with open(file_markup_true) as f:lines_true = f.readlines()[1:]
    with open(file_markup_pred) as f:lines_pred = f.readlines()[1:]

    file_true, file_pred, coord_true, coord_pred, conf_true, conf_pred, hit_true, hit_pred = calc_hits_stats(lines_true,lines_pred,class_ID,delim,folder_annotation=folder_annotation,iuo_th=iou_th)

    hit_boxes,miss_boxes = [[]],[[]]
    for box,hit in zip(coord_true,hit_true):
        if hit:
            hit_boxes.append([box])
        else:
            miss_boxes.append([box])

    write_boxes_distribution(path_out + 'boxes_hit.png', hit_boxes)
    write_boxes_distribution(path_out + 'boxes_miss.png', miss_boxes)
    return
# ----------------------------------------------------------------------------------------------------------------------