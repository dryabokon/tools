import numpy
import matplotlib.pyplot as plt
import cv2
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
def iou(boxA, boxB):
    boxA[0],boxA[2] = min(boxA[0],boxA[2]),max(boxA[0],boxA[2])
    boxA[1],boxA[3] = min(boxA[1],boxA[3]),max(boxA[1],boxA[3])
    boxB[0],boxB[2] = min(boxB[0], boxB[2]), max(boxB[0], boxB[2])
    boxB[1],boxB[3] = min(boxB[1], boxB[3]), max(boxB[1], boxB[3])

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
def ovelraps(box_true, box_pred):

    xT = max(box_true[0], box_pred[0])
    yT = max(box_true[1], box_pred[1])
    xP = min(box_true[2], box_pred[2])
    yP = min(box_true[3], box_pred[3])

    inter_area = max(0, xP - xT + 1) * max(0, yP - yT + 1)
    box_true_area = (box_true[2] - box_true[0] + 1) * (box_true[3] - box_true[1] + 1)
    box_pred_area = (box_pred[2] - box_pred[0] + 1) * (box_pred[3] - box_pred[1] + 1)

    ovp =  inter_area/box_true_area
    ovd = 1- inter_area/box_pred_area
    return ovp,ovd
# ----------------------------------------------------------------------------------------------------------------------
def calc_hits_stats_iou(lines_true, lines_pred, class_ID, delim, iuo_th=0.5,ovp_th=None,ovd_th=None):
    file_true, file_pred = [], []
    coord_true, coord_pred, = [], []
    conf_true, conf_pred = [], []
    hit_true, hit_pred = [], []

    for line in lines_true:
        split = line.split(delim)
        if int(split[5].split()[-1]) == class_ID:
            x_min,y_min,x_max,y_max = int(split[1]),int(split[2]),int(split[3]),int(split[4])
            file_true.append(split[0])
            coord_true.append([x_min,y_min,x_max,y_max])
            conf_true.append(float(-1))
            hit_true.append(0)

    for line in lines_pred:
        split = line.split(delim)
        if int(split[5]) == class_ID:
            x_min,y_min,x_max,y_max = int(split[1]),int(split[2]),int(split[3]),int(split[4])

            file_pred.append(split[0])
            coord_pred.append([x_min, y_min, x_max, y_max])
            conf_pred.append(float(split[6]))
            hit_pred.append(0)

    if iuo_th is not None:
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
    else:
        for j, filename_true in enumerate(file_true):
            for i, filename_pred in enumerate(file_pred):
                if filename_true == filename_pred:

                    ovp_value,ovd_value = ovelraps(coord_true[j], coord_pred[i])
                    if ovp_value >= ovp_th and ovd_value <= ovd_th:
                        hit_pred[i] = 1
                        hit_true[j] = 1
                        conf_true[j] = max(conf_true[j],conf_pred[i])

    conf_true, conf_pred = numpy.array(conf_true), numpy.array(conf_pred)
    hit_true , hit_pred  = numpy.array(hit_true ), numpy.array(hit_pred)
    file_true, file_pred = numpy.array(file_true ), numpy.array(file_pred)
    coord_true, coord_pred=numpy.array(coord_true ), numpy.array(coord_pred)

    return file_true, file_pred, coord_true, coord_pred, conf_true, conf_pred, hit_true, hit_pred
# ----------------------------------------------------------------------------------------------------------------------
def get_precsion_recall_data_from_markups(file_markup_true, file_markup_pred,iuo_th,ovp_th=None,ovd_th=None,delim=' '):

    dict_classes={}
    with open(file_markup_true) as f:lines_true = f.readlines()[1:]
    with open(file_markup_pred) as f:lines_pred = f.readlines()[1:]
    for line in lines_true: dict_classes[int(line.split(delim)[5])] = 0
    for line in lines_pred: dict_classes[int(line.split(delim)[5])] = 0

    precisions, recalls, confidences, class_IDs  = [],[],[],[]

    for class_ID in sorted(set(dict_classes.keys())):

        file_true, file_pred, coord_true, coord_pred, conf_true, conf_pred, hit_true, hit_pred = calc_hits_stats_iou(lines_true, lines_pred, class_ID, delim, iuo_th,ovp_th, ovd_th)
        if len(file_true)==0:
            continue



        relevant_file = numpy.array([each in file_true for each in file_pred])

        ths={}
        for each in sorted(conf_true):
            #if each>=iuo_th:
            ths[each]=0
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
# ----------------------------------------------------------------------------------------------------------------------
def plot_precision_recall(precision,recall,filename_out=None):

    AP = numpy.trapz(precision, x=recall)
    lw = 2
    plt.plot(recall, precision, color='darkgreen', lw=lw, label='AP = %0.2f' % AP)
    plt.grid(which='major', color='lightgray', linestyle='--')
    plt.minorticks_on()
    plt.grid(which='minor', axis='both', color='lightgray', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    if filename_out is not None:
        plt.savefig(filename_out)

    plt.clf()

    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_boxes(class_ID, folder_annotation, file_markup_true, file_markup_pred, path_out, delim=' ', metric='recall', confidence=0.10,iou_th=0.1, ovp_th=0.5, ovd_th=0.5):

    #tools_IO.remove_files(path_out,create=True)
    tools_IO.remove_folders(path_out)
    if metric == 'recall':
        tools_IO.remove_files(path_out+'miss',create=True)
        tools_IO.remove_files(path_out+'hit', create=True)
    else:
        tools_IO.remove_files(path_out + 'TP', create=True)
        tools_IO.remove_files(path_out + 'FP', create=True)

    with open(file_markup_true) as f:lines_true = f.readlines()[1:]
    with open(file_markup_pred) as f:lines_pred = f.readlines()[1:]

    file_true, file_pred, coord_true, coord_pred, conf_true, conf_pred, hit_true, hit_pred = calc_hits_stats_iou(lines_true, lines_pred,class_ID, delim,iuo_th=iou_th,ovp_th=ovp_th,ovd_th=ovd_th)
    hit_colors_true = [(0,0,200)  , (0,192,0)]
    hit_colors_pred = [(0,192,255), (128,128,0)]

    if metric == 'recall':

        for b,filename in enumerate(set(file_true)):

            image = cv2.imread(folder_annotation + filename)
            if image is None:continue

            image = tools_image.desaturate(image)
            is_hit=0
            idx = numpy.where(file_true==filename)
            for coord, hit, conf in zip(coord_true[idx],hit_true[idx],conf_true[idx]):
                if conf<confidence:
                    hit = 0
                image = tools_draw_numpy.draw_rect(image, coord[0], coord[1],coord[2], coord[3] ,color=hit_colors_true[hit], w=2, alpha_transp=0.0)
                is_hit = max(is_hit,hit)

            cv2.imwrite(path_out+('hit' if is_hit else 'miss')+'/'+filename.split('/')[-1],image)
    else:
        for b, filename in enumerate(set(file_true)):
            image = cv2.imread(folder_annotation + filename)
            if image is None: continue

            image = tools_image.desaturate(image)

            is_FP = 1

            idx = numpy.where(file_pred == filename)

            for coord, hit, conf in zip(coord_pred[idx],hit_pred[idx],conf_pred[idx]):
                if conf<confidence:
                    continue
                image = tools_draw_numpy.draw_rect(image, coord[0], coord[1], coord[2], coord[3],color=hit_colors_pred[hit], w=2, alpha_transp=0.0)
                is_FP = min(is_FP, 1-hit)

            cv2.imwrite(path_out + ('FP' if is_FP else 'TP') + '/' + filename.split('/')[-1], image)

    return
# ----------------------------------------------------------------------------------------------------------------------