import numpy
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
#from pycocotools.coco import COCO
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_IO
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Benchmarker(object):

    def __init__(self,folder_out,folder_images):
        self.folder_out=folder_out
        self.folder_images = folder_images
        self.hit_colors_true = [(0, 0, 200), (0, 192, 0)]
        self.hit_colors_pred = [(0, 192, 255), (128, 128, 0)]
        return
# ----------------------------------------------------------------------------------------------------------------------
    def iou(self,boxA, boxB,do_debug=False):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if do_debug:
            image = numpy.zeros((max(boxA+boxB),max(boxA+boxB),3),dtype=numpy.uint8)
            image = tools_draw_numpy.draw_rect(image, boxA[0], boxA[1], boxA[2], boxA[3], (255, 128 ,0), w=1, alpha_transp=1)
            image = tools_draw_numpy.draw_rect(image, boxB[0], boxB[1], boxB[2], boxB[3], (0, 128, 255), w=1, alpha_transp=1)
            cv2.imwrite('./test.jpg',image)


        return iou
    # ----------------------------------------------------------------------------------------------------------------------
    def ovelraps(self,box_true, box_pred):

        box_true[0],box_true[2] = min(box_true[0],box_true[2]),max(box_true[0],box_true[2])
        box_true[1],box_true[3] = min(box_true[1],box_true[3]),max(box_true[1],box_true[3])
        box_pred[0],box_pred[2] = min(box_pred[0], box_pred[2]), max(box_pred[0], box_pred[2])
        box_pred[1],box_pred[3] = min(box_pred[1], box_pred[3]), max(box_pred[1], box_pred[3])

        xT = max(box_true[0], box_pred[0])
        yT = max(box_true[1], box_pred[1])
        xP = min(box_true[2], box_pred[2])
        yP = min(box_true[3], box_pred[3])

        inter_area = max(0, xP - xT + 1) * max(0, yP - yT + 1)
        box_true_area = (box_true[2] - box_true[0] + 1) * (box_true[3] - box_true[1] + 1)
        box_pred_area = (box_pred[2] - box_pred[0] + 1) * (box_pred[3] - box_pred[1] + 1)

        ovp =     inter_area/box_true_area
        ovd = 1 - inter_area/box_pred_area
        return ovp,ovd
    # ----------------------------------------------------------------------------------------------------------------------
    def calc_hits_stats_iou(self,df_true, df_pred, iuo_th=0.5,ovp_th=None,ovd_th=None):

        coord_true,coord_pred  = [],[]

        hit_true = numpy.zeros(df_true.shape[0])
        hit_pred = numpy.zeros(df_pred.shape[0])
        file_true = df_true.iloc[:,0].values
        file_pred = df_pred.iloc[:,0].values
        conf_true = numpy.full(df_true.shape[0],-1.0)
        conf_pred = df_pred.iloc[:, 5].values

        for r in range(df_true.shape[0]):
            x_min,y_min,x_max,y_max = int(df_true.iloc[r,1]),int(df_true.iloc[r,2]),int(df_true.iloc[r,3]),int(df_true.iloc[r,4])
            coord_true.append([x_min,y_min,x_max,y_max])

        for r in range(df_pred.shape[0]):
            x_min,y_min,x_max,y_max = int(df_pred.iloc[r,1]),int(df_pred.iloc[r,2]),int(df_pred.iloc[r,3]),int(df_pred.iloc[r,4])
            coord_pred.append([x_min, y_min, x_max, y_max])


        if iuo_th is not None:
            for j, filename_true in enumerate(file_true):
                best_i, best_iou, best_conf = None, -1, None
                for i, filename_pred in enumerate(file_pred):
                    if filename_true == filename_pred:
                        iuo_value = self.iou(coord_true[j], coord_pred[i])
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

                        ovp_value,ovd_value = self.ovelraps(coord_true[j], coord_pred[i])
                        if ovp_value >= ovp_th and ovd_value <= ovd_th:
                            hit_pred[i] = 1
                            hit_true[j] = 1
                            conf_true[j] = max(conf_true[j],conf_pred[i])
        df_true2 = df_true.copy()
        df_pred2 = df_pred.copy()
        df_true2['conf'] = numpy.array(conf_true)
        df_true2['hit'] = numpy.array(hit_true)
        df_pred2['hit'] = numpy.array(hit_pred)
        return df_true2,df_pred2
    # ----------------------------------------------------------------------------------------------------------------------
    def get_precsion_recall_data_from_markups(self,df_true, df_pred,iuo_th,ovp_th=None,ovd_th=None):

        df_true, df_pred = self.calc_hits_stats_iou(df_true, df_pred, iuo_th,ovp_th, ovd_th)
        file_true, coord_true, conf_true, hit_true = df_true.iloc[:,0].values, df_true.iloc[:,1:5],df_true.iloc[:,-2].values,df_true.iloc[:,-1].values
        file_pred, coord_pred, conf_pred, hit_pred = df_pred.iloc[:,0].values, df_pred.iloc[:,1:5],df_pred.iloc[:,-2].values,df_pred.iloc[:, -1].values

        relevant_file = numpy.array([each in file_true for each in file_pred])

        ths = {k: 0 for k in sorted(conf_true)}
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
        precision = (numpy.array(precision)[idx])
        recall = (numpy.array(recall)[idx])
        conf = (numpy.array(conf)[idx])

        #df_true.to_csv(self.folder_out+'df_true.csv',index=False,sep=' ')
        return precision,recall,conf
    # ----------------------------------------------------------------------------------------------------------------------
    def plot_precision_recall(self,precision,recall,filename_out=None,iuo_th=0.5):

        AP = numpy.trapz(precision, x=recall)
        lw = 2
        plt.plot(recall, precision, color='darkgreen', lw=lw)
        plt.grid(which='major', color='lightgray', linestyle='--')
        plt.minorticks_on()
        plt.grid(which='minor', axis='both', color='lightgray', linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(['AP @ IoU %1.2f = %0.2f' % (iuo_th, AP)])

        if filename_out is not None:
            plt.savefig(filename_out)

        plt.clf()

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_boxes_GT_pred(self,df_true, df_pred, metric='xxx', confidence=0.10, iou_th=0.1, ovp_th=None, ovd_th=None):

        if metric == 'recall':
            tools_IO.remove_files(self.folder_out+'miss/',create=True)
            tools_IO.remove_files(self.folder_out+'hit/', create=True)
        else:
            tools_IO.remove_files(self.folder_out + 'TP/', create=True)
            tools_IO.remove_files(self.folder_out + 'FP/', create=True)

        df_true, df_pred = self.calc_hits_stats_iou(df_true, df_pred,iuo_th=iou_th,ovp_th=ovp_th,ovd_th=ovd_th)
        file_true, coord_true, conf_true, hit_true = df_true.iloc[:,0].values, df_true.iloc[:,1:5].values,df_true.iloc[:,-2].values,df_true.iloc[:,-1].values
        file_pred, coord_pred, conf_pred, hit_pred = df_pred.iloc[:,0].values, df_pred.iloc[:,1:5].values,df_pred.iloc[:,-2].values,df_pred.iloc[:, -1].values

        if metric == 'recall':

            for b,filename in enumerate(numpy.unique(file_true)):

                image = cv2.imread(self.folder_images + filename)
                if image is None:continue

                image = tools_image.desaturate(image)
                is_hit=0
                idx = numpy.where(file_true==filename)
                for coord, hit, conf in zip(coord_true[idx],hit_true[idx],conf_true[idx]):
                    if conf<confidence:
                        hit = 0
                    image = tools_draw_numpy.draw_rect(image, coord[0], coord[1],coord[2], coord[3] ,color=self.hit_colors_true[int(hit)], w=2, alpha_transp=0.25)
                    is_hit = max(is_hit,hit)

                for coord in coord_pred[numpy.where(file_pred == filename)]:
                    image = tools_draw_numpy.draw_rect(image, coord[0], coord[1], coord[2], coord[3],color=(128,128,128), w=1,alpha_transp=1)

                cv2.imwrite(self.folder_out+('hit' if is_hit else 'miss')+'/'+filename.split('/')[-1],image)
        else:
            for b, filename in enumerate(numpy.unique(file_true)):
                image = cv2.imread(self.folder_images + filename)
                if image is None: continue
                image = tools_image.desaturate(image)
                is_FP = 1
                idx = numpy.where(file_pred == filename)

                for coord, hit, conf in zip(coord_pred[idx],hit_pred[idx],conf_pred[idx]):
                    if conf<confidence:
                        continue
                    image = tools_draw_numpy.draw_rect(image, coord[0], coord[1], coord[2], coord[3],color=self.hit_colors_pred[int(hit)], w=2, alpha_transp=0.25)
                    is_FP = min(is_FP, 1-hit)

                for coord in coord_true[numpy.where(file_true == filename)]:
                    image = tools_draw_numpy.draw_rect(image, coord[0], coord[1], coord[2], coord[3],color=(128,128,128), w=1,alpha_transp=1)
                cv2.imwrite(self.folder_out + ('FP' if is_FP else 'TP') + '/' + filename.split('/')[-1], image)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_boxes_GT_pred_stacked(self, df_true, df_pred, confidence=0.10, iou_th=0.1, ovp_th=None, ovd_th=None):
        df_true, df_pred = self.calc_hits_stats_iou(df_true, df_pred,iuo_th=iou_th,ovp_th=ovp_th,ovd_th=ovd_th)
        file_true, coord_true, conf_true, hit_true = df_true.iloc[:,0].values, df_true.iloc[:,1:5].values,df_true.iloc[:,-2].values,df_true.iloc[:,-1].values
        file_pred, coord_pred, conf_pred, hit_pred = df_pred.iloc[:,0].values, df_pred.iloc[:,1:5].values,df_pred.iloc[:,-2].values,df_pred.iloc[:, -1].values
        for b, filename in enumerate(numpy.unique(file_true)):
            image0 = cv2.imread(self.folder_images + filename)
            if image0 is None: continue
            image_fact = tools_image.desaturate(image0)
            image_pred  = tools_image.desaturate(image0)
            is_hit,is_FP = 0,1
            idx_fact = numpy.where(file_true == filename)
            idx_pred = numpy.where(file_pred == filename)

            for coord, hit, conf in zip(coord_true[idx_fact], hit_true[idx_fact], conf_true[idx_fact]):
                if conf < confidence:hit = 0
                image_fact = tools_draw_numpy.draw_rect(image_fact, coord[0], coord[1], coord[2], coord[3],color=self.hit_colors_true[int(hit)], w=2, alpha_transp=0.25)
                is_hit = max(is_hit, hit)

            for coord in coord_pred[numpy.where(file_pred == filename)]:
                image_fact = tools_draw_numpy.draw_rect(image_fact, coord[0], coord[1], coord[2], coord[3], color=(128, 128, 128),w=1, alpha_transp=1)

            for coord, hit, conf in zip(coord_pred[idx_pred], hit_pred[idx_pred], conf_pred[idx_pred]):
                if conf < confidence:continue
                image_pred = tools_draw_numpy.draw_rect(image_pred, coord[0], coord[1], coord[2], coord[3],color=self.hit_colors_pred[int(hit)], w=2, alpha_transp=0.25)
                is_FP = min(is_FP, 1 - hit)

            for coord in coord_true[numpy.where(file_true == filename)]:
                image_pred = tools_draw_numpy.draw_rect(image_pred, coord[0], coord[1], coord[2], coord[3], color=(128, 128, 128),w=1, alpha_transp=1)

            cv2.imwrite(self.folder_out + filename.split('/')[-1], numpy.concatenate((image_fact, image_pred), axis=1))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_contours_coco(self,filaname_coco_annnotation,alpha=0.75):
        coco = COCO(filaname_coco_annnotation)
        colors = tools_draw_numpy.get_colors(1 + len(coco.cats))
        class_names = [coco.cats[key]['name'] for key in coco.cats]

        for key in coco.imgToAnns.keys():
            annotations = coco.imgToAnns[key]
            image_id = annotations[0]['image_id']
            filename = coco.imgs[image_id]['file_name']
            if not os.path.isfile(self.folder_images + filename):
                continue

            result = tools_image.desaturate(cv2.imread(self.folder_images + filename))

            for annotation in annotations:
                contour = numpy.array(annotation['segmentation']).reshape((-1,2))
                category_IDs = annotation['category_id']
                result = tools_draw_numpy.draw_contours_cv(result, contour, colors[category_IDs],w=2,transperency=alpha)
                #result = tools_draw_numpy.draw_contours_cv(result, contour, colors[category_IDs],w=2,transperency=1.0)

            cv2.imwrite(self.folder_out + filename, result)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_boxes_coco(self,filaname_coco_annnotation,draw_binary_masks=False):
        coco = COCO(filaname_coco_annnotation)
        colors = tools_draw_numpy.get_colors(1 + len(coco.cats))
        class_names = [coco.cats[key]['name'] for key in coco.cats]

        for key in coco.imgToAnns.keys():

            annotations = coco.imgToAnns[key]
            image_id = annotations[0]['image_id']
            filename = coco.imgs[image_id]['file_name']
            if not os.path.isfile(self.folder_images + filename):
                continue

            boxes, category_IDs = [], []
            for annotation in annotations:
                bbox = annotation['bbox']
                boxes.append([bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]])
                category_IDs.append(annotation['category_id'])

            if draw_binary_masks:
                image = cv2.imread(self.folder_images + filename)
                result  = numpy.zeros_like(image)
                for box in boxes:
                    top, left, bottom, right = box
                    cv2.rectangle(result, (left, top), (right, bottom), (255,255,255), thickness=-1)
                    cv2.imwrite(self.folder_out + filename.split('.')[0]+'.jpg', image)
            else:
                image = tools_image.desaturate(cv2.imread(self.folder_images + filename))
                result = tools_draw_numpy.draw_bboxes(image, boxes, [1] * len(boxes),category_IDs, colors, class_names)

            cv2.imwrite(self.folder_out + filename, result)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_boxes_flat(self,filaname_in,folder_images,folder_out):
        tools_IO.remove_files(folder_out,'*.jpg,*.png')
        df = pd.read_csv(filaname_in,sep=' ')

        filenames = df.iloc[:,0].unique()
        N = df.iloc[:, 5].unique().shape[0]
        dct_color = dict(zip(df.iloc[:, 5].unique(),numpy.arange(N)))
        colors = tools_draw_numpy.get_colors(N=N)

        for filename in filenames:

            if not os.path.isfile(folder_images + filename):
                continue

            image = cv2.imread(folder_images + filename)
            image = tools_image.desaturate(image)
            df_temp = tools_DF.apply_filter(df, df.columns[0], filename)

            for r in range(df_temp.shape[0]):
                col_left, row_up, col_right, row_down = df_temp.iloc[r,1:5]
                cat_id = df_temp.iloc[r,5]
                image = tools_draw_numpy.draw_rect(image, col_left, row_up, col_right, row_down ,colors[dct_color[cat_id]], w=1, alpha_transp=0.8)

            cv2.imwrite(folder_out + filename, image)

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def coco_to_flat(self,filaname_coco_annnotation,filename_flat,list_attributes=None):

        coco = COCO(filaname_coco_annnotation)
        filanames, bboxes, cat_ID,attributes = [],[],[],[]

        for key in coco.imgToAnns.keys():
            annotations = coco.imgToAnns[key]
            image_id = annotations[0]['image_id']
            filename = coco.imgs[image_id]['file_name']

            for annotation in annotations:
                filanames.append(filename)
                cat_ID.append(annotation['category_id'])
                bbox = annotation['bbox']
                #bboxes.append([bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]])
                bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2],bbox[1] + bbox[3]])

                if list_attributes is not None:
                    attributes+=[(annotation['attributes'][a]) for a in list_attributes if a in annotation['attributes']]

        bboxes = numpy.array(bboxes)
        df = pd.DataFrame({'filaname':filanames,'left':bboxes[:,0],'top':bboxes[:,1],'right':bboxes[:,2],'bottom':bboxes[:,3],'class_ID':cat_ID})

        if list_attributes is not None:
            attributes = numpy.array(attributes).reshape((df.shape[0],-1))
            df = pd.concat([df,pd.DataFrame(attributes)],axis=1)
            df  = df.rename(columns=dict(zip(df.columns[-len(list_attributes):], list_attributes)))

        df.to_csv(filename_flat,index=False,sep=' ')

        return
    # ----------------------------------------------------------------------------------------------------------------------