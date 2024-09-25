import numpy
import cv2
import os
import pandas as pd
from tqdm import tqdm
import inspect
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Benchmarker(object):

    def __init__(self,folder_out):
        self.folder_out=folder_out
        #self.folder_images = folder_images
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
    def calc_hits_stats_iou0(self,df_true, df_pred, iuo_th=0.5,ovp_th=None,ovd_th=None):

        hit_true = numpy.zeros(df_true.shape[0])
        hit_pred = numpy.zeros(df_pred.shape[0])
        file_true = df_true.iloc[:,0].values
        file_pred = df_pred.iloc[:,0].values
        conf_true = numpy.full(df_true.shape[0],-1.0)
        conf_pred = df_pred['conf'].values
        track_id_pred = df_pred['track_id'].values
        hit_track_pred = numpy.full(df_true.shape[0],-1)

        coord_true = df_true[['x1','y1','x2','y2']].values
        coord_pred = df_pred[['x1','y1','x2','y2']].values

        if iuo_th is not None:
            for j, filename_true in enumerate(file_true):
                best_i, best_iou, best_conf = None, -1, None
                for i, filename_pred in enumerate(file_pred):
                    if filename_true == filename_pred:
                        iuo_value = self.iou(coord_true[j], coord_pred[i])
                        if iuo_value >= iuo_th and iuo_value > best_iou:
                            best_i, best_iou, best_conf, best_pred_id = i, iuo_value, conf_pred[i], track_id_pred[i]
                if best_i is not None:
                    hit_true[j], conf_true[j],hit_track_pred[j] = 1, best_conf,best_pred_id
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
                            hit_track_pred[j] = track_id_pred[i]
                            conf_true[j] = max(conf_true[j],conf_pred[i])

        df_true2 = df_true.copy()
        df_pred2 = df_pred.copy()
        df_true2['conf'] = numpy.array(conf_true)
        df_true2['hit'] = numpy.array(hit_true)
        df_true2['track_id_pred'] = numpy.array(hit_track_pred)
        df_pred2['hit'] = numpy.array(hit_pred)
        return df_true2,df_pred2
    # ----------------------------------------------------------------------------------------------------------------------
    def calc_hits_stats_iou(self, df_true0, df_pred0, iou_th=0.5, from_cache=True,verbose=True):

        if from_cache and os.path.exists(self.folder_out+'df_true2.csv') and os.path.exists(self.folder_out+'df_pred2.csv'):
            df_true = pd.read_csv(self.folder_out+'df_true2.csv')
            df_pred = pd.read_csv(self.folder_out+'df_track2.csv')
            return df_true,df_pred

        df_true = df_true0.copy()
        df_pred = df_pred0.copy()
        df_true['pred_row'] = -1
        df_pred['true_row'] = -1

        jj = tqdm(range(df_true.shape[0]), desc=inspect.currentframe().f_code.co_name) if verbose else range(df_true.shape[0])
        for j in jj:
            best_i, best_iou  = None,-1
            boxA = df_true[['x1', 'y1', 'x2', 'y2']].iloc[j].values
            for i in numpy.where(df_true['frame_id'].iloc[j] == df_pred['frame_id'])[0]:
                boxB = df_pred[['x1', 'y1', 'x2', 'y2']].iloc[i].values
                iuo_value = self.iou(boxA,boxB)
                if (iuo_value >= iou_th) and (iuo_value > best_iou):
                    best_i = i
                    best_iou = iuo_value

            if best_i is not None:
                # df_true['pred_row'].iloc[j] = best_i
                # df_pred['true_row'].iloc[best_i] = j
                df_true.iloc[j,df_true.columns.get_loc('pred_row')] = best_i
                df_pred.iloc[best_i,df_pred.columns.get_loc('true_row')] = j


        df_true['row_id'] = numpy.arange(df_true.shape[0])
        df_pred['row_id'] = numpy.arange(df_pred.shape[0])
        df_true = tools_DF.fetch(df_true, 'pred_row', df_pred, 'row_id', ['conf', 'track_id'],col_new_name=['conf_pred', 'track_id_pred'])
        df_pred_track_id_top = tools_DF.my_agg(df_true, ['track_id'], ['track_id_pred'], aggs=['top'],list_res_names=['track_id_pred_top'])
        df_true = tools_DF.fetch(df_true, 'track_id', df_pred_track_id_top, 'track_id', ['track_id_pred_top'],col_new_name=['track_id_pred_top'])
        df_true['IDTP'] = (df_true['track_id_pred'] == df_true['track_id_pred_top'])

        df_pred = tools_DF.fetch(df_pred, 'true_row', df_true, 'row_id', ['track_id'],col_new_name=['track_id_true'])
        df_pred_track_id_top = tools_DF.my_agg(df_pred, ['track_id_true'], ['track_id'], aggs=['top'],list_res_names=['track_id_top'])
        df_pred = tools_DF.fetch(df_pred, 'track_id_true', df_pred_track_id_top, 'track_id_true', ['track_id_top'],col_new_name=['track_id_top'])
        df_pred['IDTP'] = (df_pred['track_id'] == df_pred['track_id_top']) & (df_pred['true_row']>=0)

        if verbose:
            df_true.to_csv(self.folder_out+'df_true2.csv',index=False)
            df_pred.to_csv(self.folder_out+'df_track2.csv', index=False)

        return df_true,df_pred
    # ----------------------------------------------------------------------------------------------------------------------
    def get_precsion_recall_data_from_markups(self, df_true, df_pred, iou_th,use_IDTP=False):

        df_true2, df_pred2 = self.calc_hits_stats_iou(df_true, df_pred, iou_th)
        conf_pred = df_pred2['conf'].values
        relevant_file = numpy.array([(each in df_true2.iloc[:,0].values) for each in df_pred2.iloc[:,0].values])
        hit_true = ~numpy.isnan(df_true2['conf_pred'].values)
        hit_true_IDTP = df_true2['IDTP'].values
        conf_true = df_true2['conf_pred'].values
        conf_true = conf_true[~numpy.isnan(conf_true)]
        hit_pred = df_pred2['true_row'].values>=0
        hit_pred_IDTP = df_pred2['IDTP'].values

        ths = {k: 0 for k in sorted(conf_true)}

        precision, recall, F1s, conf = [],[],[],[]
        n_TP,n_FP,n_FN = [],[],[]

        for cnt,th in enumerate(ths):
            iii = numpy.where(conf_true>=th)
            if use_IDTP:
                recall_TP = numpy.sum(hit_true_IDTP[iii])
                FN = len(hit_true_IDTP) - recall_TP
            else:
                recall_TP = numpy.sum(hit_true[iii])
                FN = len(hit_true)-recall_TP


            recall.append(float(recall_TP/(recall_TP+FN)))

            idx = numpy.where(relevant_file == True)
            if use_IDTP:
                pred_TP = numpy.sum(hit_pred_IDTP[conf_pred >= th])
                FP = len(hit_pred_IDTP[idx][conf_pred[idx] >= th]) - pred_TP
            else:
                pred_TP = numpy.sum(hit_pred[conf_pred>=th])
                FP = len(hit_pred[idx][conf_pred[idx]>=th])-pred_TP

            if pred_TP + FP>0:
                precision.append(float(pred_TP / (pred_TP + FP)))
            else:
                precision.append(0)

            n_FN.append(FN)
            n_FP.append(FP)
            n_TP.append(pred_TP)
            F1s.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1]))
            conf.append(th)

        # idx_best = numpy.argmax(F1s)
        # best_fp = FPs[idx_best]
        # best_misses = FNs[idx_best]
        # best_f1 = F1s[idx_best]

        #df_true.to_csv(self.folder_out+'df_true.csv',index=False,sep=' ')
        return n_TP,n_FP,n_FN,precision,recall,F1s,conf
# ----------------------------------------------------------------------------------------------------------------------
