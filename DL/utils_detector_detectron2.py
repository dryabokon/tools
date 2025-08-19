import numpy
import pandas as pd
import cv2
import os
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Detector_detectron2:
    def __init__(self,folder_out,config=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_th = config.confidence_th
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out

        #config_file = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
        config_file = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # Threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        cfg.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(cfg)
        self.device = device
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        self.config = config
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update_confidence_th(self,confidence_th):
        self.confidence_th = confidence_th
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update_config(self, config):
        self.config = config
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def get_detections(self,filename_in, do_debug=False):
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in

        detections = self.predictor(image)
        confs = detections['instances'].scores.cpu().numpy()
        rects = detections['instances'].pred_boxes.tensor.cpu().numpy()
        class_ids = detections['instances'].pred_classes.cpu().numpy()
        dct_class_names = self.predictor.metadata.thing_classes
        class_names = numpy.array([dct_class_names[i] for i in class_ids])

        if do_debug and isinstance(filename_in, str):
            image_res = tools_image.desaturate(image)
            image_res = self.draw_detections(image_res,rects,class_ids, confs,dct_class_names)
            cv2.imwrite(self.folder_out + filename_in.split('/')[-1], image_res)

        df_pred = pd.DataFrame(numpy.concatenate((class_ids.reshape((-1, 1)), rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['class_ids', 'x1', 'y1', 'x2', 'y2', 'conf'])
        df_pred = df_pred.astype({'class_ids': int, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'conf': float})
        df_pred['class_name'] = class_names

        return df_pred
# ----------------------------------------------------------------------------------------------------------------------
    def draw_detections(self,image,rects,class_ids,confs,dct_class_names):
        colors = [self.colors80[i % 80] for i in range(len(rects))]
        labels = [dct_class_names[i] + '%.2f' % conf for i, conf in zip(class_ids, confs)]
        image = tools_draw_numpy.draw_rects(image, rects.reshape(-1, 2, 2), colors, labels=labels, w=2,alpha_transp=0.8)
        return  image
# ----------------------------------------------------------------------------------------------------------------------