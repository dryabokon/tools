import pandas as pd
import cv2
import os
import numpy
import torch
from ultralytics import YOLO
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Detector_yolo:
    def __init__(self,folder_out,config=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = config.model_detect if (config is not None and config.model_detect is not None) else 'yolov8n.pt'
        device_colored = ('\033[92m' + 'cuda' + '\033[0m' if self.device == 'cuda' else '\033[33m' + 'CPU' + '\033[0m')
        print('[Detectr] device:',device_colored,'-',model_name)

        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out


        self.model_detect = YOLO(model_name)
        self.model_detect.to(self.device)
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        self.dct_class_names = None
        return
# ----------------------------------------------------------------------------------------------------------------------
    def update_config(self, config):
        self.config = config
        model_name = config.model_detect if (config is not None and config.model_detect is not None) else 'yolo11n.pt'
        print('Using model:', model_name)
        self.model_detect = YOLO(model_name)
        self.model_detect.to(self.device)
        return
    # ----------------------------------------------------------------------------------------------------------------------

    def get_detections(self,filename_in, col_start=None,do_debug=False):
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in
        class_names = []
        res = self.model_detect.predict(source=image, verbose=False, device=self.device)
        if res[0].boxes is not None:
            rects = res[0].boxes.xyxy.cpu().numpy()
            class_ids = res[0].boxes.cls.cpu().numpy()
            self.dct_class_names = res[0].names
            confs = res[0].boxes.conf.cpu().numpy()
            class_names = numpy.array([self.dct_class_names[i] for i in class_ids])

        if do_debug and isinstance(filename_in, str):
            image_res = tools_image.desaturate(image)
            image_res = self.draw_detections(image_res,rects,class_ids, confs)
            cv2.imwrite(self.folder_out + filename_in.split('/')[-1], image_res)

        df_pred = pd.DataFrame(numpy.concatenate((class_ids.reshape((-1, 1)),rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['class_ids',             'x1', 'y1', 'x2', 'y2', 'conf'])
        df_pred = df_pred.astype({'class_ids': int, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'conf': float})
        df_pred['class_name'] = class_names

        return df_pred
# ----------------------------------------------------------------------------------------------------------------------
    def draw_detections(self,image,rects,class_ids,confs):
        colors = [self.colors80[i % 80] for i in range(len(rects))]
        labels = [self.dct_class_names[i] + ' %.2f' % conf for i, conf in zip(class_ids, confs)]
        image = tools_draw_numpy.draw_rects(image, rects.reshape(-1, 2, 2), colors, labels=labels, w=2,alpha_transp=0.8)
        return  image
# ----------------------------------------------------------------------------------------------------------------------