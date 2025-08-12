import numpy
import cv2
import os
import torch
from ultralytics import YOLO
# ----------------------------------------------------------------------------------------------------------------------
class Classifier_yolo:
    def __init__(self,folder_out,config=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = config.model_classify if (config is not None and config.model_classify is not None) else 'yolo11n-cls.pt'
        device_colored = ('\033[92m' + 'cuda' + '\033[0m' if self.device == 'cuda' else '\033[33m' + 'CPU' + '\033[0m')
        print('[Classfy] device:',device_colored,'-',model_name)
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.model_classify = YOLO(model_name)
        self.model_classify.to(self.device)

        self.dct_class_names = None
        return
# ----------------------------------------------------------------------------------------------------------------------
    def update_config(self, config):
        self.config = config
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def get_classification(self,filename_in):
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in
        res = self.model_classify.predict(source=image, verbose=False, device=self.device)

        self.dct_class_names = res[0].names
        confs = res[0].probs.data.cpu().numpy()
        class_id = confs.argmax()
        class_name = self.dct_class_names[class_id]
        return confs[class_id], class_name

# ----------------------------------------------------------------------------------------------------------------------
