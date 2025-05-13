import cv2
import os
import torch
from ultralytics import YOLO
# ----------------------------------------------------------------------------------------------------------------------
class Classifier_yolo:
    def __init__(self,folder_out,config=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', device)

        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.device = device
        self.model_classify = YOLO(config.model_classify if config.model_classify is not None else 'yolov8n-cls.pt')
        self.model_classify.to(device)

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
        confidence = res[0].probs.data.cpu().numpy()[1]


        #confidence = res[0].probs.to(self.device).cpu().numpy()
        #class_id = numpy.argmax(confidence)
        #class_name = self.model_classify.names[class_id]

        return confidence
# ----------------------------------------------------------------------------------------------------------------------
