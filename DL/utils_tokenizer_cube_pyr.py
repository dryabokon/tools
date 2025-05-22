import cv2
import numpy
import pandas as pd
from ultralytics import YOLO
# ----------------------------------------------------------------------------------------------------------------------

import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Tokenizer_Cube_Pyr:
    def __init__(self,folder_out,device='cuda'):
        #self.model_classify = YOLO('./models/marine/yolov8n-classify-cube-pyramid.pt')
        self.model_classify = YOLO('./models/marine/yolov8n-classify-4-classes.pt')
        self.device = device
        self.model_classify.to(device)
        self.dct_class_names = self.model_classify.names
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        self.folder_out = folder_out

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_features(self,filename_in,df_pred,frame_id=0,do_debug=False):
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in

        E = []
        for r in range(df_pred.shape[0]):
            rect = df_pred.iloc[r][['x1','y1','x2','y2']].values.astype(int)
            x1, y1, x2, y2 = rect
            H, W = image.shape[:2]
            image_crop = image[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))]
            res = self.model_classify.predict(source=image_crop, verbose=False, device=self.device)
            if res[0].probs is not None:
                obj_type = res[0].probs.top1
                obj_type = self.dct_class_names[obj_type]
                conf_obj_type = res[0].probs.top1conf.cpu().numpy()
            else:
                obj_type = None
                conf_obj_type = 0

            E.append([obj_type,conf_obj_type,1])

        df_E = pd.DataFrame(E, columns=['obj_type','conf_obj_type','one'])
        df_E = df_E.astype({'obj_type': str,'conf_obj_type': float})



        return df_E
# ----------------------------------------------------------------------------------------------------------------------