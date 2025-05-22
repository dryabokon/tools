#----------------------------------------------------------------------------------------------------------------------
import cv2
import os
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
class Tracker_KCF:
    def __init__(self,folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.is_initialized = False
        #self.tracker = cv2.TrackerKCF.create()
        self.tracker = cv2.TrackerCSRT.create()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def track_detections(self,ROI,filename_in,frame_id=None,do_debug=True):

        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in

        if not self.is_initialized:
            p1 = (ROI[0], ROI[1])
            p2 = (ROI[2], ROI[3])
            ROI[2]-= ROI[0]
            ROI[3]-= ROI[1]
            self.tracker.init(image, ROI)
            self.is_initialized = True
        else:
            ok, bbox = self.tracker.update(image)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        df_pred = pd.DataFrame({'track_id': [0], 'x1': [p1[0]], 'y1': [p1[1]], 'x2': [p2[0]], 'y2': [p2[1]], 'conf': [1]})

        return df_pred
# ----------------------------------------------------------------------------------------------------------------------


