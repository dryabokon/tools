#----------------------------------------------------------------------------------------------------------------------
import cv2
import os
import pandas as pd

from SiamTrackers.NanoTrack.nanotrack.core.config import cfg
from SiamTrackers.NanoTrack.nanotrack.models.model_builder import ModelBuilder
from SiamTrackers.NanoTrack.nanotrack.tracker.tracker_builder import build_tracker
from SiamTrackers.NanoTrack.nanotrack.utils.model_load import load_pretrain


# ----------------------------------------------------------------------------------------------------------------------
class Tracker_Nano:
    def __init__(self,folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.is_initialized = False

        model = load_pretrain(ModelBuilder(),'./SiamTrackers/NanoTrack/models/pretrained/nanotrackv2.pth').cuda().eval()
        self.tracker = build_tracker(model)

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
            bbox = self.tracker.update(image)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        df_pred = pd.DataFrame({'track_id': [0], 'x1': [p1[0]], 'y1': [p1[1]], 'x2': [p2[0]], 'y2': [p2[1]], 'conf': [1]})

        return df_pred
# ----------------------------------------------------------------------------------------------------------------------


