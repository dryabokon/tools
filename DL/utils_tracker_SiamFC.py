#----------------------------------------------------------------------------------------------------------------------
import cv2
import os
import pandas as pd
from .siamfc_pytorch.siamfc.siamfc import TrackerSiamFC
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
class Tracker_SiamFC:
    def __init__(self,folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.is_initialized = False
        self.buffer_is_tracked = [False]*10
        self.ROI = None

        self.tracker = TrackerSiamFC(net_path=os.path.join(os.path.dirname(__file__), 'siamfc_pytorch/siamfc_alexnet_e50.pth'))
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        self.colors80[0] = (0, 0, 255)
        self.colors80[1] = (128, 0, 0)
        self.colors80[2] = (0, 128, 0)

        self.TP = tools_time_profiler.Time_Profiler(verbose=False)
        print('TrackerSiamFC@'+('GPU' if self.tracker.cuda else 'CPU') + ' is initialized')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.is_initialized = False
        self.ROI = None
        return
# ----------------------------------------------------------------------------------------------------------------------
    def do_init(self,image,ROI):
        self.tracker.init(image, [ROI[0], ROI[1], ROI[2] - ROI[0], ROI[3] - ROI[1]])
        self.is_initialized = True
        self.ROI = ROI
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_tracks(self, image, rects, track_ids, labels=None):

        colors = [self.colors80[track_id % 80] for track_id in track_ids]
        if labels is None: labels = [None] * len(track_ids)
        for rect, track_id, color, label in zip(rects, track_ids.astype(str), colors, labels):
            col_left, row_up, col_right, row_down = rect.flatten()
            image = tools_draw_numpy.draw_rect_fast(image, col_left, row_up, col_right, row_down, color, w=2,label=None,alpha_transp=0)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def track_detections(self,ROI,filename_in,frame_id):
        self.TP.tic('track_detections')
        df_pred = pd.DataFrame({'track_id': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'conf': []})
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in

        if (not self.is_initialized):
            if ROI is not None:
                self.do_init(image,ROI)
        else:
            #bbox,loss = self.tracker.update(image)
            res = self.tracker.update(image)
            if len(res) == 2:
                bbox, loss = res
            else:
                bbox, loss = res, 0
            self.ROI = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
            df_pred = pd.DataFrame({'track_id': [0], 'x1': [self.ROI[0]], 'y1': [self.ROI[1]], 'x2': [self.ROI[2]], 'y2': [self.ROI[3]], 'conf': ['%.2f' % loss]})

        self.TP.tic('track_detections')

        return df_pred
# ----------------------------------------------------------------------------------------------------------------------

