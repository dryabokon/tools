import inspect
import numpy
import pandas as pd
import cv2
import os
# ----------------------------------------------------------------------------------------------------------------------
from boxmot import BotSort, ByteTrack, OcSort, DeepOcSort
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Tracker_boxmot:
    def __init__(self,folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out

        algorithm = 'BYTE'
        if algorithm == 'OCSORT':
            self.tracker = OcSort()
        elif algorithm == 'BYTE':
            self.tracker = ByteTrack()
        elif algorithm == 'BOTSORT':
            self.tracker = BotSort(model_weights=Path('osnet_x0_25_msmt17.pt'), device='cuda:0', fp16=False)
        else:
            self.tracker = None

        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)


        return
# ----------------------------------------------------------------------------------------------------------------------
    def track_detections(self,df_det,filename_in,frame_id=None,do_debug=False):

        if 'class_ids' not in df_det.columns:
            df_det['class_ids'] = -1

        dets = df_det[['x1', 'y1', 'x2', 'y2','conf','class_ids']].values
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in
        embs = None
        # if df_det.shape[1]>col_start:
        #     embs = df_det.iloc[:,col_start:].values
        self.tracker.update(dets, image,embs)

        rects = numpy.array([a.history_observations[-1][:4] for a in self.tracker.active_tracks if a.history_observations and (len(a.history_observations) > 2)])
        confs = numpy.array([a.conf for a in self.tracker.active_tracks if a.history_observations and (len(a.history_observations) > 2)]).astype(float)
        track_ids = numpy.array([a.id for a in self.tracker.active_tracks if a.history_observations and (len(a.history_observations) > 2)])

        if do_debug:
            if isinstance(filename_in, str):
                image = cv2.imread(filename_in)
                filename_out = (filename_in.split('/')[-1]).split('.')[0] + '.jpg'
            else:
                image = filename_in
                filename_out = 'frame_%06d'%frame_id + '.jpg'

            cv2.imwrite(self.folder_out + filename_out,self.draw_tracks(image, rects.reshape((-1, 2, 2)), track_ids))

        df_track = pd.DataFrame(numpy.concatenate((track_ids.reshape((-1, 1)), rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['track_id', 'x1', 'y1', 'x2', 'y2', 'conf'])

        return df_track
# ----------------------------------------------------------------------------------------------------------------------
#     def draw_tracks(self,image,rects,track_ids):
#         colors = [self.colors80[track_id % 80] for track_id in track_ids]
#         image = tools_draw_numpy.draw_rects(tools_image.desaturate(image), rects, colors, labels=track_ids.astype(str), w=2,alpha_transp=0.8)
#         return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_tracks(self,image,rects,track_ids):
        colors = [self.colors80[track_id % 80] for track_id in track_ids]
        image = tools_image.desaturate(image)
        for rect,label,color in zip(rects,track_ids.astype(str),colors):
            col_left, row_up, col_right, row_down = rect.flatten()
            color_fg = (0, 0, 0) if 10 * color[0] + 60 * color[1] + 30 * color[2] > 100 * 128 else (255, 255, 255)
            image = tools_draw_numpy.draw_rect_fast(image, col_left, row_up, col_right, row_down ,color, w=2)
            image = tools_draw_numpy.draw_text_fast(image, label, (int(col_left), int(row_up)), color_fg=color_fg, clr_bg=color,fontScale=16)

        return image
# ----------------------------------------------------------------------------------------------------------------------