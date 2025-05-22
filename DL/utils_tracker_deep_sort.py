import inspect
import numpy
import pandas as pd
import cv2
import os
# ----------------------------------------------------------------------------------------------------------------------
from .deep_sort import nn_matching
from .deep_sort.tracker import Tracker
from .deep_sort.detection import Detection
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy

# ----------------------------------------------------------------------------------------------------------------------
class Tracker_deep_sort:
    def __init__(self,folder_out,max_iou_distance=0.7, max_age=30, n_init=3):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold=0.2)
        self.tracker = Tracker(metric,max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def track_detections(self,df_det,filename_in,frame_id=None,do_debug=False):

        #do_debug = True
        rects0 = df_det[['x1', 'y1', 'x2', 'y2']].values
        tlwhs0 = [(r[0],r[1],r[2]-r[0],r[3]-r[1]) for r in rects0]
        confs0 = df_det['conf'].values
        features = [[] for i in range(len(tlwhs0))]
        tlwhs0 = numpy.array(tlwhs0).astype(int)

        self.tracker.predict()
        self.tracker.update([Detection(tlwh, confidence, feature) for tlwh, confidence,feature  in zip(tlwhs0, confs0,features)])

        track_ids = numpy.array([track.track_id for track in self.tracker.tracks if (track.is_confirmed() and track.time_since_update == 0)]).astype(int)
        rects = numpy.array([track.to_tlbr().astype(int) for track in self.tracker.tracks if ( track.is_confirmed() and track.time_since_update ==0)])
        confs = numpy.array([1]*len(track_ids))

        # rects_det = rects0.copy()
        # track_ids_all = numpy.array([track.track_id for track in self.tracker.tracks ]).astype(int)
        # rects_track     = numpy.array([track.to_tlbr().astype(int) for track in self.tracker.tracks])
        # is_confirmed = numpy.array([track.is_confirmed() for track in self.tracker.tracks])
        # time_upd     = numpy.array([track.time_since_update for track in self.tracker.tracks])

        #cv2.imwrite(self.folder_out + 'frame_%06d'%frame_id + '.jpg',self.draw_tracks_debug(cv2.imread(filename_in),rects_det,rects_track,track_ids_all,is_confirmed,time_upd))

        if do_debug:
            if isinstance(filename_in, str):
                image = cv2.imread(filename_in)
                filename_out = (filename_in.split('/')[-1]).split('.')[0] + '.jpg'
            else:
                image = filename_in
                filename_out = 'frame_%06d'%frame_id + '.jpg'

            cv2.imwrite(self.folder_out + filename_out, self.draw_tracks(image, rects.reshape((-1,2,2)), track_ids))

        df_pred = pd.DataFrame(numpy.concatenate((track_ids.reshape((-1, 1)), rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['track_id', 'x1', 'y1', 'x2', 'y2', 'conf'])

        return df_pred
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
    def draw_tracks_debug(self,image,rects_det,rects_track,track_ids_all,is_confirmed,time_upd,tol=15):

        image = tools_draw_numpy.draw_rects(image, rects_det.reshape((-1,2,2)), colors=(0,0,0),w=1,alpha_transp=1)
        colors = [(self.colors80[tid%80] if (t==0 and c) else (0,0,0)) for c,t,tid in zip(is_confirmed, time_upd,track_ids_all)]
        labels = [str(tid) if (t==0 and c) else 'NC: '+str(tid) for c, t, tid in zip(is_confirmed, time_upd, track_ids_all)]

        rects_track=rects_track.reshape((-1,4))
        rects_track = numpy.array([(r[0]+tol,r[1]+tol,r[2]-tol,r[3]-tol) for r in rects_track])
        image = tools_draw_numpy.draw_rects(image, rects_track.reshape((-1,2,2)), w=5,colors=colors, alpha_transp=1,labels=labels)

        return image
# ----------------------------------------------------------------------------------------------------------------------
