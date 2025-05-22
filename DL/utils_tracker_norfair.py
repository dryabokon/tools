import numpy
import pandas as pd
import cv2
import os
from norfair import Detection, Tracker
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Tracker_norfair:
    def __init__(self,folder_out,device='cuda'):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out

        self.tracker = Tracker(distance_function="euclidean", distance_threshold=20,reid_distance_function=self.embedding_distance,reid_distance_threshold=0.5,reid_hit_counter_max=500)

        self.device = device
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def embedding_distance(self,matched_not_init_trackers, unmatched_trackers):
        snd_embedding = unmatched_trackers.last_detection.embedding

        if snd_embedding is None:
            for detection in reversed(unmatched_trackers.past_detections):
                if detection.embedding is not None:
                    snd_embedding = detection.embedding
                    break
            else:
                return 1

        for detection_fst in matched_not_init_trackers.past_detections:
            if detection_fst.embedding is None:
                continue

            distance = 1 - cv2.compareHist(snd_embedding.astype(numpy.float32), detection_fst.embedding.astype(numpy.float32), cv2.HISTCMP_CORREL)
            if distance < 0.5:
                return distance
        return 1
# ----------------------------------------------------------------------------------------------------------------------
    def derive(self,rects,centers):

        cr = numpy.concatenate([0.5 * (rects[:, 0] + rects[:, 2]).reshape((-1, 1)), 0.5 * (rects[:, 1] + rects[:, 3]).reshape((-1, 1))],axis=1)
        tol = 16
        idx_rect,idx_centers = [],[]
        for n,c in enumerate(centers):
            S = numpy.sum((cr - c) ** 2, axis=1)
            i = numpy.argmin(S)
            if (S[i] < tol):
                idx_rect.append(i)
                idx_centers.append(n)

        return idx_rect,idx_centers
# ----------------------------------------------------------------------------------------------------------------------
    def track_detections(self, df_det,filename_in=None,frame_id=None,do_debug=False):

        rects = df_det[['x1', 'y1', 'x2', 'y2']].values
        centers = 0.5 * (rects[:, 0:2] + rects[:, 2:4])
        confs = df_det['conf'].values

        norfair_detections = [Detection(c) for c in centers]

        # if df_det.shape[1]>col_start:
        #     features = df_det.iloc[:,col_start:].values
        #     for d,feature in zip(norfair_detections,features):
        #         d.embedding = feature

        tracked_objects = self.tracker.update(detections=norfair_detections)

        idx_rects, idx_centers = self.derive(rects, [o.estimate for o in tracked_objects])
        rects = rects[idx_rects].reshape((-1, 2, 2))
        confs = confs[idx_rects]
        track_ids = numpy.array([o.global_id for o in tracked_objects])[idx_centers]

        if do_debug:
            if isinstance(filename_in, str):
                image = cv2.imread(filename_in)
                filename_out = (filename_in.split('/')[-1]).split('.')[0] + '.jpg'
            else:
                image = filename_in
                filename_out = 'frame_%06d'%frame_id + '.jpg'

            cv2.imwrite(self.folder_out + filename_out, self.draw_tracks(image,rects,track_ids))

        df_track = pd.DataFrame(numpy.concatenate((track_ids.reshape((-1, 1)), rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['track_id', 'x1', 'y1', 'x2', 'y2', 'conf'])

        return df_track
# ----------------------------------------------------------------------------------------------------------------------
    def draw_tracks(self,image,rects,track_ids):
        colors = [self.colors80[track_id % 80] for track_id in track_ids]
        image = tools_draw_numpy.draw_rects(tools_image.desaturate(image), rects, colors, labels=track_ids.astype(str), w=2,alpha_transp=0.8)
        return image
# ----------------------------------------------------------------------------------------------------------------------