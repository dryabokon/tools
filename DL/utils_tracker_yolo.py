import pandas as pd
import cv2
import os
import numpy
from ultralytics import YOLO
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Tracker_yolo:
    def __init__(self,folder_out,device='cuda'):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.tracker_cfg = "bytetrack.yaml"
        #self.tracker_cfg = "botsort.yaml"
        self.device = device

        self.model_detect = YOLO('yolov8n.pt')
        self.model_detect.to(device)
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def track_detections(self,df_det,filename_in,frame_id=None,do_debug=False):
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in

        res = self.model_detect.track(source=image, persist=True, tracker=self.tracker_cfg, verbose=False, device=self.device)[0]
        track_ids = numpy.array([b.id for b in res.boxes]).flatten()
        track_ids = numpy.array([(t if t is not None else -1) for t in track_ids]).flatten().astype(int)
        rects = numpy.array([b.xyxy.cpu().numpy().reshape((-1, 2, 2)) for b in res.boxes]).reshape((-1, 2, 2))

        confs = res.boxes.conf.cpu().numpy().flatten()
        class_ids = res.boxes.cls.cpu().numpy()

        if do_debug:
            if isinstance(filename_in, str):
                image = cv2.imread(filename_in)
                filename_out = (filename_in.split('/')[-1]).split('.')[0] + '.jpg'
            else:
                image = filename_in
                filename_out = 'frame_%06d'%frame_id + '.jpg'

            cv2.imwrite(self.folder_out + filename_out, self.draw_tracks(image, rects, track_ids))

        df_track = pd.DataFrame(
            numpy.concatenate((track_ids.reshape((-1, 1)), class_ids.reshape((-1, 1)),rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['track_id', 'class_ids','x1', 'y1', 'x2', 'y2', 'conf'])
        df_track = df_track.astype({'track_id': int,'class_ids': int, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'conf': float})
        return df_track
# ----------------------------------------------------------------------------------------------------------------------
    def draw_tracks(self,image,rects,track_ids):
        colors = [self.colors80[(track_id % 80)] for track_id in track_ids]
        image = tools_draw_numpy.draw_rects(tools_image.desaturate(image), rects, colors, labels=track_ids.astype(str), w=2,alpha_transp=0.8)
        # R = 50
        # cx = rects.reshape((-1, 4))[:,[0,2]].mean(axis=1).reshape((-1, 1))
        # cy = rects.reshape((-1, 4))[:,[1,3]].mean(axis=1).reshape((-1, 1))
        # ellipsis = [([c[1],c[0]],[R,R],0) for c in numpy.concatenate((cy,cx),axis=1)]
        # image = tools_draw_numpy.draw_ellipses(tools_image.desaturate(image,level=0.8), ellipsis,color=colors ,labels=track_ids.astype(str),w=-1,transperency=0.8)
        return image
# ----------------------------------------------------------------------------------------------------------------------