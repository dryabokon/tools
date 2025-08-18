import time
import inspect
import os
import hashlib
import pandas as pd
import numpy
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_image
import tools_draw_numpy
import tools_heartbeat
import tools_IO
import tools_mAP_visualizer
import tools_animation
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
from DL import utils_detector_yolo
from DL import utils_classifier_yolo
from DL import utils_tracker_deep_sort
from DL import utils_tracker_yolo
# ----------------------------------------------------------------------------------------------------------------------
class Pipeliner:
    def __init__(self,folder_out,config,Grabber,save_tracks=True):

        self.folder_out = folder_out
        self.config = config

        self.df_true = None
        self.df_pred = None

        self.Detector = None
        self.Tracker = None
        self.Classifier = None
        self.Grabber = None
        self.update_grabber(Grabber)
        self.update_config(self.config)
        self.save_tracks = save_tracks


        self.HB = tools_heartbeat.tools_HB()
        self.TP = tools_time_profiler.Time_Profiler(verbose=False)
        #self.MLFlower = self.init_MLflow()
        self.V = tools_mAP_visualizer.Track_Visualizer(self.folder_out, stack_h=True)
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        self.frame_buffer = []

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def init_MLflow(self):
        import tools_MLflower
        username_mlflow = os.getenv("MLFLOW_TRACKING_USERNAME")
        password_mlflow = os.getenv("MLFLOW_TRACKING_PASSWORD")
        username_ssh    = os.getenv("USERNAME_SSH")
        password_ssh    = os.getenv("PASSWORD_SSH")

        if username_mlflow is None or password_mlflow is None:
            if os.path.isfile('./mlflow.env'):
                username_mlflow, password_mlflow,username_ssh, password_ssh = self.load_env_file('./mlflow.env')

        MLFlower = tools_MLflower.MLFlower(self.config.host_mlflow, self.config.port_mlflow,
                                           username_mlflow=username_mlflow, password_mlflow=password_mlflow,
                                           remote_storage_folder=self.config.remote_storage_folder,
                                           username_ssh=username_ssh, password_ssh=password_ssh)
        return MLFlower
    # ----------------------------------------------------------------------------------------------------------------------
    def load_env_file(self,filename):
        username_mlflow, password_mlflow = None, None
        username_ssh, password_ssh = None, None
        with open(filename) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key == 'MLFLOW_TRACKING_USERNAME':username_mlflow = value
                    if key == 'MLFLOW_TRACKING_PASSWORD':password_mlflow = value
                    if key == 'USERNAME_SSH':username_ssh = value
                    if key == 'PASSWORD_SSH':password_ssh = value
        return username_mlflow, password_mlflow, username_ssh,password_ssh
    # ----------------------------------------------------------------------------------------------------------------------
    def init_detector(self):
        if self.config.detection_model == 'BG_Sub2':
            from DL import utils_detector_BG_Sub2
            D = utils_detector_BG_Sub2.Detector_BG_Sub2(self.folder_out, self.config)
        elif self.config.detection_model == 'Detectron':
            from DL import utils_detector_detectron2
            D = utils_detector_detectron2.Detector_detectron2(self.folder_out, self.config)
        else:
            D = utils_detector_yolo.Detector_yolo(self.folder_out, self.config)

        return D
    # ----------------------------------------------------------------------------------------------------------------------
    def init_tracker(self):
        if self.config.tracking_model == 'DEEPSORT':
            T = utils_tracker_deep_sort.Tracker_deep_sort(self.folder_out)
        elif self.config.tracking_model == 'BOXMOT'    :
            from DL import utils_tracker_boxmot
            T = utils_tracker_boxmot.Tracker_boxmot(self.folder_out)
        else:
            T = utils_tracker_yolo.Tracker_yolo(self.folder_out,self.config)
        return T
    # ----------------------------------------------------------------------------------------------------------------------
    def init_classifier(self):
        C = utils_classifier_yolo.Classifier_yolo(self.folder_out, config=self.config)
        return C
    # ----------------------------------------------------------------------------------------------------------------------
    def get_out_filename_base(self):
        source = self.config.source
        if ('mp4' in source.lower()) or ('avi' in source.lower()) or ('mkv' in source.lower()):
            mode = 'video'
        elif ('https' in source) or (source == '0'):
            mode = 'stream'
        else:
            mode = 'folder'

        if mode == 'video':
            filename_out = source.split('/')[-1].split('.')[0]
            if filename_out == '':
                filename_out = source.split('/')[-2]

        if mode == 'folder':
            filename_out = source.split('/')[-2]

        if mode == 'stream':
            filename_out = source.split('?v=')[-1]

        return filename_out
    # ----------------------------------------------------------------------------------------------------------------------
    def update_config(self,config):
        self.config = config

        self.Tracker = self.init_tracker()
        self.Detector = self.init_detector()

        if self.config.gt is not None:
            self.update_true(pd.read_csv(config.gt, header=None))

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def name_columns(self,df):
        if df.shape[0] == 0:
            df = pd.DataFrame(columns=['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'conf'])

        cols = [c for c in df.columns]
        if 'frame_id' not in cols:
            #print('Renaming columns!!')
            cols[0] = 'frame_id'
            cols[1] = 'track_id'
            cols[2] = 'x1'
            cols[3] = 'y1'
            cols[4] = 'x2'
            cols[5] = 'y2'
            cols[6] = 'conf'
            df.columns = cols
            df = df.astype({'frame_id': int, 'track_id': int, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'conf': float})

        return df
    # ----------------------------------------------------------------------------------------------------------------------
    def update_true(self,df_true,is_xywh=True,resize_scale=None):
        self.df_true = self.name_columns(df_true)
        self.df_true['frame_id'] = self.df_true['frame_id']-1
        if is_xywh:
            self.df_true['x2'] += self.df_true['x1']
            self.df_true['y2'] += self.df_true['y1']

        if resize_scale is not None:
            self.df_true['x1'] = self.df_true['x1'] * resize_scale
            self.df_true['y1'] = self.df_true['y1'] * resize_scale
            self.df_true['x2'] = self.df_true['x2'] * resize_scale
            self.df_true['y2'] = self.df_true['y2'] * resize_scale

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def update_pred(self,df_pred,resize_scale=None):
        if isinstance(df_pred,str):
            df_pred = pd.read_csv(df_pred)

        self.df_pred = self.name_columns(df_pred)
        if resize_scale is not None:
            self.df_pred['x1'] = self.df_pred['x1'] * resize_scale
            self.df_pred['y1'] = self.df_pred['y1'] * resize_scale
            self.df_pred['x2'] = self.df_pred['x2'] * resize_scale
            self.df_pred['y2'] = self.df_pred['y2'] * resize_scale

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def classify_image(self,image):
        res = self.model_classify.predict(source=image, verbose=False, device=self.device)
        class_id = res[0].probs.top1
        conf = res[0].probs.top1conf.cpu().numpy()
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def get_hash(self,x):
        return hashlib.sha256(str(x).encode('utf-8')).hexdigest()

    # ----------------------------------------------------------------------------------------------------------------------
    def get_next_frame(self):
        frame = self.Grabber.get_frame()

        if self.config.resize_ratio is not None:
            H,W = frame.shape[0:2]
            new_H = int(H * self.config.resize_ratio)
            new_W = int(W * self.config.resize_ratio)
            frame = cv2.resize(frame, (new_W, new_H))

        return frame
    # ----------------------------------------------------------------------------------------------------------------------
    def get_tracks(self, filename, df_det, frame_id, do_debug=False):

        if self.config.do_tracking or self.config.do_profiling:
            if self.Tracker is None:
                df_track = tools_DF.add_column(df_det.copy(), 'track_id', -1, pos=1)
            elif self.Tracker.__class__.__name__ in ['Tracker_KCF','Tracker_SiamFC','Tracker_Nano','Tracker_optical_flow']:
                df_track = self.Tracker.track_detections(self.cnfg.ROI, filename,frame_id=frame_id, do_debug=do_debug)
            else:
                df_track = self.Tracker.track_detections(df_det, filename,frame_id=frame_id, do_debug=do_debug)
        else:
            df_track = tools_DF.add_column(df_det.copy(), 'track_id', -1, pos=1)

        if 'frame_id' not in [c for c in df_track.columns]:
            df_track = tools_DF.add_column(df_track, 'frame_id', frame_id)

        df_track = df_track.astype({'frame_id':int,'track_id': int,'x1':int,'x2':int,'y1':int,'y2':int,'conf':float})

        return df_track
    # ----------------------------------------------------------------------------------------------------------------------
    def match_E00(self,df_det,df_track):

        if df_det.shape[0] == 0: return df_track
        col_start = [c for c in df_det.columns].index('conf') + 1
        CCC = [c for c in df_det.columns[col_start:-1]]
        if 'class_ids' not in CCC and 'class_ids' in df_det.columns:
            CCC+=['class_ids']
        if 'class_name' not in CCC and 'class_name' in df_det.columns:
            CCC+=['class_name']


        emb_C = len(CCC)
        if emb_C<=0: return df_track

        if df_track.shape[0] > 0:
            df_det['row_id'] = numpy.arange(df_det.shape[0])
            df_track['row_id'] = -1

            for r, row in df_track.iterrows():
                d1,d2,d3,d4 = abs(df_det['x1'] - row['x1']) , abs(df_det['y1'] - row['y1']) , abs(df_det['x2'] - row['x2'])  , abs(df_det['y2'] - row['y2'])
                idx = numpy.argmin(d1+d2+d3+d4)
                val = numpy.min(d1+d2+d3+d4)
                df_track.iloc[r,-1] = idx


            df_track = tools_DF.fetch(df_track,'row_id',df_det,'row_id',[c for c in CCC])
            df_track[['detx1', 'dety1', 'detx2', 'dety2']] = numpy.nan
            df_track[df_track['row_id']!=-1] = tools_DF.fetch(df_track[df_track['row_id']!=-1],'row_id', df_det,'row_id', ['x1','y1','x2','y2'],col_new_name=['detx1','dety1','detx2','dety2'])
            df_track = df_track.astype({'x1':int,'y1':int,'x2':int,'y2':int})
            df_track = df_track.astype({'detx1': int, 'detx2': int, 'dety1': int, 'dety2': int})
            df_track.rename(columns={"x1": "tx1", "y1": "ty1", "x2": "tx2", "y2": "ty2"}, inplace=True)
            df_track.rename(columns={"detx1": "x1", "dety1": "y1", "detx2":"x2","dety2":"y2"}, inplace=True)

            df_det.drop(columns=['row_id'], inplace=True)
            df_track.drop(columns=['row_id'], inplace=True)

        return df_track
    # ----------------------------------------------------------------------------------------------------------------------
    def match_E(self,df_det,df_track,iou_thresh= 0.3,center_dist_px = 32.0,extra_cols = None):
        def _boxes_to_np(df):return df[['x1', 'y1', 'x2', 'y2']].to_numpy(dtype=float, copy=False)

        def _iou_matrix(A, B):
            Ax1, Ay1, Ax2, Ay2 = [A[:, i][:, None] for i in range(4)]
            Bx1, By1, Bx2, By2 = [B[:, i][None, :] for i in range(4)]
            inter_w = numpy.clip(numpy.minimum(Ax2, Bx2) - numpy.maximum(Ax1, Bx1), 0, None)
            inter_h = numpy.clip(numpy.minimum(Ay2, By2) - numpy.maximum(Ay1, By1), 0, None)
            inter = inter_w * inter_h
            areaA = (Ax2 - Ax1) * (Ay2 - Ay1)
            areaB = (Bx2 - Bx1) * (By2 - By1)
            union = areaA + areaB - inter
            with numpy.errstate(divide='ignore', invalid='ignore'):
                return numpy.where(union > 0, inter / union, 0.0)

        def _ensure_schema(df_track, meta_cols):
            df_track = df_track.copy()
            for c in ['tx1', 'ty1', 'tx2', 'ty2']:
                if c not in df_track.columns:
                    df_track[c] = pd.Series(dtype='Int64')
            for c in meta_cols:
                if c not in df_track.columns:
                    df_track[c] = pd.Series(dtype='object')
            return df_track

        if not self.config.do_classification:
            return df_track

        if df_track is None:
            df_track = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])

        det_cols = set(df_det.columns) if df_det is not None else set()
        meta_cols = [c for c in ['class_ids', 'class_id', 'class', 'class_name', 'conf'] if c in det_cols]
        if extra_cols:
            meta_cols += [c for c in extra_cols if c in det_cols]
        meta_cols = list(dict.fromkeys(meta_cols))


        df_track = _ensure_schema(df_track, meta_cols)


        if len(df_track) == 0:
            return df_track


        if df_det is None or len(df_det) == 0:
            df_track[['tx1', 'ty1', 'tx2', 'ty2']] = df_track[['x1', 'y1', 'x2', 'y2']].to_numpy()
            for c in ['x1', 'y1', 'x2', 'y2', 'tx1', 'ty1', 'tx2', 'ty2']:
                if c in df_track.columns:
                    df_track[c] = df_track[c].astype('Int64', errors='ignore')
            return df_track

        df_det = df_det.reset_index(drop=True).copy()
        df_track = df_track.reset_index(drop=True).copy()


        df_track[['tx1', 'ty1', 'tx2', 'ty2']] = df_track[['x1', 'y1', 'x2', 'y2']].to_numpy()

        A = _boxes_to_np(df_track)
        B = _boxes_to_np(df_det)

        iou = _iou_matrix(A, B)
        cost = 1.0 - iou
        r_idx, c_idx = linear_sum_assignment(cost)

        tracks_c = numpy.column_stack(((A[:, 0] + A[:, 2]) * 0.5, (A[:, 1] + A[:, 3]) * 0.5))
        dets_c = numpy.column_stack(((B[:, 0] + B[:, 2]) * 0.5, (B[:, 1] + B[:, 3]) * 0.5))

        matched_t, matched_d = [], []
        for ti, di in zip(r_idx, c_idx):
            if iou[ti, di] >= iou_thresh or numpy.linalg.norm(tracks_c[ti] - dets_c[di]) <= center_dist_px:
                matched_t.append(ti);
                matched_d.append(di)

        if matched_t:
            t_idx = numpy.asarray(matched_t, dtype=int)
            d_idx = numpy.asarray(matched_d, dtype=int)
            df_track.loc[t_idx, ['x1', 'y1', 'x2', 'y2']] = df_det.loc[d_idx, ['x1', 'y1', 'x2', 'y2']].to_numpy()

            for c in meta_cols:
                df_track.loc[t_idx, c] = df_det.loc[d_idx, c].to_numpy()

        for c in ['x1', 'y1', 'x2', 'y2', 'tx1', 'ty1', 'tx2', 'ty2']:
            if c in df_track.columns:
                df_track[c] = df_track[c].astype('Int64', errors='ignore')

        return df_track

    # ----------------------------------------------------------------------------------------------------------------------
    def fetch_lifetimes(self, df_track_frame):
        df_track_frame_lifetime = df_track_frame.copy()

        if (self.config.do_tracking is False and self.config.do_profiling is False) is None or self.df_pred is None or self.df_pred.shape[0]==0:
            df_track_frame_lifetime['lifetime'] = int(0)
        else:
            df_cnt = tools_DF.my_agg(self.df_pred, cols_groupby=['track_id'], cols_value=['frame_id'], aggs=['count'],list_res_names=['lifetime'])
            df_track_frame_lifetime = df_track_frame_lifetime.merge(df_cnt, how='left', on='track_id')
            df_track_frame_lifetime['lifetime'] = df_track_frame_lifetime['lifetime'].fillna(0)

        df_track_frame_lifetime = df_track_frame_lifetime.astype({'lifetime': int})

        return df_track_frame_lifetime
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_traces_normal(self, image, det, color=(0, 128, 255)):
        if det.shape[0] > 0:
            for obj_id in det['track_id'].unique().astype(int):
                det_local = det[det['track_id'] == obj_id].sort_values(by=['frame_id'], ascending=False)

                points = det_local[['x1', 'y1', 'x2', 'y2']].values
                centers = numpy.concatenate((points[:, [0, 2]].mean(axis=1).reshape((-1, 1)), points[:, [1, 3]].mean(axis=1).reshape((-1, 1))),axis=1).astype(float).reshape((-1, 2))

                lines = numpy.concatenate((centers[:-1], centers[1:]), axis=1).astype(int)
                #lines = self.smooth_line_poly(lines)
                for i in range(lines.shape[0]-1):
                    image = tools_draw_numpy.draw_line_fast(image, lines[i,1],lines[i,0],lines[i+1,1],lines[i+1,0],color, w=1)


        return image
    # ----------------------------------------------------------------------------------------------------------------------
    def get_hash(self,x):
        return hashlib.sha256(str(x).encode('utf-8')).hexdigest()
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_detects(self, image, rects, labels=None, colors=(0, 128, 255)):
        if labels is None:
            labels = ['' for r in rects]

        colors = [colors] * len(rects) if isinstance(colors, tuple) else colors

        for rect, label,color in zip(rects, labels,colors):
            col_left, row_up, col_right, row_down = rect.flatten()
            image = tools_draw_numpy.draw_line_fast(image, int(row_up),   int(col_left), int(row_up), int(col_right),color, w=3)
            image = tools_draw_numpy.draw_line_fast(image, int(row_down), int(col_left), int(row_down), int(col_right),color, w=3)
            image = tools_draw_numpy.draw_line_fast(image, int(row_up),   int(col_left), int(row_down), int(col_left),color, w=3)
            image = tools_draw_numpy.draw_line_fast(image, int(row_up),   int(col_right), int(row_down), int(col_right),color, w=3)

            if label != '':
                color_fg = (0, 0, 0) if 10 * color[0] + 60 * color[1] + 30 * color[2] > 100 * 128 else (255, 255, 255)
                image = tools_draw_numpy.draw_text_fast(image, str(label), (int(col_left), int(row_up)), color_fg=color_fg,clr_bg=color, font_size=16)

        return image

    # ----------------------------------------------------------------------------------------------------------------------
    def draw_tracks(self, image, rects, track_ids=None, color=None,highlight_ids=[]):

        if track_ids is None:
            track_ids =[-1]*len(rects)

        if color is None:
            colors = [self.colors80[track_id % 80] if track_id!=-1 else (0,128,255) for track_id in track_ids]
        else:
            colors = [color]*len(track_ids)

        for rect, track_id, color in zip(rects, track_ids, colors):
            col_left, row_up, col_right, row_down = rect.flatten()
            image = tools_draw_numpy.draw_rect_fast(image, col_left, row_up, col_right, row_down, color,w=-1,alpha_transp=0.5)
            if track_id >=0:
                color_fg = (0, 0, 0) if 10 * int(color[0]) + 60 * int(color[1]) + 30 * int(color[2]) > 100 * 128 else (255, 255, 255)
                image = tools_draw_numpy.draw_text_fast(image, f'{self.get_hash(track_id)[:2]}',(int(col_left), int(row_up)), color_fg=color_fg, clr_bg=color,font_size=16)

            if track_id in highlight_ids:
                image = tools_draw_numpy.draw_line_fast(image, int(row_up), int(col_left), int(row_up), int(col_right),color, w=3)
                image = tools_draw_numpy.draw_line_fast(image, int(row_down), int(col_left), int(row_down),int(col_right), color, w=3)
                image = tools_draw_numpy.draw_line_fast(image, int(row_up), int(col_left), int(row_down), int(col_left),color, w=3)
                image = tools_draw_numpy.draw_line_fast(image, int(row_up), int(col_right), int(row_down),int(col_right), color, w=3)

        return image

    # ----------------------------------------------------------------------------------------------------------------------
    def mix_frame_and_mask(self,frame,mask,color,alpha=0.75):
        idx = numpy.where(mask > 0)
        image_debug = frame.copy()
        image_debug[idx[0], idx[1], :] = color
        image_debug = cv2.addWeighted(tools_image.desaturate(frame), (1-alpha), image_debug, alpha, 0)
        return image_debug
    # ----------------------------------------------------------------------------------------------------------------------
    def print_debug_info(self, image, font_size=18):
        if image is None: return image
        clr_bg = (192, 192, 192) if image[:200, :200].mean() > 192 else (64, 64, 64)
        color_fg = (32, 32, 32) if image[:200, :200].mean() > 192 else (192, 192, 192)
        space = font_size + 20

        frame_id = self.HB.get_frame_id()
        total_frames = self.Grabber.get_max_frame_id()
        if total_frames == numpy.inf:
            total_frames = 0
        delta_time = self.HB.get_delta_time()
        fps = self.HB.get_fps()
        label = '%06d / %06d | %.1f sec | %.1f fps @ %dpx | %.0f ms' % (frame_id, total_frames,delta_time, fps, image.shape[0] ,1000.0 / (fps + 1e-4))
        image = tools_draw_numpy.draw_text_fast(image, label, (0, space * 1), color_fg=color_fg, clr_bg=clr_bg, font_size=font_size)
        return image
    # ----------------------------------------------------------------------------------------------------------------------
    def construct_timeline(self,W,df_pred):
        H = 200
        last_n_frames = 300
        frame_interval_stop = self.HB.get_frame_id()
        frame_interval_start = frame_interval_stop - last_n_frames

        image = numpy.full((H, W, 3), 32, dtype=numpy.uint8)
        df_pos = tools_DF.my_agg(df_pred,cols_groupby=['track_id'],cols_value=['frame_id'],aggs=['mean'],list_res_names=['position'])
        col = (0, 128, 255)

        for r in range(df_pos.shape[0]):
            track_id = df_pos['track_id'].iloc[r]
            Y = (track_id*5) % H
            frame_start = df_pred[df_pred['track_id'] == track_id]['frame_id'].min()
            frame_stop  = df_pred[df_pred['track_id'] == track_id]['frame_id'].max()

            if (frame_stop < frame_interval_start) or (frame_start > frame_interval_stop):
                continue

            frame_start = max(frame_start, frame_interval_start)
            frame_stop  = min(frame_stop, frame_interval_stop)

            x_start = int((frame_start - frame_interval_start) * (W-1) / last_n_frames)
            x_stop  = int((frame_stop  - frame_interval_start) * (W-1) / last_n_frames)

            image = tools_draw_numpy.draw_line_fast(image, Y,x_start, Y, x_stop, color_bgr=col,w=2)


        return image
    # ----------------------------------------------------------------------------------------------------------------------
    def filter_short_timers(self,df_track_frame):
        df_track_frame_filtered = df_track_frame
        df_retro= pd.DataFrame([])

        if (self.config.do_tracking or self.config.do_profiling) and self.config.track_lifetime is not None and self.config.track_lifetime > 0 and df_track_frame.shape[0]> 0:
            df_track_frame_filtered = df_track_frame[df_track_frame['lifetime'] >= self.config.track_lifetime]

            frame_id = df_track_frame['frame_id'].max()
            track_ids = df_track_frame_filtered[df_track_frame_filtered['lifetime'] == self.config.track_lifetime]['track_id'].unique()
            df_retro = self.df_pred[self.df_pred['track_id'].isin(track_ids)]
            df_retro = df_retro[df_retro['frame_id'] < frame_id]

        return df_track_frame_filtered,df_retro
    # ----------------------------------------------------------------------------------------------------------------------
    # def filter_negatives(self,frame,df_track_frame,df_retro):
    #
    #     df_track_frame_filtered = df_track_frame.copy()
    #     df_retro_filtered = df_retro.copy()
    #
    #     if self.config.do_classification and self.Classifier is not None:
    #         confs = []
    #         for r in range(df_track_frame_filtered.shape[0]):
    #             rect = df_track_frame_filtered.iloc[r][['x1', 'y1', 'x2', 'y2']].values.astype(int)
    #             c = 0.0 if rect[0] < 0 or rect[1] < 0 or rect[2] > frame.shape[1] or rect[3] > frame.shape[0] or rect[3]-rect[1] < 10 or rect[2]-rect[0] < 10 else float(self.Classifier.get_classification(frame[rect[1]:rect[3], rect[0]:rect[2]]))
    #             confs.append(c)
    #
    #         df_track_frame_filtered['conf'] = confs
    #         df_track_frame_filtered = df_track_frame_filtered[df_track_frame_filtered['conf'] >= self.config.classification_confidence_th]
    #
    #         if df_retro_filtered.shape[0] > 0:
    #             df_retro_filtered = df_retro_filtered[df_retro_filtered['track_id'].isin(df_track_frame_filtered['track_id'].unique())]
    #
    #     return df_track_frame_filtered,df_retro_filtered
    # ----------------------------------------------------------------------------------------------------------------------
    def get_classifications(self,frame,df_track_frame):
        df_track_frame_reach = df_track_frame.copy()
        if self.config.do_classification and self.Classifier is not None:
            confs = []
            class_names = []
            for r in range(df_track_frame.shape[0]):
                rect = df_track_frame.iloc[r][['x1', 'y1', 'x2', 'y2']].values.astype(int)

                if rect[0] < 0 or rect[1] < 0 or rect[2] > frame.shape[1] or rect[3] > frame.shape[0] or rect[3] - rect[1] < 10 or rect[2] - rect[0] < 10:
                    c,class_name = 0.0,''
                else:
                    c,class_name = self.Classifier.get_classification(frame[rect[1]:rect[3], rect[0]:rect[2]])

                confs.append(c)
                class_names.append(class_name)

            df_track_frame_reach['class_conf'] = confs
            df_track_frame_reach['class_name'] = class_names

        return df_track_frame_reach
    # ----------------------------------------------------------------------------------------------------------------------
    def process_frame_0_get_rects(self,filename,do_debug=True):
        frame = cv2.imread(filename) if isinstance(filename, str) else filename
        filename_out = 'df_track.csv'
        self.mode = 'a' if os.path.isfile(self.folder_out + filename_out) else 'w'

        self.df_det_frame = self.Detector.get_detections(frame)
        self.df_det_frame = tools_DF.add_column(self.df_det_frame, 'track_id', -1, pos=1)
        self.df_det_frame.to_csv(self.folder_out + filename_out, index=False, header=True if self.mode == 'w' else False,mode=self.mode, float_format='%.2f')

        image_debug = None
        if do_debug:
            image_debug = self.draw_detects(tools_image.desaturate(frame, 0.3), self.df_det_frame[['x1', 'y1', 'x2', 'y2']].values, colors=(0, 128, 255))

        return image_debug
    # ----------------------------------------------------------------------------------------------------------------------
    def process_frame_2_get_tracks(self, filename,do_debug=True):

        self.TP.tic('IO:read')
        frame = cv2.imread(filename) if isinstance(filename, str) else filename
        filename_out = 'df_track.csv'
        self.mode = 'a' if os.path.isfile(self.folder_out + filename_out) else 'w'
        frame_id = self.HB.get_frame_id()
        self.TP.tic('IO:read')

        self.TP.tic('detect')
        if self.config.do_tracking and self.Tracker.__class__.__name__ in ['Tracker_yolo']:
            df_det_frame = None
        else:
            df_det_frame = self.Detector.get_detections(frame)
        self.TP.tic('detect')

        self.TP.tic('track')
        df_track_frame = self.get_tracks(filename, df_det_frame, frame_id=frame_id)
        self.TP.tic('track')

        self.TP.tic('classify')
        df_track_frame = self.match_E(df_det_frame, df_track_frame)
        self.TP.tic('classify')

        self.TP.tic('track')
        df_track_frame = self.fetch_lifetimes(df_track_frame)
        self.df_pred = df_track_frame if self.df_pred is None else (pd.concat([self.df_pred.dropna(how='all', axis=1), df_track_frame.dropna(how='all', axis=1)], axis=0) if self.df_pred.shape[0] > 0 else df_track_frame)
        df_track_frame_filtered = self.df_pred[self.df_pred['frame_id'] == frame_id]
        df_track_frame_filtered,df_retro = self.filter_short_timers(df_track_frame_filtered)
        self.TP.tic('track')

        self.TP.tic('IO:write')
        df_track_frame_filtered.to_csv(self.folder_out + filename_out, index=False, header=True if self.mode == 'w' else False,mode=self.mode, float_format='%.2f')
        df_retro.to_csv               (self.folder_out + filename_out, index=False, header=False,mode='a', float_format='%.2f')
        self.TP.tic('IO:write')

        image_debug = None
        if do_debug:
            self.TP.tic('IO:draw')
            image_debug = tools_image.desaturate(frame, 0.3)

            if self.config.do_tracking or self.config.do_profiling:
                current_frame_id = self.HB.get_frame_id()
                highlight_ids = []
                if self.config.do_profiling:
                    highlight_ids = [track_id for track_id in df_track_frame_filtered['track_id'].values if (current_frame_id - self.df_pred[self.df_pred['track_id']==track_id]['frame_id'].min() <= 20)]
                image_debug = self.draw_tracks(image_debug, df_track_frame_filtered[['x1', 'y1', 'x2', 'y2']].values, track_ids=df_track_frame_filtered['track_id'].values,highlight_ids=highlight_ids)

            if self.config.do_classification:
                labels = (df_track_frame_filtered['class_name'].astype(str) + ' ' + (df_track_frame_filtered['conf'] * 100).round().astype(int).astype(str) + '%').values
                colors = [self.colors80[track_id % 80] if track_id != -1 else (0, 128, 255) for track_id in df_track_frame_filtered['track_id'].values]
                image_debug = self.draw_detects(image_debug, df_track_frame_filtered[['x1', 'y1', 'x2', 'y2']].values, labels=labels, colors=colors)

            if (self.config.do_tracking is False) and (self.config.do_classification is False) and (self.config.do_profiling is False) and self.config.do_detection:
                image_debug = self.draw_detects(image_debug, df_det_frame[['x1', 'y1', 'x2', 'y2']].values, colors=(0, 128, 255))

        return image_debug
    # ----------------------------------------------------------------------------------------------------------------------
    def visualize_tracks_simple(self,conf_th = 0.50):
        self.V.draw_stacked_simple(self.config.source, self.df_true, self.df_pred, conf_th=conf_th)
        tools_animation.folder_to_video_simple(self.folder_out, self.folder_out + 'tracks_simple.mp4')
        tools_IO.remove_files(self.folder_out, '*.jpg')
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def visualize_tracks_RAG(self,use_IDTP=False):
        df_true_rich, df_pred_rich = self.V.B.calc_hits_stats_iou(self.df_true, self.df_pred,iou_th=self.config.iou_th)
        self.V.draw_boxes_GT_pred_stacked(self.config.source, df_true_rich, df_pred_rich, conf_th=0.01, use_IDTP=use_IDTP, as_video=True)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_sequence_recall_precision(self,use_IDTP=False):
        df_true_rich, df_pred_rich = self.V.B.calc_hits_stats_iou(self.df_true, self.df_pred,iou_th=self.config.iou_th)
        self.V.draw_sequence_recall_precision(self.config.source, df_true_rich, df_pred_rich, use_IDTP=use_IDTP)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def calc_benchmarks_custom(self):

        df_true2, df_pred2 = self.V.B.calc_hits_stats_iou(self.df_true, self.df_pred,iou_th=self.config.iou_th)
        ths = df_true2['conf_pred'].unique()
        TPs_det,FNs_det,FPs_det,F1s_det = [],[],[],[]
        TPs_ID,FNs_ID,FPs_ID,F1s_ID = [],[],[],[]

        ths = numpy.sort(ths[~numpy.isnan(ths)])
        for th in ths:
            TPs_det.append(df_true2[(df_true2['conf_pred'] >= th) & (df_true2['pred_row'] != -1)].shape[0])
            FNs_det.append(df_true2.shape[0] - TPs_det[-1])
            FPs_det.append(df_pred2[(df_pred2['conf'] >= th) & (df_pred2['true_row'] == -1)].shape[0])
            F1s_det.append(2 * TPs_det[-1] / (2 * TPs_det[-1] + FPs_det[-1] + FNs_det[-1]) if TPs_det[-1] + FPs_det[-1] + FNs_det[-1] > 0 else 0)

            TPs_ID.append(df_true2[(df_true2['conf_pred'] >= th) & (df_true2['pred_row'] != -1) & (df_true2['IDTP']==True)].shape[0])
            FNs_ID.append(df_true2.shape[0] - TPs_ID[-1])
            FPs_ID.append(df_pred2[(df_pred2['conf'] >= th) & (df_pred2['IDTP']==False)].shape[0])
            F1s_ID.append(2 * TPs_ID[-1] / (2 * TPs_ID[-1] + FPs_ID[-1] + FNs_ID[-1]) if TPs_ID[-1] + FPs_ID[-1] + FNs_ID[-1] > 0 else 0)

        self.V.plot_f1_curve(F1s_det, ths, filename_out=self.folder_out + 'F1_det.png')
        self.V.plot_precision_recall(numpy.array(TPs_det) / (numpy.array(TPs_det) + numpy.array(FPs_det)), numpy.array(TPs_det) / (numpy.array(TPs_det) + numpy.array(FNs_det)),filename_out=self.folder_out + 'PR_det.png',iuo_th=self.config.iou_th)
        self.V.plot_f1_curve(F1s_ID, ths, filename_out=self.folder_out + 'F1_ID.png')
        self.V.plot_precision_recall(numpy.array(TPs_ID) / (numpy.array(TPs_ID) + numpy.array(FPs_ID)), numpy.array(TPs_ID) / (numpy.array(TPs_ID) + numpy.array(FNs_ID)),filename_out=self.folder_out + 'PR_ID.png',iuo_th=self.config.iou_th)

        idx_best = numpy.argmax(F1s_det)
        precision_det = TPs_det[idx_best] / (TPs_det[idx_best] + FPs_det[idx_best]) if TPs_det[idx_best] + FPs_det[idx_best] > 0 else 0
        recall_det = TPs_det[idx_best] / (TPs_det[idx_best] + FNs_det[idx_best]) if TPs_det[idx_best] + FNs_det[idx_best] > 0 else 0
        df_det = pd.DataFrame({'TP':[TPs_det[idx_best]],'FP':[FPs_det[idx_best]],'FN':[FNs_det[idx_best]],'Pr':precision_det,'Rc':recall_det,'F1':[F1s_det[idx_best]],'th':ths[idx_best]}).T
        df_det = df_det.map("{0:.2f}".format)
        dct_names = dict(zip([c for c in df_det.columns], ['Det @%.2f' % iou for iou in [self.config.iou_th]]))
        df_det = df_det.rename(columns=dct_names)

        idx_best = numpy.argmax(F1s_ID)
        precision_ID = TPs_ID[idx_best] / (TPs_ID[idx_best] + FPs_ID[idx_best]) if TPs_ID[idx_best] + FPs_ID[idx_best] > 0 else 0
        recall_ID = TPs_ID[idx_best] / (TPs_ID[idx_best] + FNs_ID[idx_best]) if TPs_ID[idx_best] + FNs_ID[idx_best] > 0 else 0
        df_ID = pd.DataFrame({'TP':[TPs_ID[idx_best]],'FP':[FPs_ID[idx_best]],'FN':[FNs_ID[idx_best]],'Pr':precision_ID,'Rc':recall_ID,'F1':[F1s_ID[idx_best]],'th':ths[idx_best]}).T
        df_ID = df_ID.map("{0:.2f}".format)
        dct_names = dict(zip([c for c in df_ID.columns], ['ID @%.2f' % iou for iou in [self.config.iou_th]]))
        df_ID = df_ID.rename(columns=dct_names)

        self.df_summary_custom = pd.concat([df_det,df_ID],axis=1)
        self.df_summary_custom.to_csv(self.folder_out + 'df_summary.csv', index=True)

        print(tools_DF.prettify(self.df_summary_custom, showindex=True))

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def benchmarks_to_dict(self,filename_df_summary=None):
        dct_metrics = {}
        if filename_df_summary is not None:
            if os.path.isfile(filename_df_summary):
                self.df_summary_custom = pd.read_csv(filename_df_summary)
            if self.df_summary_custom is not None and self.df_summary_custom.shape[0] > 0:
                dct_metrics = dict(zip(self.df_summary_custom.iloc[:,0].values.flatten(), self.df_summary_custom.iloc[:,1].values.flatten()))

        return dct_metrics
    # ----------------------------------------------------------------------------------------------------------------------
    def create_profiles(self,use_gt=False):
        images,significance,track_ids,meta_seconds = [],[],[],[]
        resize_ratio = self.config.resize_ratio

        if use_gt:
            for obj_id in self.df_true['track_id'].unique():
                df_repr = self.df_true[self.df_true['track_id'] == obj_id].copy()
                df_repr['size'] = (df_repr['x2'] - df_repr['x1']) * (df_repr['y2'] - df_repr['y1'])
                df_repr = df_repr.sort_values(by='size', ascending=False)
                image = self.Grabber.get_frame(frame_id=df_repr.iloc[0]['frame_id'])
                rect = df_repr.iloc[0][['x1', 'y1', 'x2', 'y2']].values.astype(int)
                rect = (rect / resize_ratio).astype(int) if resize_ratio is not None else rect

                images.append(image[rect[1]:rect[3], rect[0]:rect[2]])
                track_ids.append(obj_id)
        else:
            for obj_id in self.df_pred['track_id'].unique():
                df_repr = self.df_pred[self.df_pred['track_id'] == obj_id].copy()
                df_repr['size'] = (df_repr['x2'] - df_repr['x1']) * (df_repr['y2'] - df_repr['y1'])
                df_repr = df_repr.sort_values(by='size', ascending=False)
                image = self.Grabber.get_frame(frame_id=df_repr.iloc[0]['frame_id'])
                if image is None:
                    print(f'Warning: Frame {df_repr.iloc[0]["frame_id"]} not found for object {obj_id}. Skipping.')
                    continue

                rect = df_repr.iloc[0][['x1', 'y1', 'x2', 'y2']].values.astype(int)
                rect[0] = max(0, rect[0])
                rect[1] = max(0, rect[1])
                rect[2] = min(image.shape[1], rect[2])
                rect[3] = min(image.shape[0], rect[3])
                rect = (rect / resize_ratio).astype(int) if resize_ratio is not None else rect

                images.append(image[rect[1]:rect[3], rect[0]:rect[2]])
                significance.append(df_repr['frame_id'].min())
                track_ids.append(obj_id)
                meta_seconds.append(int((self.HB.get_frame_id() - self.df_pred[self.df_pred['track_id'] == obj_id]['frame_id'].min()) / self.HB.get_fps()))

            idx = numpy.argsort(significance)[::-1]
            images = [images[i] for i in idx]
            track_ids = [track_ids[i] for i in idx]

        return images, track_ids,meta_seconds
    # ----------------------------------------------------------------------------------------------------------------------

    def stack_profiles(self,images,track_ids,width=64,height=64,seconds_keep_inactive=30,color_bg=(240, 240, 240),color_fg=(0, 0, 0)):
        tol = 2
        result = None
        images_live = []
        images_retro = []

        for i,image in enumerate(images):
            track_id = track_ids[i]
            class_name = None
            if 'class_name' in self.df_pred.columns:
                class_name = self.df_pred[self.df_pred['track_id'] == track_id]['class_name'].values[-1]
                if str(class_name) == 'nan':
                    class_name = None

            is_live = self.df_pred[self.df_pred['frame_id']>= self.HB.get_frame_id()-tol]['track_id'].isin([track_id]).any()
            color = (self.colors80[track_id % 80] if track_id >= 0 else (0, 128, 255)) if is_live else (64, 64, 64)
            color_fg_id = (0, 0, 0) if 10 * int(color[0]) + 60 * int(color[1]) + 30 * int(color[2]) > 100 * 128 else (255, 255, 255)

            meta_seconds = int((self.HB.get_frame_id() - self.df_pred[self.df_pred['track_id'] == track_id]['frame_id'].min() )/ self.HB.get_fps())
            image_obj = tools_image.smart_resize(image, height, width,bg_color=color_bg,align_center=False)
            image_metadata = numpy.full((height, 80, 3), color_bg, dtype=numpy.uint8)
            image_metadata = tools_draw_numpy.draw_text_fast(image_metadata, f'{class_name}', (2, 2), color_fg=color_fg, clr_bg=color_bg, font_size=16) if class_name is not None else image_metadata
            image_metadata = tools_draw_numpy.draw_text_fast(image_metadata, f'{meta_seconds} sec', (2, 20),color_fg=color_fg, clr_bg=color_bg, font_size=16)
            image_colorbar = numpy.full((height, 30, 3), color, dtype=numpy.uint8)
            image_colorbar = tools_draw_numpy.draw_text_fast(image_colorbar, f'{self.get_hash(track_id)[:2]}',(int(2), int(2)), color_fg=color_fg_id, clr_bg=color,font_size=16)
            image = numpy.concatenate((image_metadata, image_colorbar, image_obj), axis=1)

            if is_live                              :images_live.append(image)
            elif meta_seconds<=seconds_keep_inactive:images_retro.append(image)

        if len(images_live+images_retro)>0:
            image_break = numpy.full((24, width+80+30, 3), color_bg, dtype=numpy.uint8)
            image_break[12] = color_fg
            result = numpy.concatenate(images_live +[image_break] +images_retro,axis=0)

        return result
    # ----------------------------------------------------------------------------------------------------------------------
    def save_profiles(self):

        tools_IO.remove_files(self.folder_out, 'profile_*.png')
        if not self.Grabber.exhausted:
            progress_bar = tqdm(total=self.Grabber.get_max_frame_id(), desc='Profiling',bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            prev = 0
            while not self.Grabber.exhausted:
                time.sleep(0.1)
                current = len(self.Grabber.frame_buffer)
                progress_bar.update(current-prev)
                prev = len(self.Grabber.frame_buffer)
            progress_bar.close()

        images,track_ids,meta_seconds = self.create_profiles(use_gt=False)

        for i,image in enumerate(images):
            cv2.imwrite(self.folder_out + f'profile_{self.get_hash(track_ids[i])[:2]}.png', image)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_time_lapse(self,df_pred,W,H,color_bg_rgb,dct_profiles,W_seconds = 10):
        max_items = 5
        margin = 100
        height_px = int(H/ max_items)
        now = self.HB.get_frame_id()
        fps = self.HB.get_fps()
        W_frames = int(W_seconds*fps)
        def frame_id_to_pixel_id(frame_id):return (W-1-margin) - (now - frame_id)*(W-1-margin)/W_frames

        image = numpy.full((H, W, 3), color_bg_rgb, dtype=numpy.uint8)
        if df_pred is None or df_pred.shape[0] == 0:
            return image

        df_pos_start = tools_DF.my_agg(df_pred,cols_groupby=['track_id'],cols_value=['frame_id'],aggs=['min'],list_res_names=['position_start'])
        df_pos_stop  = tools_DF.my_agg(df_pred,cols_groupby=['track_id'],cols_value=['frame_id'],aggs=['max'],list_res_names=['position_stop'])
        df_pos = df_pos_start.merge(df_pos_stop, on='track_id', how='inner')
        df_pos = df_pos.sort_values(by=['track_id'],ascending=False)

        for r,track_id in enumerate(dct_profiles.keys()):
            if r * height_px> H:break
            track_id = df_pos['track_id'].iloc[r]
            small_image = dct_profiles[track_id]
            scale = height_px / small_image.shape[0]
            small_image = cv2.resize(small_image, (int(small_image.shape[1] * scale), height_px))
            i1 = frame_id_to_pixel_id(df_pos['position_start'].iloc[r])
            i2 = frame_id_to_pixel_id(df_pos['position_stop'].iloc[r])
            Y = int(r * height_px)
            col = self.colors80[(track_id) % 80]
            image = tools_draw_numpy.draw_line_fast(image, Y+height_px/2, i1, Y+height_px/2, i2, color_bgr=col, w=2)
            image= tools_image.put_image(image,small_image,Y,int(i2)-10)

        return image
    # ----------------------------------------------------------------------------------------------------------------------
    def create_folder_train_test_val(self):
        for name in ['train/', 'test/', 'val/']:
            if not os.path.isdir(self.folder_out + name):
                os.mkdir(self.folder_out + name)

            tools_IO.remove_files(self.folder_out + name, '*.jpg')
            tools_IO.remove_folders(self.folder_out + name)
            os.mkdir(self.folder_out + name + 'P')
            os.mkdir(self.folder_out + name + 'N')
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def get_train_test_val_type(self, splits=(0.7,0.9,1.0)):
        if   numpy.random.rand() < splits[0]:name = 'train/'
        elif numpy.random.rand() < splits[1]:name = 'test/'
        else:                                name = 'val/'
        return name
    # ----------------------------------------------------------------------------------------------------------------------
    def create_samples_pos_neg(self):

        df_true_rich, df_pred_rich = self.V.B.calc_hits_stats_iou(self.df_true, self.df_pred,iou_th=self.config.iou_th)
        self.create_folder_train_test_val()

        for frame_id in tqdm(df_pred_rich['frame_id'].unique(), total=df_pred_rich['frame_id'].unique().shape[0],desc=inspect.currentframe().f_code.co_name):
            image = self.get_image_by_frame_id(frame_id)
            df_repr = df_pred_rich[df_pred_rich['frame_id'] == frame_id].copy()

            for r in range(df_repr.shape[0]):
                is_TP = df_repr.iloc[r]['true_row'] > 0
                track_id = df_repr.iloc[r]['track_id'].astype(int)
                track_id_true = df_repr.iloc[r]['track_id_true'].astype(int) if is_TP else 0
                rect = df_repr.iloc[r][['x1', 'y1', 'x2', 'y2']].values.astype(int)

                rect[0] = max(0, rect[0])
                rect[1] = max(0, rect[1])
                rect[2] = min(image.shape[1], rect[2])
                rect[3] = min(image.shape[0], rect[3])
                small = image[rect[1]:rect[3], rect[0]:rect[2]]
                if small.shape[0] <10 or small.shape[1] < 10:
                    continue

                name=  f"{('P_' if is_TP else 'N_')}"
                name+= f"{self.config.exp_name}_"
                name+= f"{track_id_true}_" if is_TP else f"{track_id}_"
                name+= f"{frame_id:06d}.jpg"

                sub_folder = self.get_train_test_val_type() + ('P/' if is_TP else 'N/')
                cv2.imwrite(self.folder_out + sub_folder + name , small)

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def save_experiment(self,name):
        
        if not self.MLFlower.is_available:
            print('MLFlow unavailable')
            return

        #self.TP.stage_stats(self.folder_out + 'time_profiles.csv')
        self.config.save(self.folder_out + 'config.json')

        artifacts = [(self.folder_out + filename) for ext in ['*.csv', '*.mp4', '*.png'] for filename in tools_IO.get_filenames(self.folder_out, ext)]

        run_id = self.MLFlower.save_experiment(experiment_name=name,
                             params={k: getattr(self.config, k) for k in dir(self.config) if not k.startswith('__') and not callable(getattr(self.config, k))},
                             metrics=self.benchmarks_to_dict(self.folder_out + 'df_summary.csv'),
                             artifacts=artifacts
                             )
        print(f'Run saved: {run_id}')
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def save_captured_video(self):
        if len(self.frame_buffer)>0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            resize_H, resize_W = self.frame_buffer[0].shape[:2]
            out = cv2.VideoWriter(self.folder_out+'video.mp4', fourcc, len(self.frame_buffer) / self.HB.get_delta_time(), (resize_W, resize_H))
            for image in tqdm(self.frame_buffer, total=len(self.frame_buffer), desc='Writing video'):
                out.write(image)
            out.release()

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update_grabber(self, Grabber):
        if self.Grabber is not None:
            self.Grabber.should_be_closed = True
            time.sleep(0.1)

        self.Grabber = Grabber
        self.df_pred = None
        time.sleep(0.2)
        self.HB = tools_heartbeat.tools_HB()

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def process_video(self,save_images=None):
        do_debug = ((save_images is not None) and save_images is not False )
        tools_IO.remove_files(self.folder_out, '*.csv')
        tools_IO.remove_files(self.folder_out, '*.jpg')
        tools_IO.remove_files(self.folder_out, '*.png')
        tools_IO.remove_files(self.folder_out, '*.mp4')

        N = min(self.Grabber.get_max_frame_id(),(self.config.limit if self.config.limit is not None else numpy.inf))

        for i in tqdm(range(int(N)), desc='processing video'):
            self.HB.do_heartbeat()
            image = self.get_next_frame()

            if image is None: break

            if   self.config.do_tracking:       image = self.process_frame_2_get_tracks(image,do_debug)
            elif self.config.do_classification: image = self.process_frame_2_get_tracks(image,do_debug)
            elif self.config.do_detection:      image = self.process_frame_0_get_rects(image,do_debug)

            image = self.print_debug_info(image)
            image = self.Grabber.print_debug_info(image)

            if   save_images=='as_video':self.frame_buffer.append(image)
            elif save_images is True    :cv2.imwrite(self.folder_out + 'frame_%06d.jpg' % i, image)


        self.TP.stage_stats(self.folder_out + 'time_profiles.csv')
        if save_images == 'as_video':
            self.save_captured_video()

        return
# ----------------------------------------------------------------------------------------------------------------------
