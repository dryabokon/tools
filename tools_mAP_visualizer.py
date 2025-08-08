import cv2
import os
import numpy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import inspect
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_draw_numpy
import tools_mAP
import tools_DF

# ----------------------------------------------------------------------------------------------------------------------
class Track_Visualizer:
    def __init__(self,folder_out,stack_h=True):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.stack_h = stack_h
        self.folder_out = folder_out
        self.B = tools_mAP.Benchmarker(folder_out=folder_out)
        self.hit_colors_true = [(0, 0, 192), (32, 132, 0),(0, 104, 192)]
        self.hit_colors_pred = [(0, 158, 168), (128, 128, 0)]
        self.colors80 = tools_draw_numpy.get_colors(80, shuffle=True)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_color_true(self,is_hit,is_hit_track,use_IDTP):
        if use_IDTP:
            if is_hit_track:
                color = self.hit_colors_true[1]
            else:
                if is_hit:
                    color = self.hit_colors_true[2]
                else:
                    color = self.hit_colors_true[0]
        else:
            color = self.hit_colors_true[int(is_hit)]

        return color
# ----------------------------------------------------------------------------------------------------------------------
    def get_color_pred(self,is_hit,is_hit_track,use_IDTP):
        if use_IDTP:
            color = self.hit_colors_pred[int(is_hit_track)]
        else:
            color = self.hit_colors_pred[int(is_hit)]

        return color
    # ----------------------------------------------------------------------------------------------------------------------
    def get_image_by_frame_id(self, source, df_filenames, frame_id, vidcap):
        if vidcap is None:
            filename = df_filenames[df_filenames['frame_id'] == frame_id]['filename'].values[0]
            image = cv2.imread(source + filename)
        else:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success, image = vidcap.read()
        return image
    # ----------------------------------------------------------------------------------------------------------------------
    def get_representatives(self,source,df,idx_sort=None,H=64,layout='ver'):

        is_video = ('mp4' in source.lower()) or ('avi' in source.lower()) or ('mkv' in source.lower())

        if is_video:
            vidcap = cv2.VideoCapture(source)
            df_filenames = pd.DataFrame({'frame_id': numpy.arange(0, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))})
        else:
            vidcap = None
            df_filenames = pd.DataFrame({'filename': tools_IO.get_filenames(source, '*.jpg,*.png')})
            df_filenames['frame_id'] = numpy.arange(0, df_filenames.shape[0])

        df['obj_size'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
        df['size'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])

        df_agg = tools_DF.my_agg(df,cols_groupby=['track_id'],cols_value=['obj_size'],aggs=['max'],list_res_names=['obj_size_max'])
        df2 = tools_DF.fetch(df,'track_id',df_agg,col_name2='track_id',col_value='obj_size_max')
        df2 = df[df['obj_size'] == df2['obj_size_max']]
        df2 = df2.drop_duplicates(subset=['track_id'])

        if idx_sort is not None:
            df2 = pd.concat([df2[df2['track_id']==i] for i in idx_sort])
        else:
            df2.sort_values(by=['track_id'], inplace=True)

        if layout=='ver':
            im_large = numpy.full((H * df2.shape[0],2*H, 3), 128,dtype=numpy.uint8)
            im_large[:,:H]=32
        else:
            im_large = numpy.full((2 * H, H * df2.shape[0], 3), 128,dtype=numpy.uint8)
            im_large[:H, :]=32

        for r in range(df2.shape[0]):
            frame_id = df2['frame_id'].iloc[r]
            track_id = df2['track_id'].iloc[r]
            rect = df2[['x1','y1','x2','y2']].iloc[r].values.astype(int)
            image = self.get_image_by_frame_id(source, df_filenames, frame_id, vidcap)
            if image is None: continue

            rect[0] = max(0, rect[0])
            rect[1] = max(0, rect[1])
            rect[2] = min(image.shape[1], rect[2])
            rect[3] = min(image.shape[0], rect[3])

            image = image[rect[1]:rect[3], rect[0]:rect[2]]
            if image.shape[0]>image.shape[1]:
                image = tools_image.smart_resize(image, H)
            else:
                image = tools_image.smart_resize(image,None, H)
            if layout=='ver':
                im_large = tools_image.put_image(im_large, image, r*H+(H-image.shape[0])//2,H+(H-image.shape[1])//2)
                im_large = tools_draw_numpy.draw_text(im_large, str(track_id), (H//2, r*H + H//2), color_fg=(255, 255, 255),hor_align='center',vert_align='middle')
            else:
                im_large = tools_image.put_image(im_large, image, H+(H-image.shape[0])//2, r*H + (H-image.shape[1])//2)
                im_large = tools_draw_numpy.draw_text(im_large, str(track_id), (r*H + H//2, H//2), color_fg=(255, 255, 255),hor_align='center',vert_align='middle')

        return im_large
# ----------------------------------------------------------------------------------------------------------------------
    def draw_confusion_matrix(self,folder_in_images,df_true_rich, df_pred_rich,df_conf,H=64):

        im_represent_true = self.get_representatives(folder_in_images, df_true_rich,idx_sort=df_conf.index.values,H=H)
        im_represent_pred = self.get_representatives(folder_in_images, df_pred_rich,idx_sort=df_conf.columns.values,H=H,layout='hor')
        image = numpy.full((im_represent_true.shape[0]+2*H,im_represent_pred.shape[1]+2*H,3), 32, dtype=numpy.uint8)

        conf_mat = df_conf.values

        for r in range(conf_mat.shape[0]):
            conf_mat[r,:]/=(conf_mat[r,:].sum()+1e-6)

        im_cnf = tools_image.hitmap2d_to_viridis(255*conf_mat)
        im_cnf = cv2.resize(im_cnf, (im_cnf.shape[1] * H, im_cnf.shape[0] * H), interpolation=cv2.INTER_NEAREST)
        im_cnf = tools_image.replace_color(im_cnf, (84,1,68), (32,32,32))
        for r in range(df_conf.shape[0]):im_cnf[r*H] = 64
        for c in range(df_conf.shape[1]):im_cnf[:,c*H] = 64

        image = tools_image.put_image(image,im_represent_true,2*H,0)
        image = tools_image.put_image(image,im_represent_pred,0,2*H)
        image = tools_image.put_image(image, im_cnf, 2*H, 2*H)

        cv2.imwrite(self.folder_out + 'conf_mat.png', image)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_sequence_recall_precision(self, source, df_true_rich, df_pred_rich, conf_th=0.10, use_IDTP=True, H = 64):

        if df_true_rich.shape[0]>0:
            cols = df_true_rich.iloc[:, 0].max()
            rows = df_true_rich.iloc[:, 1].unique().shape[0]
            image = numpy.full((rows, cols, 3), 32, dtype=numpy.uint8)
            dct_map = dict(zip(sorted(df_true_rich.iloc[:, 1].unique()), range(rows)))

            for r in range(df_true_rich.shape[0]):
                frame = df_true_rich['frame_id'].iloc[r] - 1
                obj_id = df_true_rich['track_id'].iloc[r]
                is_hit = (df_true_rich['pred_row'].iloc[r]!=-1) and (df_true_rich['conf_pred'].iloc[r]>=conf_th)
                is_hit_track = df_true_rich['IDTP'].iloc[r]
                image[dct_map[obj_id],frame]= self.get_color_true(is_hit,is_hit_track,use_IDTP)

            im_represent = self.get_representatives(source, df_true_rich,H=H)
            image = cv2.resize(image, (800,im_represent.shape[0]), interpolation=cv2.INTER_NEAREST)
            for r in range(rows):image[r*H] = 64
            image = numpy.concatenate((im_represent, image), axis=1)

            cv2.imwrite(self.folder_out+'seq_recall_%02d.png'%int(100*conf_th),image)

        if df_pred_rich.shape[0]>0:
            cols = df_pred_rich['frame_id'].max()
            rows = df_pred_rich['track_id'].unique().shape[0]
            if rows>=2:
                image = numpy.full((rows, cols, 3), 32, dtype=numpy.uint8)
                dct_map = dict(zip(sorted(df_pred_rich.iloc[:, 1].unique()), range(rows)))
                for r in range(df_pred_rich.shape[0]):
                    if df_pred_rich['conf'].iloc[r]<conf_th:continue
                    frame = df_pred_rich['frame_id'].iloc[r] - 1
                    obj_id = df_pred_rich['track_id'].iloc[r]
                    is_hit = (df_pred_rich['true_row'].iloc[r]!=-1)
                    is_hit_track = df_pred_rich['IDTP'].iloc[r]
                    image[dct_map[obj_id], frame] = self.get_color_pred(is_hit,is_hit_track,use_IDTP)

                im_represent = self.get_representatives(source, df_pred_rich)
                image = cv2.resize(image, (800, im_represent.shape[0]), interpolation=cv2.INTER_NEAREST)
                for r in range(rows): image[r * H] = 64
                image = numpy.concatenate((im_represent, image), axis=1)
                cv2.imwrite(self.folder_out + 'seq_precision_%02d.png'%int(100*conf_th),image)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_f1_curve(self, F1, conf, filename_out=None):
        lw = 2
        plt.plot(conf, F1, color='darkgreen', lw=lw)
        plt.grid(which='major', color='lightgray', linestyle='--')
        plt.minorticks_on()
        plt.grid(which='minor', axis='both', color='lightgray', linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Confidence")
        plt.ylabel("F1 score")

        if filename_out is not None:
            plt.savefig(filename_out)

        plt.clf()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_precision_recall(self,precision,recall,filename_out=None,iuo_th=0.5):

        AP = numpy.trapz(precision, x=recall)
        lw = 2
        plt.plot(recall, precision, color='darkgreen', lw=lw)
        plt.grid(which='major', color='lightgray', linestyle='--')
        plt.minorticks_on()
        plt.grid(which='minor', axis='both', color='lightgray', linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(['AP @ IoU %1.2f = %0.2f' % (iuo_th, AP)])

        if filename_out is not None:
            plt.savefig(filename_out)

        plt.clf()

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def get_stacked_simple(self, frame, frame_id, df_true, df_pred, conf_th=0.10):
        frame = tools_image.desaturate(frame)
        gt = df_true[df_true['frame_id']==(frame_id)]
        det = df_pred[df_pred['frame_id'] == (frame_id)]
        det = det[det['conf'] >= conf_th]

        rects_gt = gt[['x1', 'y1', 'x2', 'y2']].values
        obj_id_gt = gt.iloc[:, 1].values.astype(int)

        rects_det = det[['x1', 'y1', 'x2', 'y2']].values
        obj_id_det = det.iloc[:, 1].values.astype(int)

        colors_true = [self.colors80[(x - 1) % 80] for x in obj_id_gt]
        colors_pred = [self.colors80[(x - 1) % 80] for x in obj_id_det]

        image_gt = tools_draw_numpy.draw_rects(frame, rects_gt.reshape((-1, 2, 2)), colors=colors_true, labels=obj_id_gt.astype(str), w=2, alpha_transp=0.8)
        image_det = tools_draw_numpy.draw_rects(frame, rects_det.reshape((-1, 2, 2)), colors=colors_pred, labels=obj_id_det.astype(str), w=2, alpha_transp=0.8)
        return image_gt, image_det
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_stacked_simple(self,source, df_true, df_pred, iou_th=0.5,conf_th=0.10):
        tools_IO.remove_files(self.folder_out,'*.jpg')

        is_video = ('mp4' in source.lower()) or ('avi' in source.lower()) or ('mkv' in source.lower())

        if is_video:
            vidcap = cv2.VideoCapture(source)
            df_filenames = pd.DataFrame({'frame_id': numpy.arange(0, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))})
        else:
            vidcap = None
            df_filenames = pd.DataFrame({'filename': tools_IO.get_filenames(source, '*.jpg,*.png')})
            df_filenames['frame_id'] = numpy.arange(0,df_filenames.shape[0])

        for frame_id in tqdm(range(df_filenames.shape[0]), total=df_filenames.shape[0], desc=inspect.currentframe().f_code.co_name):
            frame = self.get_image_by_frame_id(source, df_filenames, frame_id, vidcap)
            image_gt, image_det = self.get_stacked_simple(frame, frame_id, df_true, df_pred, conf_th=conf_th)
            cv2.imwrite(self.folder_out + '%06d' % frame_id + '.jpg',numpy.concatenate((image_gt, image_det), axis=1 if self.stack_h else 0))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_boxes_GT_pred_stacked(self, source, df_true_rich, df_pred_rich, conf_th=0.10, use_IDTP=True,as_video=True):

        tools_IO.remove_files(self.folder_out, '*.jpg')

        is_video = ('mp4' in source.lower()) or ('avi' in source.lower()) or ('mkv' in source.lower())

        if is_video:
            vidcap = cv2.VideoCapture(source)
            df_filenames = pd.DataFrame({'frame_id': numpy.arange(0, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))})
        else:
            vidcap = None
            df_filenames = pd.DataFrame({'filename': tools_IO.get_filenames(source, '*.jpg,*.png')})
            df_filenames['frame_id'] = numpy.arange(0, df_filenames.shape[0])

        frame_id_true = df_true_rich['frame_id'].values
        coord_true = df_true_rich[['x1', 'y1', 'x2', 'y2']].values
        conf_true = df_true_rich['conf_pred'].values
        obj_id_gt = df_true_rich['track_id'].values.astype(int)
        hit_true = ~numpy.isnan(df_true_rich['conf_pred'].values)
        hit_true_IDTP = df_true_rich['IDTP'].values

        frame_id_pred = df_pred_rich['frame_id'].values
        coord_pred = df_pred_rich[['x1', 'y1', 'x2', 'y2']].values
        conf_pred = df_pred_rich['conf'].values
        obj_id_pred = df_pred_rich['track_id'].values
        hit_pred = df_pred_rich['true_row'].values >= 0
        hit_pred_IDTP = df_pred_rich['IDTP'].values

        for frame_id in tqdm(range(df_filenames.shape[0]), total=df_filenames.shape[0],desc=inspect.currentframe().f_code.co_name):

            image0 = self.get_image_by_frame_id(source, df_filenames, frame_id, vidcap)
            if image0 is None: continue

            image_fact = tools_image.desaturate(image0)
            image_pred = tools_image.desaturate(image0)

            idx_fact = numpy.where(df_true_rich['frame_id'] == frame_id)
            idx_pred = numpy.where(df_pred_rich['frame_id'] == frame_id)

            for coord, hit, conf, obj_id, hit_IDTP in zip(coord_true[idx_fact], hit_true[idx_fact], conf_true[idx_fact],obj_id_gt[idx_fact],hit_true_IDTP[idx_fact]):
                if conf < conf_th:hit,hit_IDTP = 0,0
                label = str(obj_id)
                label = None
                image_fact = tools_draw_numpy.draw_rect(image_fact, coord[0], coord[1], coord[2], coord[3],color=self.get_color_true(hit,hit_IDTP,use_IDTP), label=label,w=2, alpha_transp=0.25)

            for coord in coord_pred[numpy.where(frame_id_pred == frame_id)]:
                image_fact = tools_draw_numpy.draw_rect(image_fact, coord[0], coord[1], coord[2], coord[3], color=(128, 128, 128),w=1, alpha_transp=1)

            for coord, hit, conf,obj_id,hit_IDTP in zip(coord_pred[idx_pred], hit_pred[idx_pred], conf_pred[idx_pred],obj_id_pred[idx_pred],hit_pred_IDTP[idx_pred]):
                is_hit = (hit and hit_IDTP) if use_IDTP else hit
                if conf < conf_th:continue
                label = str(obj_id)
                label = None
                image_pred = tools_draw_numpy.draw_rect(image_pred, coord[0], coord[1], coord[2], coord[3],color=self.get_color_pred(is_hit,hit_IDTP,use_IDTP),label=label, w=2, alpha_transp=0.25)

            for coord in coord_true[numpy.where(frame_id_true == frame_id)]:
                image_pred = tools_draw_numpy.draw_rect(image_pred, coord[0], coord[1], coord[2], coord[3], color=(128, 128, 128),w=1, alpha_transp=1)

            image = numpy.concatenate((image_fact, image_pred), axis=1 if self.stack_h else 0)
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(self.folder_out + '%06d' % frame_id + '.jpg', image)

        vidcap.release()

        if as_video:
            filenames = tools_IO.get_filenames(self.folder_out, '*.jpg')
            resize_H, resize_W = cv2.imread(self.folder_out+filenames[0]).shape[:2]
            out = cv2.VideoWriter(self.folder_out + 'tracks_RAG.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (resize_W, resize_H))

            for filename in tqdm(filenames, desc="Writing frames to video", total=len(filenames)):
                out.write(cv2.imread(self.folder_out+filename))
            out.release()
            tools_IO.remove_files(self.folder_out, '*.jpg')

        return
# ----------------------------------------------------------------------------------------------------------------------

    def draw_boxes_BEV(self,df_pred,h_ipersp):
        # colors80 = tools_draw_numpy.get_colors(80, shuffle=True)
        # for r in range(df_pred.shape[0]):
        #     obj_id = df_pred.iloc[r]['track_id']
        #     points = df_pred.iloc[r][['x1', 'y1', 'x2', 'y2']].values.reshape((-1, 2))
        #     points_BEV = cv2.perspectiveTransform(numpy.array(points).astype(numpy.float32).reshape((-1, 1, 2)),h_ipersp).reshape((-1, 2))
        #     color = colors80[(obj_id - 1) % 80]
        #     image_BEV = tools_draw_numpy.draw_contours_cv(image_BEV, points_BEV.reshape((-1, 2)), color=color,w=self.lines_width + 1, transperency=0.60)
        return

# ----------------------------------------------------------------------------------------------------------------------
    def cut_boxes_at_original(self, image, xyxy, pad=10):
        H, W = image.shape[:2]
        mask = numpy.full((image.shape[0], image.shape[1]), 1, dtype=numpy.uint8)
        for box in xyxy.astype(int):
            image[max(0, box[1] - pad):min(box[3] + pad, W), max(0, box[0] - pad):min(box[2] + pad, H)] = 0
            mask[max(0, box[1] - pad):min(box[3] + pad, W), max(0, box[0] - pad):min(box[2] + pad, H)] = 0

        return image, mask
    # ----------------------------------------------------------------------------------------------------------------------
    def remove_bg(self,folder_in,df_pred,list_of_masks='*.jpg,*.png',limit=5000):

        filenames = tools_IO.get_filenames(folder_in, list_of_masks)[:limit]
        H,W = cv2.imread(folder_in+filenames[0]).shape[:2]
        image_S = numpy.zeros((H,W,3),dtype=numpy.longlong)
        image_C = numpy.zeros((H,W  ),dtype=numpy.longlong)

        for i, filename in enumerate(filenames):
            frame_id = i + 1
            image = cv2.imread(folder_in+filename)
            xyxy = df_pred[df_pred['frame_id'] == frame_id][['x1','y1','x2','y2']].values
            image_cut, mask = self.cut_boxes_at_original(image, xyxy)
            if image_cut.shape[0]!=image_S.shape[0] or image_cut.shape[1]!=image_S.shape[1]:
                continue
            image_S+=image_cut
            image_C+=mask

        image_S[:, :, 0] = image_S[:, :, 0] / image_C
        image_S[:, :, 1] = image_S[:, :, 1] / image_C
        image_S[:, :, 2] = image_S[:, :, 2] / image_C
        return image_S.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------