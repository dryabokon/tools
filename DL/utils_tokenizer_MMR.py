import cv2
import numpy
import pandas as pd
import torch
from PIL import Image
from onnxruntime import InferenceSession
from torchvision import transforms
from ultralytics import YOLO
import json
# ----------------------------------------------------------------------------------------------------------------------
import tools_mAP
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Tokenizer_MMR:
    def __init__(self,folder_out,df_true=pd.DataFrame([])):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_model_LP_type()
        self.init_model_color()
        self.init_model_MMR()
        self.init_model_symb()
        self.init_model_LPVD()
        self.B = tools_mAP.Benchmarker(folder_out)

        self.folder_out = folder_out
        self.df_true = self.name_columns(df_true)
        self.df_true['x2'] += self.df_true['x1']
        self.df_true['y2'] += self.df_true['y1']
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        self.dct_mmr_clr={
            "beige":tools_draw_numpy.BGR_from_HTML(color_HTML='#FBECD3')[[2,1,0]],
            "black":tools_draw_numpy.BGR_from_HTML(color_HTML='#000000')[[2,1,0]],
            "blue":tools_draw_numpy.BGR_from_HTML(color_HTML='#0080FF')[[2,1,0]],
            "brown":tools_draw_numpy.BGR_from_HTML(color_HTML='#743C00')[[2,1,0]],
            "gray":tools_draw_numpy.BGR_from_HTML(color_HTML='#808080')[[2,1,0]],
            "green":tools_draw_numpy.BGR_from_HTML(color_HTML='#008000')[[2,1,0]],
            "orange":tools_draw_numpy.BGR_from_HTML(color_HTML='#FF8000')[[2,1,0]],
            "red":tools_draw_numpy.BGR_from_HTML(color_HTML='#C00000')[[2,1,0]],
            "silver":tools_draw_numpy.BGR_from_HTML(color_HTML='#808080')[[2,1,0]],
            "white":tools_draw_numpy.BGR_from_HTML(color_HTML='#FFFFFF')[[2,1,0]],
            "yellow":tools_draw_numpy.BGR_from_HTML(color_HTML='#FFC000')[[2,1,0]],
            "nan":tools_draw_numpy.BGR_from_HTML(color_HTML='#FFFFFF')[[2,1,0]],
            "": tools_draw_numpy.BGR_from_HTML(color_HTML='#FFFFFF')[[2, 1, 0]]
        }
        return
# ----------------------------------------------------------------------------------------------------------------------
    def name_columns(self, df):
        if df is None or df.shape[0]==0:
            return pd.DataFrame([{'frame_id':[-1],'track_id':[-1],'x1':[0],'y1':[0],'x2':[0],'y2':[0],'conf':[0]}])
        cols = [c for c in df.columns]
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
    def init_model_LPVD(self):
        self.model_LPVD = YOLO('./models/MMR/LPVD_yolo8n_ReLU_21_640_041223.pt')
        self.model_LPVD.to(self.device)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_model_symb(self):
        self.model_symb = YOLO('./models/MMR/Symb_yolo8_relu_stacked_200_224_210324.pt')
        self.model_symb.to(self.device)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_model_LP_type(self):
        self.model_LP_type = torch.load('./models/MMR/yolov8n_LP_type_usa_rgb_v6.pt')['model']
        self.model_LP_type.eval()
        if self.device == 'cuda':
            self.model_LP_type = self.model_LP_type.half().cuda()
        else:
            self.model_LP_type = self.model_LP_type.float().cpu()
        self.dct_class_names_LP = self.model_LP_type.names
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_model_MMR(self):
        #providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),"user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
        self.ort_sess_MMR = InferenceSession('./models/MMR/MMR_2700&truck_35_224_180324.onnx')

        with open('./models/MMR/MMR_2700&truck_35_224_180324.json') as f:
            self.mmr_types_dct = json.load(f)
            self.mmr_types_dct = dict(zip(range(len(self.mmr_types_dct['make'])), self.mmr_types_dct['make']))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_model_color(self):
        self.ort_sess_color = InferenceSession('./models/MMR/color_resnet_relu_30_224_170324.onnx')
        with open('./models/MMR/color_resnet_relu_30_224_170324.json') as f:
            self.colors_dct = json.load(f)
            self.colors_dct = dict(zip(range(len(self.colors_dct['color'])), self.colors_dct['color']))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def preprocess_image(self, image,img_size = 640):
        img = Image.open(image).convert("RGB") if isinstance(image, str) else Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        res = transform(img).unsqueeze(0)
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def run_model_LPVD(self,image,do_debug=False):

        img_size = 640
        img = cv2.imread(image) if isinstance(image, str) else image
        limg = tools_image.smart_resize(img, target_image_height=img_size, target_image_width=img_size,bg_color=(128, 128, 128))
        res = self.model_LPVD.predict(source=limg,verbose=False, device=self.device)
        confs = 0
        if res[0].boxes is not None:
            rects = res[0].boxes.xyxy.cpu().numpy()
            class_ids = res[0].boxes.cls.cpu().numpy()
            dct_class_names = res[0].names
            confs = res[0].boxes.conf.cpu().numpy()

        image_LP = None
        conf_res = 0
        rect_LP = None
        idx = numpy.where(class_ids == 0)[0]
        if idx.shape[0]>0:
            rects = rects[idx]
            idx2 = numpy.argmax(confs)
            conf_res = max(confs[idx])
            rect_LP = rects[idx2]
            p1 = tools_image.smart_resize_point_inv(rect_LP[0], rect_LP[1], img.shape[1], img.shape[0],img_size, img_size)
            p2 = tools_image.smart_resize_point_inv(rect_LP[2], rect_LP[3], img.shape[1], img.shape[0], img_size, img_size)
            image_LP = img[p1[1]:p2[1],p1[0]:p2[0]]


        if do_debug:
            image_res = self.draw_detections(limg, rects, class_ids, confs, dct_class_names,rect_LP)
            cv2.imwrite(self.folder_out + 'model_LPVD.png', image_res)

        return image_LP,conf_res,rect_LP
# ----------------------------------------------------------------------------------------------------------------------
    def run_model_symb(self,image_LP,do_debug=False):

        result, cnf_res = '',0
        if image_LP is None:
            return result, cnf_res

        img_size = 640
        img = cv2.imread(image_LP) if isinstance(image_LP, str) else image_LP
        if img.shape[0]*img.shape[1] == 0:
            return result, cnf_res
        limg = tools_image.smart_resize(img, target_image_height=img_size, target_image_width=img_size,bg_color=(128, 128, 128))
        res = self.model_symb.predict(source=limg, verbose=False, device=self.device)

        if res[0].boxes is not None:
            rects = res[0].boxes.xyxy.cpu().numpy()
            class_ids = res[0].boxes.cls.cpu().numpy()
            dct_class_names = res[0].names
            confs = res[0].boxes.conf.cpu().numpy()

        result = ''.join([dct_class_names[i] for i in class_ids[numpy.argsort(rects[:,0])]])
        if rects.shape[0]>0:
            cnf_res = confs.mean()

        if do_debug:
            image_res = self.draw_detections(limg, rects, class_ids, confs, dct_class_names)
            cv2.imwrite(self.folder_out + 'model_LP_symb.png', image_res)

        return result,cnf_res
# ----------------------------------------------------------------------------------------------------------------------
    def run_model_LP_type(self,image):
        output = self.model_LP_type(self.preprocess_image(image).half().cuda())
        output = output.to('cpu').detach().numpy()
        return output.flatten()
# ----------------------------------------------------------------------------------------------------------------------
    def run_model_MMR(self,image):
        image = self.preprocess_image(image,224)
        output = self.ort_sess_MMR.run(None, {'input': image.numpy()})
        return output
# ----------------------------------------------------------------------------------------------------------------------
    def run_model_color(self,image):
        image = self.preprocess_image(image,224)
        output = self.ort_sess_color.run(None, {'input': image.numpy()})
        return output
# ----------------------------------------------------------------------------------------------------------------------
    def draw_features(self,image,rects,df_E):
        #colors = [self.colors80[track_id % 80] for track_id in track_ids]
        labels = [','.join([str(value) for name,value in zip(df_E.columns,df_E.iloc[r].values) if str(value)!='nan']) for r in range(rects.shape[0])]
        image = tools_draw_numpy.draw_rects(image, rects.reshape((-1,2,2)), colors=(0,0,200), labels=labels, w=2,alpha_transp=0.95)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_detections(self,image,rects,class_ids,confs,dct_class_names,rect_LP=None):
        colors = [self.colors80[i % 80] for i in range(len(rects))]
        labels = [dct_class_names[i] + ' %.2f' % conf for i, conf in zip(class_ids, confs)]
        image = tools_draw_numpy.draw_rects(image, rects.reshape(-1, 2, 2), colors, labels=labels, w=2,alpha_transp=0.8)
        if rect_LP is not None:
            image = tools_draw_numpy.draw_rects(image, rect_LP.reshape(-1, 2, 2), (128,255,0),w=2,alpha_transp=0.8)
        return  image
# ----------------------------------------------------------------------------------------------------------------------
    def get_features(self,filename_in,df_pred,frame_id=0,do_debug=False):
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in

        E = []
        for r in range(df_pred.shape[0]):
            rect = df_pred.iloc[r][['x1','y1','x2','y2']].values.astype(int)
            im = image[rect[1]:rect[3],rect[0]:rect[2]]
            model_color = numpy.argmax(self.run_model_color(im))
            model_color = self.colors_dct[model_color]

            mmr_out = self.run_model_MMR(im)
            mmr_type = numpy.argmax(mmr_out)
            mmr_type = self.mmr_types_dct[mmr_type]
            mmr_type = mmr_type.split('~')[1]
            conf_mmr = numpy.max(mmr_out)
            E.append([model_color, mmr_type,conf_mmr])

        df_E = pd.DataFrame(E, columns=['model_color', 'mmr_type','conf_mmr'])
        df_E = df_E.astype({'model_color': str, 'mmr_type': str,'conf_mmr': float})

        if do_debug:
            df_t = self.df_true[self.df_true['frame_id'] == frame_id]
            df_pred_local = df_pred.copy()
            df_pred_local['frame_id'] = frame_id
            df_pred_local['track_id'] = -1
            df_pred_local = df_pred_local[['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'conf']]
            df_true2, df_pred2 = self.B.calc_hits_stats_iou(df_t, df_pred_local, from_cache=False, verbose=False)

            image = self.draw_features(image,df_pred2[['x1','y1','x2','y2']].values,df_E)
            cv2.imwrite(self.folder_out + filename_in.split('/')[-1],image)

        return df_E
# ----------------------------------------------------------------------------------------------------------------------
    def get_features_full(self,filename_in,df_pred,frame_id=0,do_debug=False):
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in

        E = []
        for r in range(df_pred.shape[0]):
            rect = df_pred.iloc[r][['x1','y1','x2','y2']].values.astype(int)
            im = image[rect[1]:rect[3],rect[0]:rect[2]]
            model_color = numpy.argmax(self.run_model_color(im))
            model_color = self.colors_dct[model_color]

            mmr_out = self.run_model_MMR(im)
            mmr_type = numpy.argmax(mmr_out)
            mmr_type = self.mmr_types_dct[mmr_type]
            #mmr_type = ' '.join(mmr_type.split('~')[1:3])
            mmr_type = mmr_type.split('~')[1]
            conf_mmr = numpy.max(mmr_out)
            LP_image,confs_LP_det,rect_LP = self.run_model_LPVD(im)
            LP_symb,conf_LP_read = self.run_model_symb(LP_image, do_debug=False)
            LP_H, LP_W = LP_image.shape[:2] if LP_image is not None else (0,0)
            veh_H, veh_W = im.shape[:2] if im is not None else (0,0)
            E.append([model_color, mmr_type, LP_symb,veh_W,veh_H,LP_W,LP_H,conf_mmr,confs_LP_det,conf_LP_read])

        df_E = pd.DataFrame(E, columns=['model_color', 'mmr_type', 'lp_symb','v_W','v_H','lp_W','lp_H','conf_mmr','conf_LP_read','conf_LP_det'])
        df_E = df_E.astype({'model_color': str, 'mmr_type': str, 'lp_symb': str,'v_W':int,'v_H':int,'lp_W':int,'lp_H':int,'conf_mmr': float,'conf_LP_read': float,'conf_LP_det': float})

        if do_debug:
            df_t = self.df_true[self.df_true['frame_id'] == frame_id]
            df_pred_local = df_pred.copy()
            df_pred_local['frame_id'] = frame_id
            df_pred_local['track_id'] = -1
            df_pred_local = df_pred_local[['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'conf']]
            df_true2, df_pred2 = self.B.calc_hits_stats_iou(df_t, df_pred_local, from_cache=False, verbose=False)

            image = self.draw_features(image,df_pred2[['x1','y1','x2','y2']].values,df_E)
            cv2.imwrite(self.folder_out + filename_in.split('/')[-1],image)

        return df_E
# ----------------------------------------------------------------------------------------------------------------------