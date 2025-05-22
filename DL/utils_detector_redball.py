import pandas as pd
import cv2
import os
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Detector_RedBall:
    def __init__(self,folder_out):

        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.dct_class_names = {0: 'red_ball'}
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_detections(self,filename_in, col_start=None,do_debug=False):

        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = numpy.array([0, 120, 70])
        upper_red1 = numpy.array([10, 255, 255])
        lower_red2 = numpy.array([170, 120, 70])
        upper_red2 = numpy.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, numpy.ones((5, 5), numpy.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rects_xywh = numpy.array([cv2.boundingRect(contour) for contour in contours])
        rects = numpy.array([[rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]] for rect in rects_xywh])

        class_ids = numpy.zeros(len(contours), dtype=int)
        confs = numpy.ones(len(contours), dtype=float)
        class_names = numpy.array([self.dct_class_names[i] for i in class_ids])

        if do_debug and isinstance(filename_in, str):
            image_res = tools_image.desaturate(image)
            image_res = self.draw_detections(image_res,rects,class_ids, confs)
            cv2.imwrite(self.folder_out + filename_in.split('/')[-1], image_res)

        df_pred = pd.DataFrame(numpy.concatenate((class_ids.reshape((-1, 1)),rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['class_ids',             'x1', 'y1', 'x2', 'y2', 'conf'])
        df_pred = df_pred.astype({'class_ids': int, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'conf': float})
        df_pred['class_name'] = class_names


        return df_pred
# ----------------------------------------------------------------------------------------------------------------------
    def draw_detections(self,image,rects,class_ids,confs):
        colors = [self.colors80[i % 80] for i in range(len(rects))]
        #labels = [self.dct_class_names[i] + '%.2f' % conf for i, conf in zip(class_ids, confs)]
        labels = None

        image = tools_draw_numpy.draw_rects(image, rects.reshape(-1, 2, 2), colors, labels=labels, w=2,alpha_transp=0.8)
        rects = rects.reshape((-1, 2))
        image[int(rects[:,1].mean()),:] = (0,0,128)
        image[:,int(rects[:,0].mean())] = (0,0,128)

        return  image
# ----------------------------------------------------------------------------------------------------------------------