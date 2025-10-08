import pandas as pd
import cv2
import onnxruntime as ort
import os
import numpy
import torch
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Detector_yolo:
    def __init__(self,folder_out,config=None):
        self.name = 'yolo'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.confidence_th = config.confidence_th if config is not None else None
        device_colored = ('\033[92m' + 'cuda' + '\033[0m' if self.device == 'cuda' else '\033[33m' + 'CPU' + '\033[0m')
        print('[Detectr] device:',device_colored,'-',config.detection_model)

        if (folder_out is not None) and (not os.path.isdir(folder_out)):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.session = ort.InferenceSession(config.detection_model)
        self.img_height, self.img_width = 640, 640
        self.dct_class_names = dict(zip(range(80), ['%d' % i for i in range(80)]))
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update_confidence_th(self,confidence_th):
        self.confidence_th = confidence_th
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update_config(self, config):
        self.config = config
        model_name = config.model_detect
        print('Using model:', model_name)
        self.session = ort.InferenceSession(config.detection_model)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def nms(self,rects, confidences, threshold):
        if len(rects) == 0:
            return []

        rects = numpy.array(rects, dtype=float)
        confidences = numpy.array(confidences, dtype=float)

        pick = []
        x1 = rects[:, 0]
        y1 = rects[:, 1]
        x2 = x1 + rects[:, 2]
        y2 = y1 + rects[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = numpy.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
            yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
            xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
            yy2 = numpy.minimum(y2[i], y2[idxs[:last]])
            w = numpy.maximum(0, xx2 - xx1 + 1)
            h = numpy.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap > threshold)[0])))

        return pick

    # ----------------------------------------------------------------------------------------------------------------------
    def rescale_boxes(self,boxes, input_shape, image_shape):
        boxes[:, 0] = boxes[:, 0] / (image_shape[1] / input_shape[1])
        boxes[:, 1] = boxes[:, 1] / (image_shape[0] / input_shape[0])
        boxes[:, 2] = boxes[:, 2] / (image_shape[1] / input_shape[1])
        boxes[:, 3] = boxes[:, 3] / (image_shape[0] / input_shape[0])
        return boxes

    # ----------------------------------------------------------------------------------------------------------------------
    def xywh2xyxy(self,x):
        y = numpy.copy(x)
        w, h = x[..., 2], x[..., 3]
        y[..., 0] = x[..., 0] - w / 2
        y[..., 1] = x[..., 1] - h / 2
        y[..., 2] = x[..., 0] + w / 2
        y[..., 3] = x[..., 1] + h / 2
        return y

    # ----------------------------------------------------------------------------------------------------------------------
    def extract_boxes(self,predictions, input_height, input_width, img_height, img_width):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes, (input_height, input_width), (img_height, img_width))
        boxes = self.xywh2xyxy(boxes)
        boxes[:, 0] = numpy.clip(boxes[:, 0], 0, img_width)
        boxes[:, 1] = numpy.clip(boxes[:, 1], 0, img_height)
        boxes[:, 2] = numpy.clip(boxes[:, 2], 0, img_width)
        boxes[:, 3] = numpy.clip(boxes[:, 3], 0, img_height)
        return boxes
    # ----------------------------------------------------------------------------------------------------------------------
    def process_output(self,output, input_height, input_width, img_height, img_width, conf_threshold=0.3, iou_threshold=0.5):

        predictions = numpy.squeeze(output[0]).T
        confidences = numpy.max(predictions[:, 4:], axis=1)
        predictions = predictions[confidences > conf_threshold, :]
        confidences = confidences[confidences > conf_threshold]

        if len(confidences) == 0:
            return [], [], []

        class_ids = numpy.argmax(predictions[:, 4:], axis=1)
        rects = self.extract_boxes(predictions, input_height, input_width, img_height, img_width)
        indices = self.nms(rects, confidences, iou_threshold)
        return rects[indices], confidences[indices], class_ids[indices]

    # ----------------------------------------------------------------------------------------------------------------------
    def get_detections(self,filename_in,do_debug=False):
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in
        df_pred = pd.DataFrame({'class_ids': [], 'class_name': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'conf': []})
        if image is None:return df_pred

        input_height, input_width = image.shape[:2]
        image_preproc = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (self.img_width, self.img_height),interpolation=cv2.INTER_LINEAR)
        image_preproc = numpy.expand_dims(image_preproc, axis=0).astype('float32') / 255.
        output = self.session.run(None, {self.session.get_inputs()[0].name: numpy.transpose(image_preproc, [0, 3, 1, 2])})
        rects, confs, class_ids = self.process_output(output, input_height, input_width, self.img_height, self.img_width)
        class_names = numpy.array([self.dct_class_names[i] for i in class_ids])

        df_pred = pd.DataFrame(numpy.concatenate((class_ids.reshape((-1, 1)), rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['class_ids', 'x1', 'y1', 'x2', 'y2', 'conf'])
        df_pred = df_pred.astype({'class_ids': int, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'conf': float})
        df_pred['class_name'] = class_names
        if self.confidence_th is not None:
            df_pred = df_pred[df_pred['conf'] >= self.confidence_th]

        return df_pred
# ----------------------------------------------------------------------------------------------------------------------
    def draw_detections(self,image,rects,class_ids,confs):
        colors = [self.colors80[i % 80] for i in range(len(rects))]
        labels = [self.dct_class_names[i] + ' %.2f' % conf for i, conf in zip(class_ids, confs)]
        image = tools_draw_numpy.draw_rects(image, rects.reshape(-1, 2, 2), colors, labels=labels, w=2,alpha_transp=0.8)
        return  image
    # ----------------------------------------------------------------------------------------------------------------------