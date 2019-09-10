# ----------------------------------------------------------------------------------------------------------------------
import numpy
import cv2
import os
from os import listdir
import fnmatch
import time
# ----------------------------------------------------------------------------------------------------------------------
import tools_SSD
import detector_SSD300_core
import tools_image
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from keras import backend as K

from keras.layers import Lambda, Input
from keras.callbacks import EarlyStopping
from keras.models import Model, load_model
# ----------------------------------------------------------------------------------------------------------------------
class Logger(object):
    def __init__(self):
        self.data_source = None
        self.time_train = None
        self.time_validate = None
        self.time_test = None
        self.AP_train = None
        self.AP_test = None

# ----------------------------------------------------------------------------------------------------------------------
class detector_SSD300(object):

    def __init__(self, model_weights_h5,filename_metadata,obj_threshold=None):

        self.class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']

        self.colors = tools_SSD.generate_colors(len(self.class_names))


        self.input_image_size = (300, 300)
        self.nms_threshold = 0.5

        aspect_ratios = [[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]

        K.clear_session()
        self.model = detector_SSD300_core.ssd_300(image_size=(self.input_image_size[0], self.input_image_size[1], 3),
                        n_classes=len(self.class_names)-1,
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                        aspect_ratios_per_layer=aspect_ratios,
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels= [2, 1, 0],
                        confidence_thresh=0.01,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)

        self.model.load_weights(model_weights_h5, by_name=True)
        ssd_loss = detector_SSD300_core.SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile(optimizer='adam', loss=ssd_loss.compute_loss)

        self.logs = Logger()
# ----------------------------------------------------------------------------------------------------------------------
    def save_model(self,filename_out):
        self.model.save(filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_image(self, image):

        image_resized = cv2.resize(image, (self.input_image_size[0], self.input_image_size[1]))

        predictions = self.model.predict(numpy.array([image_resized]))

        predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)

        decoded_predictions = detector_SSD300_core.DecodeDetections(confidence_thresh=0.01,
                                               iou_threshold=self.nms_threshold,
                                               top_k=200,
                                               nms_max_output_size=400,
                                               coords='centroids',
                                               normalize_coords=True,
                                               img_height=self.input_image_size[0],
                                               img_width=self.input_image_size[1],
                                               name='decoded_predictions')(predictions)

        y_pred = decoded_predictions.eval(session=K.get_session())
        y_pred_thresh = numpy.array([y_pred[k][y_pred[k, :, 1] > 0.5] for k in range(y_pred.shape[0])])[0]
        classes = numpy.array(y_pred_thresh[:, 0], dtype=numpy.int)
        scores = y_pred_thresh[:, 1]
        boxes_yxyx = numpy.array([y_pred_thresh[:, 3], y_pred_thresh[:, 2], y_pred_thresh[:, 5], y_pred_thresh[:, 4]]).T
        boxes_yxyx[:, [0, 2]] *= float(image.shape[0] / 300.0)
        boxes_yxyx[:, [1, 3]] *= float(image.shape[1] / 300.0)


        return boxes_yxyx, classes, scores
# ----------------------------------------------------------------------------------------------------------------------
    def process_file(self, filename_in, filename_out):

        image = cv2.imread(filename_in)
        boxes_yxyx, classes, scores  = self.process_image(image)

        tools_SSD.draw_and_save(filename_out, image, boxes_yxyx, scores, classes,self.colors, self.class_names)
        markup = tools_SSD.get_markup(filename_in,boxes_yxyx,scores,classes)

        return markup
# ----------------------------------------------------------------------------------------------------------------------
    def process_folder(self, path_input, path_out, mask='*.jpg', limit=1000000,markup_only=False):
        tools_IO.remove_files(path_out)
        start_time = time.time()
        local_filenames = numpy.array(fnmatch.filter(listdir(path_input), mask))[:limit]
        result = [('filename', 'x_left','y_top','x_right','y_bottom','class_ID','confidence')]
        local_filenames = numpy.sort(local_filenames)
        for local_filename in local_filenames:
            filename_out = path_out + local_filename if not markup_only else None
            for each in self.process_file(path_input + local_filename, filename_out):
                result.append(each)
            tools_IO.save_mat(result, path_out + 'markup_res.txt', delim=' ')
        total_time = (time.time() - start_time)
        print('%s sec in total - %f per image\n\n' % (total_time, int(total_time) / len(local_filenames)))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_annotation(self, file_annotations, path_out, folder_annotation=None, markup_only=False):

        start_time = time.time()
        if folder_annotation is None:
            foldername = '/'.join(file_annotations.split('/')[:-1]) + '/'
        else:
            foldername = folder_annotation

        filename_markup_out = file_annotations.split('/')[-1].split('.')[0]+'_pred.txt'
        filename_markup_copy = path_out+file_annotations.split('/')[-1]
        if file_annotations!= filename_markup_copy:
            tools_IO.copyfile(file_annotations,filename_markup_copy)

        with open(file_annotations) as f:lines = f.readlines()[1:]
        local_filenames = [line.split(' ')[0] for line in lines]
        local_filenames = sorted(set(local_filenames))

        result = [('filename', 'x_left', 'y_top', 'x_right', 'y_bottom', 'class_ID', 'confidence')]
        for local_filename in local_filenames:
            filename_image_out = path_out + local_filename if not markup_only else None
            for each in self.process_file(foldername+local_filename, filename_image_out):
                result.append(each)
            tools_IO.save_mat(result, path_out + filename_markup_out, delim=' ')

        total_time = (time.time() - start_time)
        print('%s sec in total - %f per image\n\n' % (total_time, int(total_time) / len(local_filenames)))

        return

# ----------------------------------------------------------------------------------------------------------------------
    def save_default_model_and_metadata(self,default_model_weights,num_classes,filename_model_out,filename_metadata_out):
        #out_model = detector_YOLO3_core.yolo_body_tiny(Input(shape=(None, None, 3)), 3, num_classes)
        #default_model = load_model(default_model_weights, compile=False)
        #detector_YOLO3_core.assign_weights(default_model, out_model)
        #out_model.save(filename_model_out)
        #input_image_size, class_names, anchors, anchor_mask, obj_threshold, nms_threshold = tools_YOLO.init_default_metadata_tiny(num_classes)
        #tools_YOLO.save_metadata(filename_metadata_out, input_image_size, class_names, anchors, anchor_mask,obj_threshold, nms_threshold)
        return
# ----------------------------------------------------------------------------------------------------------------------
