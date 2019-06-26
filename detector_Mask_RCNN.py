# ----------------------------------------------------------------------------------------------------------------------
import time
import numpy
import tensorflow as tf
import cv2
import os
from os import listdir
import fnmatch
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_YOLO
import tools_IO
import detector_Mask_RCNN_core
# ----------------------------------------------------------------------------------------------------------------------
class InferenceConfig(detector_Mask_RCNN_core.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # ----------------------------------------------------------------------------------------------------------------------
class detector_Mask_RCNN(object):

    def __init__(self,filename_weights,path_root,obj_threshold=0.5):
        config = InferenceConfig()
        self.model = detector_Mask_RCNN_core.MaskRCNN(mode="inference", model_dir=path_root, config=config)
        self.model.load_weights(filename_weights, by_name=True)
        self.class_names = ['BG']+tools_YOLO.get_COCO_class_names()
        self.colors = tools_YOLO.generate_colors(len(self.class_names) - 1)
        self.colors.insert(0, self.colors[-1])
        self.obj_threshold = obj_threshold

        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_image(self,image):

        r = self.model.detect([image], verbose=0)[0]
        boxes_yxyx, scores, classes = r['rois'],r['scores'],r['class_ids']

        msk = scores>self.obj_threshold

        boxes_yxyx=boxes_yxyx[msk]
        scores = scores[msk]
        classes = classes[msk]
        return boxes_yxyx, scores, classes
# ----------------------------------------------------------------------------------------------------------------------
    def process_file(self, filename_in, filename_out,draw_spline=True):

        if not os.path.isfile(filename_in):
            return []

        image = cv2.imread(filename_in)
        image = tools_image.rgb2bgr(image)
        if image is None:
            return []

        if draw_spline==False:
            boxes_yxyx, scores, classes  = self.process_image(image)
            markup = tools_YOLO.get_markup(filename_in, boxes_yxyx, scores, classes)
            tools_YOLO.draw_and_save(filename_out, image, boxes_yxyx, scores, classes,self.colors, self.class_names)
        else:
            r = self.model.detect([image], verbose=0)[0]
            boxes_yxyx, scores, classes,masks = r['rois'], r['scores'], r['class_ids'],r['masks']
            class_ID = tools_IO.smart_index(self.class_names,'person')[0]
            msk1 = (scores > self.obj_threshold)
            msk2 = (classes == class_ID)
            msk = msk1.tolist() and msk2.tolist()
            msk = numpy.array(msk)
            boxes_yxyx, scores, classes, masks = boxes_yxyx[msk], scores[msk], classes[msk], masks[:,:,msk]
            markup = tools_YOLO.get_markup(filename_in, boxes_yxyx, scores, classes)

            detector_Mask_RCNN_core.display_instances(tools_image.desaturate(image), boxes_yxyx, masks, classes, self.class_names,filename_out,scores,self.colors)

        return markup
# ----------------------------------------------------------------------------------------------------------------------
    def process_folder(self, path_input, path_out, mask='*.jpg', limit=1000000,markup_only=False):
        tools_IO.remove_files(path_out)
        start_time = time.time()
        local_filenames = numpy.array(fnmatch.filter(listdir(path_input), mask))[:limit]
        result = [('filename', 'x_right','y_top','x_left','y_bottom','class_ID','confidence')]
        local_filenames = numpy.sort(local_filenames)
        for local_filename in local_filenames:
            filename_out = path_out + local_filename if not markup_only else None
            for each in self.process_file(path_input + local_filename, filename_out):
                result.append(each)
            tools_IO.save_mat(result, path_out + 'markup_res.txt', delim=' ')
        total_time = (time.time() - start_time)
        print('Processing: %s sec in total - %f per image' % (total_time, int(total_time) / len(local_filenames)))
        return
# ----------------------------------------------------------------------------------------------------------------------