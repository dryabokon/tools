# ----------------------------------------------------------------------------------------------------------------------
import time
import numpy
import tensorflow as tf
import cv2
import os
from os import listdir
import fnmatch
# ----------------------------------------------------------------------------------------------------------------------
import tools_YOLO
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
class detector_SSD_TF(object):

    def __init__(self, filename_topology,obj_threshold=0.5):
        self.obj_threshold=obj_threshold

        self.class_names = ['BG']+tools_YOLO.get_COCO_class_names()
        self.colors = tools_YOLO.generate_colors(len(self.class_names)-1)
        self.colors.insert(0,self.colors[-1])

        with tf.gfile.FastGFile(filename_topology, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_image(self,image):

        boxes_yxyx, scores, classes = [],[],[]

        with tf.Session() as sess:
            sess.graph.as_default()
            tf.import_graph_def(self.graph_def, name='')
            rows = image.shape[0]
            cols = image.shape[1]
            inp = image[:, :, [2, 1, 0]]  # BGR2RGB

            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})


            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]

                if score > self.obj_threshold:
                    boxes_yxyx.append([bbox[0] * rows,bbox[1] * cols, bbox[2] * rows,bbox[3] * cols])
                    scores.append(score)
                    classes.append(classId)

        return boxes_yxyx, scores, classes
# ----------------------------------------------------------------------------------------------------------------------
    def process_file(self, filename_in, filename_out):

        if not os.path.isfile(filename_in):
            return []

        image = cv2.imread(filename_in)
        if image is None:
            return []

        boxes_yxyx, scores, classes  = self.process_image(image)

        tools_YOLO.draw_and_save(filename_out, image, boxes_yxyx, scores, classes,self.colors, self.class_names)
        markup = tools_YOLO.get_markup(filename_in,boxes_yxyx,scores,classes)

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
        print('Processing: %s sec in total - %f per image' % (total_time, int(total_time) / len(local_filenames)))
        return
# ----------------------------------------------------------------------------------------------------------------------