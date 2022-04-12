import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from PIL import ImageDraw,ImageFont,Image
import numpy
import cv2
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_time_profiler
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class detector_TF_Zoo(object):

    def __init__(self,folder_out):
        #self.detector = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1").signatures['default']
        self.detector = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures['default']
        self.folder_out = folder_out
        self.TP = tools_time_profiler.Time_Profiler()

        self.name = 'TF Zoo'
        self.whitelabels = ['Car']
        self.min_confidence_score = 0.2
        self.counter =0
        self.color_markup = tools_draw_numpy.color_red
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_image(self,image,filename_out=None):
        if isinstance(image,str):
            ID = image.split('/')[-1]
            image = cv2.imread(image)
        else:
            ID = self.counter

        img_converted = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
        result = self.detector(img_converted)
        result = {key: value.numpy() for key, value in result.items()}
        boxes, classes, scores = self.filter_results(result["detection_boxes"], result["detection_class_entities"],result["detection_scores"])
        self.counter+=1

        H,W = image.shape[:2]
        df_boxes = pd.DataFrame({'ID':ID,'score':scores,'class':classes,'x1':boxes[:,1]*W,'y1':boxes[:,0]*H,'x2':boxes[:,3]*W,'y2':boxes[:,2]*H})
        df_boxes[['x1','y1','x2','y2']] = df_boxes[['x1','y1','x2','y2']].astype(int)

        if filename_out is not None:
            cv2.imwrite(self.folder_out+filename_out,self.draw_all_boxes(image,boxes, classes, scores))

        return df_boxes
# ----------------------------------------------------------------------------------------------------------------------
    def filter_results(self,boxes, classes, scores):

        idx = scores >= self.min_confidence_score
        boxes, classes, scores = boxes[idx], classes[idx], scores[idx]
        classes = numpy.array([cls.decode("utf-8") for cls in classes])

        if len(self.whitelabels)>0 and classes.shape[0]>0:
            idx = numpy.array([cls in self.whitelabels for cls in classes])
            return boxes[idx],classes[idx], scores[idx]
        else:
            return boxes, classes, scores
# ----------------------------------------------------------------------------------------------------------------------
    def draw_all_boxes(self, image, boxes, class_names, scores, max_boxes=10):
        #font = ImageFont.load_default()
        for i in range(min(boxes.shape[0], max_boxes)):
            if scores[i] < self.min_confidence_score:continue
            image = tools_draw_numpy.draw_rect(image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],self.color_markup, w=1, alpha_transp=0.8,label='%d%%'%int(100 * scores[i]))
            #image_pil = Image.fromarray(numpy.uint8(image)).convert("RGB")
            #self.draw_box(image_pil, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], self.color_markup, font, display_str_list=['%d%%'%int(100 * scores[i])])
            #numpy.copyto(image, numpy.array(image_pil))
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_box(self, image, ymin, xmin, ymax, xmax, color, font, thickness=1, display_str_list=()):
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness,fill=color)
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
        text_bottom = top if top > total_display_str_height else top + total_display_str_height

        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = numpy.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],fill=color)
            draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
            text_bottom -= text_height - 2 * margin
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_folder(self, folder_in, list_of_masks='*.jpg',limit=1000000):
        tools_IO.remove_files(self.folder_out)
        self.counter = 0
        filenames  = tools_IO.get_filenames(folder_in, list_of_masks)[:limit]
        mode = 'w'
        for filename in filenames:
            print(filename)
            df_boxes = self.process_image(folder_in + filename,filename)
            if df_boxes.shape[0]>0:
                df_boxes.to_csv(self.folder_out+'df_boxes.csv',index=False,mode=mode,header=(mode=='w'))
                mode = 'a'

        return
# ----------------------------------------------------------------------------------------------------------------------
