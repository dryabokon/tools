import numpy
import  cv2
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
import tools_YOLO
from keras.layers import Input
# ----------------------------------------------------------------------------------------------------------------------
def generate_colors(N):
    return tools_YOLO.generate_colors(N)
# ----------------------------------------------------------------------------------------------------------------------
def get_markup(filename_in,boxes_yxyx,scores,classes):
    return tools_YOLO.get_markup(filename_in,boxes_yxyx,scores,classes)
# ----------------------------------------------------------------------------------------------------------------------
def draw_and_save(filename_out,image,boxes_yxyx, scores,classes,colors, class_names):
    return tools_YOLO.draw_and_save(filename_out,image,boxes_yxyx, scores,classes,colors, class_names)
# ----------------------------------------------------------------------------------------------------------------------
def get_true_boxes(filename, delim=' ', limit=10000):

    with open(filename) as f:lines = f.readlines()[1:limit]
    list_filenames = [line.split(' ')[0] for line in lines]
    filenames_dict = sorted(set(list_filenames))

    true_boxes = []

    for filename in filenames_dict:
        local_boxes = []
        for line in lines:
            split = line.split(delim)
            if split[0]==filename:
                class_ID = int(split[5])
                x_min, y_min, x_max, y_max = numpy.array(split[1:5]).astype(numpy.float)

                local_boxes.append([class_ID,x_min, y_min, x_max, y_max])

        true_boxes.append(numpy.array(local_boxes))

    return true_boxes
# ----------------------------------------------------------------------------------------------------------------------
def get_images(foldername, filename, delim=' ', resized_target=None,limit=10000):

    with open(filename) as f:lines = f.readlines()[1:limit]
    list_filenames = [line.split(' ')[0] for line in lines]
    filenames_dict = sorted(set(list_filenames))

    images = []

    for filename in filenames_dict:
        image = tools_image.rgb2bgr(cv2.imread(foldername + filename))
        if resized_target is not None:
            image = cv2.resize(image, resized_target)

        images.append(image)

    return numpy.array(images)
# ----------------------------------------------------------------------------------------------------------------------
