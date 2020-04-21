# ----------------------------------------------------------------------------------------------------------------------
import numpy
import cv2
import os
import time
import progressbar
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_image
import tools_IO
import detector_YOLO3_core
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
# ----------------------------------------------------------------------------------------------------------------------
def get_COCO_class_names():
    return ['person','bicycle','car','motorbike',
        'aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
        'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
        'hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
        'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
# ----------------------------------------------------------------------------------------------------------------------
def generate_colors(N=80):
    numpy.random.seed(42)
    colors = []
    for i in range(0, N):
        if i==0:
            hue = 0
        else:
            hue = int(255 * numpy.random.rand())
        color = cv2.cvtColor(numpy.array([hue, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3),cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))


    numpy.random.seed(None)

    return colors
# ----------------------------------------------------------------------------------------------------------------------
def draw_classes_on_image(image, boxes_yxyx, scores, color,draw_score=False):

    for box, score in zip(boxes_yxyx,scores):
        top, left, bottom, right = box
        top = max(0, numpy.floor(top + 0.5).astype('int32'))
        left = max(0, numpy.floor(left + 0.5).astype('int32'))
        bottom = min(image.shape[0], numpy.floor(bottom + 0.5).astype('int32'))
        right = min(image.shape[1], numpy.floor(right + 0.5).astype('int32'))

        image = tools_draw_numpy.draw_rect(image, top, left, bottom,right,  color, alpha_transp=1.0-score)
        if draw_score==True:
            position = top - 6 if top - 6 > 10 else top + 26
            cv2.putText(image, '{0:.2f}'.format(score), (left + 4, position), cv2.FONT_HERSHEY_SIMPLEX,0.4, color, 1, cv2.LINE_AA)

    return image
# ----------------------------------------------------------------------------------------------------------------------
def draw_objects_on_image(image, boxes_bound, scores, classes, colors, class_names):

    if boxes_bound is None:
        return image

    for box, score, cl in zip(boxes_bound, scores, classes):
        #if class_names[cl]!='car':continue
        top, left, bottom,right = box

        top     = max(0, numpy.floor(top + 0.5).astype('int32'))
        left    = max(0, numpy.floor(left + 0.5).astype('int32'))
        bottom  = min(image.shape[0], numpy.floor(bottom + 0.5).astype('int32'))
        right   = min(image.shape[1], numpy.floor(right + 0.5).astype('int32'))

        color = colors[cl]

        cv2.rectangle(image, (left,top), (right,bottom), color.tolist(), 2)
        position = top - 6 if top - 6 > 10 else top + 26
        #cv2.putText(image, '{0} {1:.2f}'.format(class_names[cl], score), (left+4, position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        #cv2.putText(image, '{0}'.format(class_names[cl]), (left+4, position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return image
# ----------------------------------------------------------------------------------------------------------------------
def get_true_boxes(foldername, filename, smart_resized_target=None, delim=' ',limit=1000000):



    with open(filename) as f:lines = f.readlines()[1:limit]
    filenames_dict = sorted(set([line.split(delim)[0] for line in lines]))

    true_boxes = []

    print('\nGet %d true boxes\n' % len(filenames_dict))
    bar = progressbar.ProgressBar(max_value=len(filenames_dict))

    for b,filename in enumerate(filenames_dict):
        bar.update(b)
        if not os.path.isfile(foldername + filename):continue
        image = Image.open(foldername + filename)
        if image is None: continue
        width, height = image.size

        local_boxes = []
        for line in lines:
            split = line.split(delim)
            if split[0]==filename:
                class_ID = int(split[5])
                x_min, y_min, x_max, y_max = numpy.array(split[1:5]).astype(numpy.float)

                if smart_resized_target is not None:
                    x_min, y_min = tools_image.smart_resize_point(x_min, y_min, width,height,smart_resized_target[1], smart_resized_target[0])
                    x_max, y_max = tools_image.smart_resize_point(x_max, y_max, width,height,smart_resized_target[1], smart_resized_target[0])

                local_boxes.append([x_min, y_min, x_max, y_max, class_ID])

        true_boxes.append(local_boxes)

    return true_boxes
# ----------------------------------------------------------------------------------------------------------------------
def get_images(foldername, filename, delim=' ', smart_resized_target=None,limit=10000):

    with open(filename) as f:lines = f.readlines()[1:limit]
    list_filenames = [line.split(' ')[0] for line in lines]
    filenames_dict = sorted(set(list_filenames))

    images = []

    for filename in filenames_dict:
        image = tools_image.rgb2bgr(cv2.imread(foldername + filename))
        if smart_resized_target is not None:
            image = tools_image.smart_resize(image,smart_resized_target[0],smart_resized_target[1])

        images.append(image)

    return numpy.array(images)
# ----------------------------------------------------------------------------------------------------------------------
def draw_annotation_boxes(file_annotations, file_classes, file_metadata, path_out,delim=' '):

    tools_IO.remove_files(path_out,create=True)

    input_image_size, class_names, anchors, anchor_mask, obj_threshold, nms_threshold = load_metadata(file_metadata)
    mat = tools_IO.load_mat(file_classes, numpy.str)
    if len(mat)<=len(class_names):
        class_names[:len(mat)] = mat

    foldername = '/'.join(file_annotations.split('/')[:-1]) + '/'
    with open(file_annotations) as f:lines = f.readlines()[1:]
    boxes_xyxy  = numpy.array([line.split(delim)[1:5] for line in lines],dtype=numpy.int)
    filenames = numpy.array([line.split(delim)[0] for line in lines])
    class_IDs = numpy.array([line.split(delim)[5] for line in lines],dtype=numpy.int)
    colors = generate_colors(numpy.max(class_IDs)+1)

    true_boxes = get_true_boxes(foldername, file_annotations, (416, 416), delim=' ')
    if len(true_boxes) > 6:
        anchors = annotation_boxes_to_ancors(true_boxes, 6)

    descript_ion = []
    for filename in set(filenames):

        image = cv2.imread(foldername + filename)
        image = tools_image.desaturate(image,0.9)
        idx = numpy.where(filenames == filename)

        boxes_resized = []
        for box in boxes_xyxy[idx]:
            x_min, y_min = tools_image.smart_resize_point(box[0], box[1], image.shape[1], image.shape[0], 416, 416)
            x_max, y_max = tools_image.smart_resize_point(box[2], box[3], image.shape[1], image.shape[0], 416, 416)
            boxes_resized.append([x_min, y_min, x_max, y_max, 0])

        statuses = are_boxes_preprocessed_well(boxes_resized,anchors,anchor_mask,len(class_names))
        descript_ion.append([filename, 1*(statuses.sum() == len(statuses))])

        for box,class_ID,status in zip(boxes_xyxy[idx],class_IDs[idx],statuses):
            w = 2 if status>0 else -1
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[class_ID],thickness=w)
            cv2.putText(image, '{0:d} {1:s}'.format(class_ID, class_names[class_ID]), (box[0], box[1] - 4),cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_ID], 1, cv2.LINE_AA)

        cv2.imwrite(path_out + filename, image)


    tools_IO.save_mat(descript_ion,path_out+'descript.ion', delim=' ')
    return
# ----------------------------------------------------------------------------------------------------------------------
def are_boxes_preprocessed_well(boxes,anchors,anchor_mask,num_classes):

    statuses = []
    for box in boxes:
        true_boxes = numpy.array([[box]])
        y = detector_YOLO3_core.preprocess_true_boxes(true_boxes, (416,416), anchors,anchor_mask, num_classes)
        status = ((y[0].max() > 0) or (y[1].max() > 0))
        statuses.append(status)

    return numpy.array(statuses)
# ----------------------------------------------------------------------------------------------------------------------
def filter_dups(boxes,classes,scores,nms_threshold):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = numpy.where(classes == c)
        b, c, s = boxes[inds], classes[inds], scores[inds]
        keep = nms_boxes(b, s, nms_threshold)
        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return [], [], []

    nboxes = numpy.concatenate(nboxes)
    nclasses = numpy.concatenate(nclasses)
    nscores = numpy.concatenate(nscores)

    #keep = nms_boxes(nboxes, nscores, nms_threshold)
    #nboxes=nboxes[keep]
    #nclasses=nclasses[keep]
    #nscores=nscores[keep]

    return nboxes,nclasses,nscores
# ----------------------------------------------------------------------------------------------------------------------
def nms_boxes(boxes, scores,nms_threshold):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = numpy.maximum(x[i], x[order[1:]])
        yy1 = numpy.maximum(y[i], y[order[1:]])
        xx2 = numpy.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = numpy.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = numpy.maximum(0.0, xx2 - xx1 + 1)
        h1 = numpy.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = numpy.where(ovr <= nms_threshold)[0]
        order = order[inds + 1]

    keep = numpy.array(keep)

    return keep
# ----------------------------------------------------------------------------------------------------------------------
def draw_and_save(filename_out,image,boxes_yxyx, scores,classes,colors, class_names):
    if filename_out is not None:
        res_image = draw_objects_on_image(tools_image.desaturate(image, 1.0), boxes_yxyx, scores,classes, colors, class_names)
        cv2.imwrite(filename_out, res_image)
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_markup(filename_in,boxes_yxyx,scores,classes):

    markup = []
    for i, each in enumerate(scores):
        markup.append([filename_in.split('/')[-1], int(boxes_yxyx[i][1]), int(boxes_yxyx[i][0]), int(boxes_yxyx[i][3]),int(boxes_yxyx[i][2]), classes[i], scores[i]])
    return markup
# ----------------------------------------------------------------------------------------------------------------------
def XML_indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            XML_indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
# ----------------------------------------------------------------------------------------------------------------------
def init_default_metadata_tiny(num_classes):
    input_image_size = (416,416)
    if num_classes<=80:
        class_names = get_COCO_class_names()[:num_classes]
    else:
        class_names = ['class_'+str(i) for i in range(num_classes)]

    anchors = numpy.array([[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]).astype(numpy.float)
    anchor_mask = [[3, 4, 5], [1, 2, 3]]
    obj_threshold = 0.5
    nms_threshold = 0.5
    return input_image_size, class_names, anchors, anchor_mask,obj_threshold, nms_threshold
# ----------------------------------------------------------------------------------------------------------------------
def init_default_metadata_full(num_classes):
    input_image_size = (416,416)
    if num_classes<=80:
        class_names = get_COCO_class_names()[:num_classes]
    else:
        class_names = ['class_'+str(i) for i in range(num_classes)]

    anchors = numpy.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]).astype(numpy.float)
    anchor_mask = [[6,7,8],[3, 4, 5], [0, 1, 2]]
    obj_threshold = 0.5
    nms_threshold = 0.5
    return input_image_size, class_names, anchors, anchor_mask,obj_threshold, nms_threshold
# ----------------------------------------------------------------------------------------------------------------------
def save_metadata(filename_out,input_image_size, class_names, anchors,anchor_mask, obj_threshold, nms_threshold):

    model = ET.Element("model")

    ET.SubElement(model, 'input_image_size').text = str(input_image_size)
    ET.SubElement(model, 'obj_threshold').text = str(obj_threshold)
    ET.SubElement(model, 'nms_threshold').text = str(nms_threshold)

    xml_anchors = ET.SubElement(model, "anchors")
    for i,each in enumerate(anchors):
        ET.SubElement(xml_anchors, 'A'+str(i)).text = str((each[0],each[1]))

    xml_anchor_mask = ET.SubElement(model, "anchor_mask")
    for i,each in enumerate(anchor_mask):
        ET.SubElement(xml_anchor_mask, 'M'+str(i)).text = str((each[0],each[1],each[2]))

    xml_class_names = ET.SubElement(model, "class_names")
    for i,each in enumerate(class_names):
        ET.SubElement(xml_class_names, 'C'+str(i)).text = str(each)


    XML_indent(model)
    tree = ET.ElementTree(model)
    tree.write(filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def load_metadata(filename_input):
    root = ET.parse(filename_input).getroot()

    input_image_size, class_names, anchors,anchors_mask, obj_threshold, nms_threshold = None,[],[],[],0,0

    for each in root.findall('obj_threshold'):obj_threshold = float(each.text)
    for each in root.findall('nms_threshold'):nms_threshold = float(each.text)
    for each in root.findall('input_image_size'):input_image_size = (int(each.text.split(',')[0].split('(')[-1]),int(each.text.split(',')[1].split(')')[0]))
    for each in root.findall('anchors')[0]:anchors.append((float(each.text.split(',')[0].split('(')[-1]),float(each.text.split(',')[1].split(')')[0])))
    anchors = numpy.array(anchors,dtype=numpy.float)

    for each in root.findall('anchor_mask')[0]:
        v0 = int(each.text.split(',')[0].split('(')[-1])
        v1 = int(each.text.split(',')[1].split(',')[ 0])
        v2 = int(each.text.split(',')[2].split(')')[0])
        anchors_mask.append([v0,v1,v2])

    for each in root.findall('class_names')[0]:class_names.append(each.text)

    return input_image_size, class_names, anchors, anchors_mask, obj_threshold, nms_threshold
# ----------------------------------------------------------------------------------------------------------------------
def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = numpy.minimum(clusters[:, 0], box[0])
    y = numpy.minimum(clusters[:, 1], box[1])
    if numpy.count_nonzero(x == 0) > 0 or numpy.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


# ----------------------------------------------------------------------------------------------------------------------
def kmeans(boxes, k, dist=numpy.median, seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances     = numpy.empty((rows, k)) ## N row x N cluster
    last_clusters = numpy.zeros((rows,))

    numpy.random.seed(seed)

    # initialize the cluster centers to be k items
    clusters = boxes[numpy.random.choice(rows, k, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(k): # I made change to lars76's code here to make the code faster
            distances[:,icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = numpy.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances
# ----------------------------------------------------------------------------------------------------------------------
def plot_cluster_result(clusters,nearest_clusters,wh,k):

    for icluster in numpy.unique(nearest_clusters):
        pick = nearest_clusters==icluster
        plt.rc('font', size=8)
        plt.plot(wh[pick,0], wh[pick,1],"o",alpha=0.5, label="cluster = {}, N = {:6.0f}".format(icluster, numpy.sum(pick)))
        plt.xlabel("width")
        plt.ylabel("height")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("./kmeans.jpg")
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
def annotation_boxes_to_ancors(list_of_boxes,num_clusters):

    w,h=[],[]

    for boxes in list_of_boxes:
        for box in boxes:
            if (box[2] - box[0])*(box[3] - box[1])>0:
                w.append(box[2] - box[0])
                h.append(box[3] - box[1])

    wh = numpy.vstack((w,h)).T

    clusters, nearest_clusters, distances = kmeans(wh, num_clusters)
    area = clusters[:, 0] * clusters[:, 1]
    clusters = clusters[numpy.argsort(area)]
    clusters = clusters.astype(numpy.float)

    #plot_cluster_result(clusters, nearest_clusters, wh, num_clusters)

    return clusters
# ----------------------------------------------------------------------------------------------------------------------
def annotation_boxes_to_ancors2(list_of_boxes,num_clusters,delim=' '):

    w,h=[],[]

    for boxes in list_of_boxes:
        for box in boxes:
            w.append(box[2] - box[0])
            h.append(box[3] - box[1])

    minW,maxW = numpy.min(w),numpy.max(w)
    stepW = (maxW - minW )/ (num_clusters+1)
    aW = numpy.arange(minW+stepW,maxW,stepW)[:num_clusters]

    minH,maxH = numpy.min(h),numpy.max(h)
    stepH = (maxH - minH )/ (num_clusters+1)
    aH = numpy.arange(minH+stepH,maxH,stepH)[:num_clusters]

    clusters = [numpy.array([eachW,eachH]) for eachW,eachH in zip(aW,aH)]
    clusters = numpy.array(clusters)

    return clusters
# ----------------------------------------------------------------------------------------------------------------------


