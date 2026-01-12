import numpy
import cv2
import os
import pandas as pd
from pycocotools.coco import COCO
from tqdm import tqdm
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
def draw_contours_coco(filaname_coco_annnotation,alpha=0.75):
    coco = COCO(filaname_coco_annnotation)
    colors = tools_draw_numpy.get_colors(1 + len(coco.cats))
    class_names = [coco.cats[key]['name'] for key in coco.cats]

    for key in coco.imgToAnns.keys():
        annotations = coco.imgToAnns[key]
        image_id = annotations[0]['image_id']
        filename = coco.imgs[image_id]['file_name']
        if not os.path.isfile(self.folder_images + filename):
            continue

        result = tools_image.desaturate(cv2.imread(self.folder_images + filename))

        for annotation in annotations:
            contour = numpy.array(annotation['segmentation']).reshape((-1,2))
            category_IDs = annotation['category_id']
            result = tools_draw_numpy.draw_contours_cv(result, contour, colors[category_IDs],w=2,transperency=alpha)
            #result = tools_draw_numpy.draw_contours_cv(result, contour, colors[category_IDs],w=2,transperency=1.0)

        cv2.imwrite(self.folder_out + filename, result)
    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_boxes_coco(folder_images,filaname_coco_annnotation,folder_out,draw_binary_masks=False):
    tools_IO.remove_files(folder_out,'*.jpg')

    coco = COCO(filaname_coco_annnotation)
    colors80 = tools_draw_numpy.get_colors(1 + len(coco.cats))

    dct_class_names = {}
    for k in coco.cats.keys():
        dct_class_names[k] = coco.cats[k]['name']

    for key in coco.imgToAnns.keys():

        annotations = coco.imgToAnns[key]
        image_id = annotations[0]['image_id']
        filename = coco.imgs[image_id]['file_name']
        if not os.path.isfile(folder_images + filename):
            continue

        boxes, labels,colors =[],[], []
        for annotation in annotations:
            cat_id = annotation['category_id']
            if cat_id>=len(coco.cats):continue
            bbox = annotation['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(dct_class_names[cat_id])
            colors.append(colors80[cat_id])

        if draw_binary_masks:
            image = cv2.imread(folder_images + filename)
            result  = numpy.zeros_like(image)
            for box in boxes:
                top, left, bottom, right = box
                cv2.rectangle(result, (left, top), (right, bottom), (255,255,255), thickness=-1)
                cv2.imwrite(folder_out + filename.split('.')[0]+'.jpg', image)
        else:
            result = tools_draw_numpy.draw_rects(tools_image.desaturate(cv2.imread(folder_images + filename)), numpy.array(boxes).reshape((-1,2,2)), colors,labels=labels, w=2, alpha_transp=0.75)

        cv2.imwrite(folder_out + filename, result)
    return
# ----------------------------------------------------------------------------------------------------------------------
def coco_to_flat(filaname_coco_annnotation,filename_flat,list_attributes=None):

    coco = COCO(filaname_coco_annnotation)
    filanames, bboxes, cat_ID,attributes = [],[],[],[]

    for key in coco.imgToAnns.keys():
        annotations = coco.imgToAnns[key]
        image_id = annotations[0]['image_id']
        filename = coco.imgs[image_id]['file_name']

        for annotation in annotations:
            filanames.append(filename)
            cat_ID.append(annotation['category_id'])
            bbox = annotation['bbox']
            #bboxes.append([bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]])
            bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2],bbox[1] + bbox[3]])

            if list_attributes is not None:
                attributes+=[(annotation['attributes'][a]) for a in list_attributes if a in annotation['attributes']]

    bboxes = numpy.array(bboxes)
    df = pd.DataFrame({'filaname':filanames,'left':bboxes[:,0],'top':bboxes[:,1],'right':bboxes[:,2],'bottom':bboxes[:,3],'class_ID':cat_ID})

    if list_attributes is not None:
        attributes = numpy.array(attributes).reshape((df.shape[0],-1))
        df = pd.concat([df,pd.DataFrame(attributes)],axis=1)
        df  = df.rename(columns=dict(zip(df.columns[-len(list_attributes):], list_attributes)))

    df.to_csv(filename_flat,index=False,sep=' ')

    return
# ----------------------------------------------------------------------------------------------------------------------
def extract_supercategories(folder_images,filename_coco_annnotation,folder_out):
    tools_IO.remove_files(folder_out, '*.jpg')
    coco = COCO(filename_coco_annnotation)

    dct_class_names = dict(zip(coco.cats.keys(),[coco.cats[k]['name'] for k in coco.cats.keys()]))
    dct_super_categories = dict(zip(coco.cats.keys(),[coco.cats[k]['supercategory'] for k in coco.cats.keys()]))

    for supercat in set(dct_super_categories.values()):
        if not os.path.exists(folder_out+supercat):
            os.mkdir(folder_out+supercat)
        else:
            tools_IO.remove_files(folder_out+supercat+'/', '*.jpg')


    for key in tqdm(coco.imgToAnns.keys(), total=len(coco.imgToAnns.keys())):
        annotations = coco.imgToAnns[key]
        image_id = annotations[0]['image_id']
        filename = coco.imgs[image_id]['file_name']
        if not os.path.isfile(folder_images + filename):
            continue

        image = cv2.imread(folder_images + filename)

        for annotation in annotations:
            cat_id = annotation['category_id']
            supercat_id = dct_super_categories[cat_id]
            bbox = annotation['bbox']
            #rect = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            image_crop = image[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
            if image_crop.shape[0]>0 and image_crop.shape[1]>0:
                cv2.imwrite(folder_out + supercat_id + '/' + filename, image_crop)

    return
# ----------------------------------------------------------------------------------------------------------------------