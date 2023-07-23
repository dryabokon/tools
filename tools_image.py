import cv2
import numpy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import math
from scipy import ndimage
from skimage.transform import rescale, resize
from PIL import Image as PillowImage
from io import BytesIO
import requests
import base64
#--------------------------------------------------------------------------------------------------------------------------
def numerical_devisor(n):

    candidates = []

    for i in numpy.arange(int(math.sqrt(n))+1,1,-1):
        if n%i==0:
            candidates.append((i,n//i))

    if len(candidates)>0:
        d = [abs(x-y) for x,y in candidates]
        i = int(numpy.argmin(d))
        return candidates[i][0]


    return n
# ---------------------------------------------------------------------------------------------------------------------
def smart_resize(img, target_image_height, target_image_width,bg_color=(128, 128, 128)):
    '''resize image with unchanged aspect ratio using padding'''

    pillow_image = PillowImage.fromarray(img)

    original_image_width, original_image_height = pillow_image.size

    if target_image_width is None:
        target_image_width = int(img.shape[1]*target_image_height/img.shape[0])

    if target_image_height is None:
        target_image_height = int(img.shape[0]*target_image_width/img.shape[1])


    scale = min(target_image_width / original_image_width, target_image_height / original_image_height)
    nw = int(original_image_width * scale)
    nh = int(original_image_height * scale)

    pillow_image = pillow_image.resize((nw, nh), PillowImage.BICUBIC)
    new_image = PillowImage.new('RGB', (target_image_width, target_image_height), bg_color)
    new_image.paste(pillow_image, ((target_image_width - nw) // 2, (target_image_height - nh) // 2))
    return numpy.array(new_image)
# ---------------------------------------------------------------------------------------------------------------------
def center_crop(img, size):
    h, w = img.shape[0],img.shape[1]

    if h>w:
        new_height = int(size*h/w)
        new_width = size
        img = cv2.resize(img,(new_width,new_height))
        return crop_image(img, top=(new_height-new_width)//2, left=0, bottom=new_height-1-(new_height-new_width)//2, right=new_width)
    else:
        new_height = size
        new_width = int(size*w/h)
        img = cv2.resize(img, (new_width, new_height))
        return crop_image(img, top=0, left=(new_width-new_height)//2, bottom=new_height, right=new_width-1-(new_width-new_height)//2)
# ---------------------------------------------------------------------------------------------------------------------
def smart_resize_point(original_x, original_y, original_w, original_h, target_w, target_h):
    scale_hor = target_h / original_h
    scale_ver = target_w / original_w
    if scale_hor<scale_ver:
        scale = scale_hor
        y = int(original_y * scale)
        x = int(original_x * scale)
        x+=int(0.5 * (target_w - original_w * scale))
    else:
        scale = scale_ver
        x = int(original_x * scale)
        y = int(original_y * scale)
        y+=int(0.5 * (target_h - original_h * scale))

    return x,y
# ---------------------------------------------------------------------------------------------------------------------
def canvas_extrapolate(img,new_height,new_width):

    if len(img.shape)==3:
        newimage = numpy.zeros((new_height,new_width,3),numpy.uint8)
    else:
        newimage = numpy.zeros((new_height, new_width), numpy.uint8)

    shift_x = int((newimage.shape[0] - img.shape[0]) / 2)
    shift_y = int((newimage.shape[1] - img.shape[1]) / 2)
    newimage[shift_x:shift_x + img.shape[0], shift_y:shift_y + img.shape[1]] = img[:,:]

    newimage[:shift_x, shift_y:shift_y + img.shape[1]] = img[0, :]
    newimage[shift_x + img.shape[0]:, shift_y:shift_y + img.shape[1]] = img[-1, :]

    for row in range(0, newimage.shape[0]):
        newimage[row, :shift_y] = newimage[row, shift_y]
        newimage[row, shift_y + img.shape[1]:] = newimage[row, shift_y + img.shape[1] - 1]

    return newimage
# ---------------------------------------------------------------------------------------------------------------------
def de_vignette(img):
    newimage = img.copy()
    newimage = newimage.astype(numpy.float32)
    row, cols = img.shape[0],img.shape[1]

    a = cv2.getGaussianKernel(cols, 500)
    b = cv2.getGaussianKernel(row, 500)
    c = b * a.T
    d = c / c.max()

    newimage[:, :, 0] = img[:, :, 0] / d
    newimage[:, :, 1] = img[:, :, 1] / d
    newimage[:, :, 2] = img[:, :, 2] / d

    newimage = numpy.clip(newimage,0,255)
    return newimage.astype(img.dtype)
# ---------------------------------------------------------------------------------------------------------------------
def draw_padding(image,top, left, bottom, right,color):
    result = image.copy()
    result[:top, :] = color
    result[-bottom:, :] = color
    result[:, :left] = color
    result[:, -right:] = color
    return result
# ---------------------------------------------------------------------------------------------------------------------
def fade_header(img,color,top):
    newimage = img.copy()
    for row in range(top):
        alpha = row/top
        for col in range(img.shape[1]):
            newimage[row, col, 0] = newimage[row, col, 0] * alpha + (1 - alpha) * color[0]
            newimage[row, col, 1] = newimage[row, col, 1] * alpha + (1 - alpha) * color[1]
            newimage[row, col, 2] = newimage[row, col, 2] * alpha + (1 - alpha) * color[2]

    return newimage
# ---------------------------------------------------------------------------------------------------------------------
def fade_left_right(img,left,right):
    newimage = img.copy()

    K = numpy.ones((15, 50), numpy.float32)
    K/= K.shape[0]*K.shape[1]
    avg = cv2.filter2D(img, -1, K)

    for col in range(right, img.shape[1]):
        alpha = (img.shape[1]-1-col)/(img.shape[1]-1-right)
        for row in range(img.shape[0]):
            newimage[row, col, 0] = newimage[row, col, 0] * alpha + (1 - alpha) * avg[row,img.shape[1]-1,0]
            newimage[row, col, 1] = newimage[row, col, 1] * alpha + (1 - alpha) * avg[row,img.shape[1]-1,1]
            newimage[row, col, 2] = newimage[row, col, 2] * alpha + (1 - alpha) * avg[row,img.shape[1]-1,2]

    for col in range(0,left):
        alpha = col/left
        for row in range(img.shape[0]):
            newimage[row, col, 0] = newimage[row, col, 0] * alpha + (1 - alpha) * avg[row,0,0]
            newimage[row, col, 1] = newimage[row, col, 1] * alpha + (1 - alpha) * avg[row,0,1]
            newimage[row, col, 2] = newimage[row, col, 2] * alpha + (1 - alpha) * avg[row,0,2]


    return newimage
# ---------------------------------------------------------------------------------------------------------------------
def get_image_affine_rotation_mat(image, angle_deg,reshape=False):
    image_center = tuple(numpy.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    h, w = image.shape[:2]
    if reshape:
        sin = math.sin(math.radians(angle_deg))
        cos = math.cos(math.radians(angle_deg))
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))
        rot_mat[0, 2] += ((b_w / 2) - image_center[0])
        rot_mat[1, 2] += ((b_h / 2) - image_center[1])

    #check
    x1, y1 = numpy.matmul(rot_mat, numpy.array((0, 0, 1)))
    x2, y2 = numpy.matmul(rot_mat, numpy.array((w, 0, 1)))
    x3, y3 = numpy.matmul(rot_mat, numpy.array((0, h, 1)))
    x4, y4 = numpy.matmul(rot_mat, numpy.array((w, h, 1)))

    return rot_mat
# ---------------------------------------------------------------------------------------------------------------------
def rotate_image(image, angle_deg,reshape=False,borderValue=None):
    image_center = tuple(numpy.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    if reshape:
        h, w = image.shape[:2]
        sin = math.sin(math.radians(angle_deg))
        cos = math.cos(math.radians(angle_deg))
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))
        rot_mat[0, 2] += ((b_w / 2) - image_center[0])
        rot_mat[1, 2] += ((b_h / 2) - image_center[1])
        result = cv2.warpAffine(image, rot_mat, (b_w, b_h), flags=cv2.INTER_LINEAR,borderValue=borderValue)
    else:
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=borderValue)
    return result
# ---------------------------------------------------------------------------------------------------------------------
def rotate_point(xy,xy_center,angle_deg,reshape=False):
    rot_mat = cv2.getRotationMatrix2D(xy_center, angle_deg, 1.0)
    h, w = xy_center[0]*2,xy_center[1]*2
    if reshape:
        sin = math.sin(math.radians(angle_deg))
        cos = math.cos(math.radians(angle_deg))
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))
        rot_mat[0, 2] += ((b_w / 2) - xy_center[0])
        rot_mat[1, 2] += ((b_h / 2) - xy_center[1])


    result = numpy.matmul(rot_mat,(xy[0],xy[1],1))
    return result
# ---------------------------------------------------------------------------------------------------------------------
def transpone_image(image):
    return numpy.transpose(image,(1,0,2))
# ---------------------------------------------------------------------------------------------------------------------
def hstack_images(image1, image2,background_color=(32, 32, 32)):
    image2 = do_resize(auto_crop(image2, background_color=background_color), numpy.array((-1, image1.shape[0])))
    return numpy.concatenate([image1, image2], axis=1)
# ---------------------------------------------------------------------------------------------------------------------
def crop_image(img, top, left, bottom, right,extrapolate_border=False):

    if top >=0 and left >= 0 and bottom <= img.shape[0] and right <= img.shape[1]:
        return img[top:bottom, left:right]
    if len(img.shape)==2:
        result = numpy.zeros((bottom-top,right-left),numpy.uint8)
    else:
        result = numpy.zeros((bottom - top, right - left,3),numpy.uint8)

    if left > img.shape[1] or right <= 0 or top > img.shape[0] or bottom <= 0:
        return result

    if top < 0:
        start_row_source = 0
        start_row_target = - top
    else:
        start_row_source = top
        start_row_target = 0

    if bottom > img.shape[0]:
        finish_row_source = img.shape[0]
        finish_row_target = img.shape[0]-top
    else:
        finish_row_source = bottom
        finish_row_target = bottom-top

    if left < 0:
        start_col_source = 0
        start_col_target = - left
    else:
        start_col_source = left
        start_col_target = 0

    if right > img.shape[1]:
        finish_col_source = img.shape[1]
        finish_col_target = img.shape[1]-left
    else:
        finish_col_source = right
        finish_col_target = right-left

    result[start_row_target:finish_row_target,start_col_target:finish_col_target] = img[start_row_source:finish_row_source,start_col_source:finish_col_source]

    if extrapolate_border == True:
        fill_border(result,start_row_target,start_col_target,finish_row_target,finish_col_target)

    return result
# ---------------------------------------------------------------------------------------------------------------------
def get_mask(image_layer,background_color=(0,0,0)):

    mask_layer = numpy.zeros((image_layer.shape[0], image_layer.shape[1]), numpy.uint8)
    mask_layer[numpy.where(image_layer[:, :, 0] == background_color[0])] += 1
    mask_layer[numpy.where(image_layer[:, :, 1] == background_color[1])] += 1
    mask_layer[numpy.where(image_layer[:, :, 2] == background_color[2])] += 1
    mask_layer[mask_layer != 3] = 255
    mask_layer[mask_layer == 3] = 0

    return mask_layer
# ---------------------------------------------------------------------------------------------------------------------
def auto_crop(image,background_color=(0,0,0)):
    mask = get_mask(image,background_color)
    Sx = numpy.sum(mask,axis=0)
    Sy = numpy.sum(mask,axis=1)

    left,right = 0,Sx.shape[0] - 1
    top, bottom = 0, Sy.shape[0] - 1
    while left<Sx.shape[0] and Sx[left]==0:left+=1
    while right>0 and Sx[right]==0:right-=1

    while top<Sy.shape[0] and Sy[top]==0:top+=1
    while bottom>0 and Sy[bottom]==0:bottom-=1

    res = crop_image(image, top, left, bottom, right)
    return res
# ---------------------------------------------------------------------------------------------------------------------
def put_layer_on_image(image_background,image_layer,background_color=(0,0,0)):

    mask_layer = numpy.zeros((image_layer.shape[0],image_layer.shape[1]),numpy.uint8)
    mask_layer[numpy.where(image_layer[:, :, 0] == background_color[0])] += 1
    mask_layer[numpy.where(image_layer[:, :, 1] == background_color[1])] += 1
    mask_layer[numpy.where(image_layer[:, :, 2] == background_color[2])] += 1
    mask_layer[mask_layer !=3 ] = 255
    mask_layer[mask_layer == 3] = 0

    mask_layer_inv = cv2.bitwise_not(mask_layer)

    img1 = cv2.bitwise_and(image_background, image_background, mask=mask_layer_inv).astype(numpy.uint8)
    img2 = cv2.bitwise_and(image_layer     , image_layer     , mask=mask_layer).astype(numpy.uint8)


    im_result = cv2.add(img1, img2).astype(numpy.uint8)

    return im_result
#--------------------------------------------------------------------------------------------------------------------------
def put_image(image_large,image_small,start_row,start_col):
    result = image_large.copy()
    end_row_large = start_row+image_small.shape[0]
    end_row_small= image_small.shape[0]
    delta = image_large.shape[0] - end_row_large
    if delta<0:
        end_row_large+=delta
        end_row_small+=delta

    delta = start_row
    start_row_small = 0
    if delta < 0:
        start_row -= delta
        start_row_small -= delta

    end_col_large = start_col + image_small.shape[1]
    end_col_small = image_small.shape[1]
    delta = image_large.shape[1] - end_col_large
    if delta < 0:
        end_col_large += delta
        end_col_small += delta

    delta = start_col
    start_col_small = 0
    if delta < 0:
        start_col -= delta
        start_col_small -= delta



    if start_col<end_col_large and start_row<end_row_large and end_row_large>=0 and end_col_large>=0:
        result[start_row:end_row_large,start_col:end_col_large]=image_small[start_row_small:end_row_small,start_col_small:end_col_small]
    return result
# ---------------------------------------------------------------------------------------------------------------------
def desaturate_2d(image):
    if len(image.shape)==2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#--------------------------------------------------------------------------------------------------------------------------
def saturate(image):
    if len(image.shape)==2:
        return cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_GRAY2BGR)
    else:
        return image
#--------------------------------------------------------------------------------------------------------------------------
def desaturate(image,level=1.0):

    if level==1:
        gray = cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_BGR2GRAY)
        gray = numpy.expand_dims(gray, 2)
        result = numpy.concatenate([gray, gray, gray], axis=2)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(float)
        hsv[:,:,1]*=(1-level)
        hsv = hsv.astype(numpy.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


    return result
#--------------------------------------------------------------------------------------------------------------------------
def rgb2bgr(image):
    return image[:, :, [2, 1, 0]]#return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# --------------------------------------------------------------------------------------------------------------------------
def hitmap2d_to_jet(hitmap_2d,colormap=None):
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    hitmap_RGB_jet = cv2.applyColorMap(hitmap_2d.astype(numpy.uint8), colormap)
    return hitmap_RGB_jet
# ----------------------------------------------------------------------------------------------------------------------
def hsv2bgr(hsv):
    if hsv.flatten().shape[0] == 3:
        res= cv2.cvtColor(numpy.array([hsv[0], hsv[1], hsv[2]], dtype=numpy.uint8).reshape((1, 1, 3)), cv2.COLOR_HSV2BGR)
    else:
        res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return res
# ----------------------------------------------------------------------------------------------------------------------
def bgr2hsv(brg):
    if brg.flatten().shape[0]==3:
        res = cv2.cvtColor(numpy.array([brg[0], brg[1], brg[2]], dtype=numpy.uint8).reshape((1, 1, 3)), cv2.COLOR_BGR2HSV)
    else:
        res = cv2.cvtColor(brg, cv2.COLOR_BGR2HLS)
    return res
# ----------------------------------------------------------------------------------------------------------------------
def bgr2CMYK(img):
    K = 255.0 - numpy.max(img, axis=2)
    C = (255.0 - img[..., 2] - K) / (255.0 - K + 1e-4)*255
    M = (255.0 - img[..., 1] - K) / (255.0 - K + 1e-4)*255
    Y = (255.0 - img[..., 0] - K) / (255.0 - K + 1e-4)*255
    res = numpy.concatenate([numpy.expand_dims(C,axis=2), numpy.expand_dims(M,axis=2), numpy.expand_dims(Y,axis=2)],axis=2).astype(numpy.uint8)
    return res
# ----------------------------------------------------------------------------------------------------------------------
def bgr2YCR(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return res
# ----------------------------------------------------------------------------------------------------------------------
def gre2jet(rgb):
    return cv2.applyColorMap(numpy.array(rgb, dtype=numpy.uint8).reshape((1, 1, 3)), cv2.COLORMAP_JET).reshape(3)
# ----------------------------------------------------------------------------------------------------------------------
def gre2viridis(rgb):
    colormap = numpy.flip((numpy.array(cm.cmaps_listed['viridis'].colors255) * 256).astype(int), axis=1)
    return colormap[int(rgb[0])]
# ----------------------------------------------------------------------------------------------------------------------
def gre2colormap(gray255,cm_name):
    cmap = plt.get_cmap(cm_name)
    if gray255==0:gray255+=1
    if gray255==255:gray255-=1
    res_color = 255*numpy.array(cmap(gray255))[[2, 1, 0]]
    return res_color
# ----------------------------------------------------------------------------------------------------------------------
def hitmap2d_to_viridis(hitmap_2d):
    colormap = (numpy.array(cm.cmaps_listed['viridis'].colors255) * 256).astype(int)
    colormap = numpy.flip(colormap,axis=1)
    hitmap_RGB = colormap[hitmap_2d[:, :].astype(int)]

    return hitmap_RGB
# ----------------------------------------------------------------------------------------------------------------------
def hitmap2d_to_colormap(hitmap_2d,cmap=plt.cm.Set3,interpolate_colors=False):

    method=2

    if not interpolate_colors:
        colors = 255*numpy.array([cmap(i/256) for i in range(256)])
        colors = colors[:,[2,1,0]]
    else:
        M = len(cmap.colors255)
        colors = []
        N = 256
        for i in range(N):
            i1 = (M - 1) * i // (N - 1)
            alpha = float((M - 1) * i / (N - 1)) - i1
            colors.append(numpy.array(cmap(i1)) * (1 - alpha) + numpy.array(cmap(i1 + 1)) * (alpha))
        colors = 255*numpy.array(colors)[:,[2,1,0]]

    hitmap_RGB= numpy.zeros((hitmap_2d.shape[0], hitmap_2d.shape[1], 3)).astype(numpy.uint8)

    for r in range (0,hitmap_RGB.shape[0]):
        for c in range(0, hitmap_RGB.shape[1]):
            clr = int(hitmap_2d[r, c])
            hitmap_RGB[r,c] = colors[clr]

    return  hitmap_RGB
# ----------------------------------------------------------------------------------------------------------------------

def fill_border(array,row_top,col_left,row_bottom,col_right):
    array[:row_top,:]=array[row_top,:]
    array[row_bottom:, :] = array[row_bottom-1, :]

    array = cv2.transpose(array)
    #array=array.T
    array[:col_left,:]=array[col_left,:]
    array[col_right:, :] = array[col_right - 1, :]
    #array = array.T
    array = cv2.transpose(array)
    return array
# ----------------------------------------------------------------------------------------------------------------------
def shift_image_vert(image,dv):
    res = image.copy()

    dv = -dv

    if dv != 0:
        if (dv > 0):
            res[0:image.shape[0] - dv, :] = image[dv:image.shape[0], :]
            res[image.shape[0] - dv:, :] = image[image.shape[0] - 1, :]
        else:
            res[-dv:image.shape[0], :] = image[0:image.shape[0] + dv, :]
            res[:-dv, :, :] = image[0, :]



    return res
# ----------------------------------------------------------------------------------------------------------------------
def shift_image(image,dv,dh):

    res = image.copy()

    if dv!=0:
        res = shift_image_vert(image, dv)

    res2 = cv2.transpose(image)
    res2 = shift_image_vert(res2, dh)
    res2 = cv2.transpose(res2)

    return res2
# ----------------------------------------------------------------------------------------------------------------------
def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP
# --------------------------------------------------------------------------------------------------------------------------
def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        size = img.shape[1::-1]

        temp_image = cv2.pyrUp(next_img, dstsize=img.shape[1::-1])
        LP.append(img - temp_image)
        img = next_img
    LP.append(img)
    return LP
# --------------------------------------------------------------------------------------------------------------------------
def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended

# --------------------------------------------------------------------------------------------------------------------------
def reconstruct_from_pyramid(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, dstsize=lev_img.shape[1::-1])
        img += lev_img
    return img
    # --------------------------------------------------------------------------------------------------------------------------
def get_borders(image, bg=(255, 255, 255)):

    if (bg == (255, 255, 255)):
        prj = numpy.min(image, axis=0)
    else:
        prj = numpy.max(image, axis=0)

    flg = (prj == bg)[:, 0]

    l = numpy.argmax(flg == False)
    r = numpy.argmin(flg == False)
    return l, r, 0, 0
# --------------------------------------------------------------------------------------------------------------------------
def blend_multi_band(left, rght, background_color=(255, 255, 255),do_debug=False):

    left_l, left_r, left_t, left_b = get_borders(left, background_color)
    rght_l, rght_r, rght_t, rght_b = get_borders(rght, background_color)
    border = int((left_r+rght_l)/2)
    mask = numpy.zeros(left.shape)
    mask[:, :border] = 1

    #leveln = int(numpy.floor(numpy.log2(min(left.shape[0], left.shape[1]))))
    leveln = 5


    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(numpy.array(left).astype('float'), leveln)
    LPB = LaplacianPyramid(numpy.array(rght).astype('float'), leveln)
    blended = blend_pyramid(LPA, LPB, MP)

    if do_debug:
        for i,image in enumerate(LPA):
            cv2.imwrite('./images/output/%02d.jpg'%i,255*image)

    result = reconstruct_from_pyramid(blended)
    result[result > 255] = 255
    result[result < 0] = 0
    return result
#----------------------------------------------------------------------------------------------------------------------
def align_color(small,large,mask):

    small = put_layer_on_image(large,small).astype(numpy.float)
    filter_size = 20#small.shape[0]/5

    B = 128+small[:, :, 0]//2 - large[:, :, 0]//2
    G = 128+small[:, :, 1]//2 - large[:, :, 1]//2
    R = 128+small[:, :, 2]//2 - large[:, :, 2]//2

    Rf = ndimage.uniform_filter(R, size=(filter_size, filter_size), mode='reflect')
    Gf = ndimage.uniform_filter(G, size=(filter_size, filter_size), mode='reflect')
    Bf = ndimage.uniform_filter(B, size=(filter_size, filter_size), mode='reflect')

    res = small.copy()
    res[:, :, 0] -= (Bf-128)*2
    res[:, :, 1] -= (Gf-128)*2
    res[:, :, 2] -= (Rf-128)*2

    res = numpy.clip(res,0,255)
    res[numpy.where(mask != 0)] = 0

    return res
#----------------------------------------------------------------------------------------------------------------------
def blend_multi_band_large_small0(large, small, background_color=(0, 0, 0), adjust_colors='avg', filter_size=50, n_clips=1, do_debug=True):

    mask_original = 1*(small[:, :] == background_color)
    mask = mask_original.copy()
    mask = numpy.array(numpy.min(mask,axis=2),dtype=numpy.float32)

    if do_debug: cv2.imwrite('./images/output/mask0.png', 255 * mask)

    if n_clips>0:
        mask = ndimage.uniform_filter(mask, size=(filter_size,filter_size), mode='nearest')
        #mask = cv2.GaussianBlur(mask, (int(filter_size), int(filter_size)), 0)

    if do_debug: cv2.imwrite('./images/output/mask1.png', 255 * mask)

    for c in range(n_clips):
        mask = numpy.clip(2 * mask, 0, 1.0)
        if do_debug == 1: cv2.imwrite('./images/output/mask2.png', 255 * mask)

    if adjust_colors is not None:
        large = large.astype(numpy.float32)
        small = small.astype(numpy.float32)

        idx = numpy.where(mask[:,:]<0.5)
        if len(idx[0])>0:
            for c in range(3):
                scale = numpy.average(small[:,:,c][idx])/ numpy.average(large[:,:,c][idx])
                if adjust_colors=='avg':
                    scale = numpy.sqrt(scale)
                    large[:,:,c] = large[:,:,c]*scale
                    small[:,:,c]=small[:,:,c]/scale
                if adjust_colors=='large':
                    large[:,:,c] = large[:,:,c]*scale
                if adjust_colors=='small':
                    small[:,:,c]=small[:,:,c]/scale

    if do_debug: cv2.imwrite('./images/output/small_corrected.png', small)

    mask = numpy.stack((mask,mask,mask),axis=2)

    result = do_blend(large,small,mask)

    return result
#----------------------------------------------------------------------------------------------------------------------
def blend_multi_band_large_small(large, small, background_color=(255, 255, 255), adjust_colors=False, filter_size=50,  do_debug=False):

    mask_original = 1*(small[:, :] == background_color)
    mask_bin = mask_original.copy()
    mask_bin = numpy.array(numpy.min(mask_bin,axis=2),dtype=int)

    if do_debug:
        cv2.imwrite('./images/output/mask0.png', 255 * mask_bin)

    mask = sliding_2d(mask_bin,-filter_size//2,filter_size//2,-filter_size//2,filter_size//2,stat='avg',mode='reflect')
    if do_debug:
        cv2.imwrite('./images/output/mask1.png', 255 * mask)

    mask = numpy.clip(2 * mask, 0, 1.0)
    if do_debug:
        cv2.imwrite('./images/output/mask2.png', 255 * mask)

    if adjust_colors is not None:
        large = large.astype(float)
        small = small.astype(float)

        idx = numpy.where(mask[:,:]<0.5)
        if len(idx[0])>0:
            for c in range(3):
                scale = numpy.average(small[:,:,c][idx])/ numpy.average(large[:,:,c][idx])
                if adjust_colors=='avg':
                    scale = numpy.sqrt(scale)
                    large[:,:,c] = large[:,:,c]*scale
                    small[:,:,c]=small[:,:,c]/scale
                if adjust_colors=='large':
                    large[:,:,c] = large[:,:,c]*scale
                if adjust_colors=='small':
                    small[:,:,c]=small[:,:,c]/scale

    if do_debug: cv2.imwrite('./images/output/small_corrected.png', small)

    mask = numpy.stack((mask,mask,mask),axis=2)

    result = do_blend(large,small,mask)

    return result
#----------------------------------------------------------------------------------------------------------------------
def do_blend(large,small,mask):

    if len(mask.shape)==2:
        mask = numpy.array([mask,mask,mask]).transpose([1,2,0])

    if numpy.max(mask)>1:
        mask = (mask.astype(numpy.float)/255.0)

    background = numpy.multiply(mask, large)
    background = numpy.clip(background, 0, 255)


    foreground = numpy.multiply(1 - mask, small)
    foreground = numpy.clip(foreground, 0, 255)

    result = cv2.add(background, foreground)
    result = numpy.clip(result, 0, 255)

    result = result.astype(numpy.uint8)
    return result
#----------------------------------------------------------------------------------------------------------------------
def blend_avg(img1, img2,background_color=(255,255,255),weight=0.5):

    im1 = put_layer_on_image(img1, img2,background_color)
    im2 = put_layer_on_image(img2, img1,background_color)
    res = cv2.add(im1*(1-weight), im2*(weight))

    return res.astype(numpy.uint8)
#----------------------------------------------------------------------------------------------------------------------
def batch_convert(path_in, path_out, format_in='.bmp', format_out='.png'):
    filelist = [f for f in os.listdir(path_in)]
    if path_out[-1]!='/':
        path_out+='/'

    if path_in[-1]!='/':
        path_in+='/'

    for f in filelist:
        if f.endswith(format_in):
            name = f.split('.')[0]
            im = cv2.imread(path_in + f)
            out_name = path_out+name+format_out
            cv2.imwrite(out_name,im)

    return
# ----------------------------------------------------------------------------------------------------------------------
def convolve_with_mask_filter2d(image255, mask_pn, min_value=None, max_value=None):

    res = cv2.filter2D(image255.astype(numpy.int64), -1, mask_pn)

    if min_value is None or max_value is None:
        #min_value = 255*(mask_pn[mask_pn<=0]).sum()
        #max_value = 255*(mask_pn[mask_pn>0]).sum()
        min_value = -255*mask_pn.shape[0]*mask_pn.shape[1]
        max_value = +255*mask_pn.shape[0]*mask_pn.shape[1]

    res -= min_value
    res = 255*res/(max_value-min_value)

    return res.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
def convolve_with_mask_fft(image255, mask_pn,min_value=None,max_value=None):

    KH,KW = mask_pn.shape

    res = numpy.real(numpy.fft.ifft2(numpy.fft.fft2(image255) * numpy.fft.fft2(mask_pn, s=image255.shape)))
    res = res[KW:, KW:]

    if min_value is None or max_value is None:
        min_value = -255*mask_pn.shape[0]*mask_pn.shape[1]
        max_value = +255*mask_pn.shape[0]*mask_pn.shape[1]

    res -= min_value
    res = 255 * res / (max_value - min_value)
    res = canvas_extrapolate(res,image255.shape[0],image255.shape[1])
    res = res.astype(numpy.uint8)

    return res.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
def convolve_with_mask_cupy(image255, mask_pn, min_value=None, max_value=None):
    import cupy

    KH,KW = mask_pn.shape[:2]

    A = cupy.array(image255, dtype=cupy.float32)
    B = cupy.array(mask_pn, dtype=cupy.float32)
    A = cupy.fft.fft2(A)
    B = cupy.fft.fft2(B, s=image255.shape)
    C = (cupy.fft.ifft2(A * B))

    hitmap = cupy.asnumpy(cupy.real(C))
    hitmap = numpy.roll(hitmap, -KW // 2, axis=0)
    hitmap = numpy.roll(hitmap, -KW // 2, axis=1)

    hitmap = 255 * hitmap / (255 * mask_pn.sum())

    return hitmap
# ----------------------------------------------------------------------------------------------------------------------
def convolve_with_mask_cupyx_scipy(image255, mask_pn, min_value=None, max_value=None):
    import cupy, cupyx

    KH, KW = mask_pn.shape[:2]
    A = cupy.array(image255, dtype=cupy.float32)
    B = cupy.array(mask_pn, dtype=cupy.float32)

    A = cupyx.scipy.fftpack.fft2(A)
    B = cupyx.scipy.fftpack.fft2(B, shape=image255.shape)
    C = cupyx.scipy.fftpack.ifft2(cupy.multiply(A,B))



    C = cupy.roll(C, -KW // 2, axis=0)
    C = cupy.roll(C, -KW // 2, axis=1)
    C = 255 * C/ (255 * mask_pn.sum())
    hitmap = cupy.asnumpy(cupy.real(C))
    hitmap = numpy.array(hitmap,dtype=numpy.uint8)


    return hitmap
# ----------------------------------------------------------------------------------------------------------------------
def do_patch(img, patch_size=128, boarder=12, color=True):

    OUT_PATCH_SIZE = patch_size - 2 * boarder
    img_h, img_w = img.shape[:2]

    top, bottom, left, right = boarder, boarder + patch_size, boarder, boarder + patch_size
    ext_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)

    input_patch_im = []

    for y in range(0, img_h, OUT_PATCH_SIZE):
        for x in range(0, img_w, OUT_PATCH_SIZE):
            if not color:
                input_patch_im.append(numpy.expand_dims(numpy.asarray(ext_img[y:y + patch_size, x:x + patch_size], dtype=img.dtype), axis=2))
            else:
                input_patch_im.append((numpy.asarray(ext_img[y:y + patch_size, x:x + patch_size], dtype=img.dtype)))

    return numpy.array(input_patch_im)
# ----------------------------------------------------------------------------------------------------------------------
def do_stitch(img_h, img_w, patches, boarder=12, color=True):
    patch_size = patches.shape[1]

    out_patch_size = patch_size - 2 * boarder
    if color:
        output_img = numpy.zeros((img_h + out_patch_size, img_w + out_patch_size, 3), dtype="uint8")
    else:
        output_img = numpy.zeros((img_h + out_patch_size, img_w + out_patch_size, 1), dtype="uint8")
    idx = 0
    for y in range(0, img_h, out_patch_size):
        for x in range(0, img_w, out_patch_size):
            patch = patches[idx,:,:]
            patch = patch[boarder: boarder + out_patch_size, boarder: boarder + out_patch_size]
            output_img[y: y + out_patch_size, x: x + out_patch_size] = patch
            idx = idx + 1

    return output_img[0:img_h,0:img_w]
# ----------------------------------------------------------------------------------------------------------------------
def sliding_2d(A,h_neg,h_pos,w_neg,w_pos, stat='avg',mode='constant'):

    B = cv2.copyMakeBorder(A, -h_neg, h_pos, -w_neg, w_pos, cv2.BORDER_CONSTANT)
    B = numpy.roll(B, 1, axis=0)
    B = numpy.roll(B, 1, axis=1)

    C1 = numpy.cumsum(B , axis=0)
    C2 = numpy.cumsum(C1, axis=1)

    up = numpy.roll(C2, h_pos, axis=0)
    S1 = numpy.roll(up, w_pos, axis=1)
    S2 = numpy.roll(up, w_neg, axis=1)

    dn = numpy.roll(C2, h_neg, axis=0)
    S3 = numpy.roll(dn, w_pos, axis=1)
    S4 = numpy.roll(dn, w_neg, axis=1)

    if stat=='avg':
        R = (S1-S2-S3+S4)/((w_pos-w_neg)*(h_pos-h_neg))
    else:
        R = (S1 - S2 - S3 + S4)

    R = R[-h_neg:-h_pos, -w_neg:-w_pos]

    return R
# --------------------------------------------------------------------------------------------------------------------
def skew_hor(A,value,do_inverce=False):
    shape = numpy.array(A.shape)

    if do_inverce:
        shape[1]-=numpy.abs(value)
    else:
        shape[1]+=numpy.abs(value)

    B = numpy.full(tuple(shape),128,dtype=numpy.uint8)

    if not do_inverce:
        for r in range(B.shape[0]):
            v = int(value * r / (B.shape[0] - 1))
            if v<0:v+=-value

            if len(shape)==3:B[r,v:v+A.shape[1],[0,1,2]] = A[r,:,[0,1,2]]
            else:            B[r,v:v+A.shape[1]        ] = A[r,:]
    else:
        for r in range(B.shape[0]):
            v = int(value * r / (B.shape[0] - 1))
            if v<0:v+=-value
            if len(shape)==3:B[r,:,[0,1,2]] = A[r,v:v+B.shape[1],[0,1,2]]
            else:B[r,:] = A[r,v:v+B.shape[1]]

    return B
# --------------------------------------------------------------------------------------------------------------------
def do_resize(image, dsize):
    M=1
    if image.max()<=1:M=255

    if dsize[0] is None or dsize[0]==-1:
        dsize[0] = int(dsize[1]*image.shape[1]/image.shape[0])
    if dsize[1] is None or dsize[1]==-1:
        dsize[1] = int(dsize[0]*image.shape[0]/image.shape[1])

    image_resized = M*resize(image, (dsize[1],dsize[0]),anti_aliasing=True)
    if image_resized.max() <= 1:
        image_resized*= 255

    image_resized = numpy.clip(image_resized,0,255).astype(numpy.uint8)

    return image_resized
# --------------------------------------------------------------------------------------------------------------------
def do_rescale(image,scale,anti_aliasing=True,multichannel=False):
    pImage = PillowImage.fromarray(image)
    resized = pImage.resize((int(image.shape[1]*scale),int(image.shape[0]*scale)),resample=PillowImage.BICUBIC)
    result = numpy.array(resized)
    return result
# --------------------------------------------------------------------------------------------------------------------
def put_color_by_mask(image, mask2d, color):


    idx = (mask2d > 0)
    image[~idx] = 0

    return image
# --------------------------------------------------------------------------------------------------------------------
def from_URL(url,img_size=';s=640x480'):
    split = url.split(';s=')

    if len(split)>1:
        split[-1]=img_size
        url = ''.join(split)

    response = requests.get(url)
    pil_image = PillowImage.open(BytesIO(response.content))
    image = numpy.array(pil_image)[:, :, [2, 1, 0]]
    return image
# --------------------------------------------------------------------------------------------------------------------
def encode_base64(image):
    out_img = BytesIO()
    PillowImage.fromarray(image[:, :, [2, 1, 0]]).save(out_img, 'PNG')
    out_img.seek(0)
    encoded_bytes = base64.b64encode(out_img.read())#.decode("ascii")#.replace("\n", "")
    return encoded_bytes
# --------------------------------------------------------------------------------------------------------------------
def decode_base64(encoded_bytes):
    decoded_bytes = base64.b64decode(encoded_bytes)
    pil_image = PillowImage.open(BytesIO(decoded_bytes))
    image = numpy.array(pil_image)[:, :, [2, 1, 0]]
    return image
# --------------------------------------------------------------------------------------------------------------------