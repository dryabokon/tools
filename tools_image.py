import cv2
import numpy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import math
from scipy import ndimage
from skimage.transform import rescale, resize
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
#--------------------------------------------------------------------------------------------------------------------------
def canvas_extrapolate_gray(gray, new_height, new_width):

    newimage = numpy.zeros((new_height,new_width),numpy.uint8)
    shift_x = int((newimage.shape[0] - gray.shape[0]) / 2)
    shift_y = int((newimage.shape[1] - gray.shape[1]) / 2)
    newimage[shift_x:shift_x + gray.shape[0], shift_y:shift_y + gray.shape[1]] = gray[:, :]

    newimage[:shift_x, shift_y:shift_y + gray.shape[1]] = gray[0, :]
    newimage[shift_x + gray.shape[0]:, shift_y:shift_y + gray.shape[1]] = gray[-1, :]

    for row in range(0, newimage.shape[0]):
        newimage[row, :shift_y] = newimage[row, shift_y]
        newimage[row, shift_y + gray.shape[1]:] = newimage[row, shift_y + gray.shape[1] - 1]

    return newimage
# ---------------------------------------------------------------------------------------------------------------------
def smart_resize(img, target_image_height, target_image_width,bg_color=(128, 128, 128)):
    '''resize image with unchanged aspect ratio using padding'''
    from PIL import Image

    pillow_image = Image.fromarray(img)

    original_image_width, original_image_height = pillow_image.size

    scale = min(target_image_width / original_image_width, target_image_height / original_image_height)
    nw = int(original_image_width * scale)
    nh = int(original_image_height * scale)

    pillow_image = pillow_image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (target_image_width, target_image_height), bg_color)
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
def rotate_image(image, angle):
    image_center = tuple(numpy.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
# ---------------------------------------------------------------------------------------------------------------------
def transpone_image(image):
    return numpy.transpose(image,(1,0,2))
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
def rgb2bgr(image):
    return image[:, :, [2, 1, 0]]
    #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# --------------------------------------------------------------------------------------------------------------------------
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
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(numpy.float)
        hsv[:,:,1]*=(1-level)
        hsv = hsv.astype(numpy.uint8)

        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        result = cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_GRAY2BGR)

    return result
#--------------------------------------------------------------------------------------------------------------------------
def hitmap2d_to_jet(hitmap_2d,colormap=None):
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    hitmap_RGB_jet = cv2.applyColorMap(hitmap_2d.astype(numpy.uint8), colormap)
    return hitmap_RGB_jet
# ----------------------------------------------------------------------------------------------------------------------
def hsv2bgr(hsv):
    return cv2.cvtColor(numpy.array([hsv[0], hsv[1], hsv[2]], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)
# ----------------------------------------------------------------------------------------------------------------------
def bgr2hsv(brg):
    return cv2.cvtColor(numpy.array([brg[0], brg[1], brg[2]], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)
# ----------------------------------------------------------------------------------------------------------------------
def gre2jet(rgb):
    return cv2.applyColorMap(numpy.array(rgb, dtype=numpy.uint8).reshape((1, 1, 3)), cv2.COLORMAP_JET).reshape(3)
# ----------------------------------------------------------------------------------------------------------------------
def gre2viridis(rgb):
    colormap = numpy.flip((numpy.array(cm.cmaps_listed['viridis'].colors) * 256).astype(int), axis=1)
    return colormap[int(rgb[0])]
# ----------------------------------------------------------------------------------------------------------------------
def hitmap2d_to_viridis(hitmap_2d):
    colormap = (numpy.array(cm.cmaps_listed['viridis'].colors)*256).astype(int)
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
        M = len(cmap.colors)
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
def blend_multi_band(left, rght, background_color=(255, 255, 255)):

    left_l, left_r, left_t, left_b = get_borders(left, background_color)
    rght_l, rght_r, rght_t, rght_b = get_borders(rght, background_color)
    border = int((left_r+rght_l)/2)

    mask = numpy.zeros(left.shape)
    mask[:, :border] = 1

    leveln = int(numpy.floor(numpy.log2(min(left.shape[0], left.shape[1]))))

    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(numpy.array(left).astype('float'), leveln)
    LPB = LaplacianPyramid(numpy.array(rght).astype('float'), leveln)
    blended = blend_pyramid(LPA, LPB, MP)

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
def blend_multi_band_large_small0(large, small, background_color=(255, 255, 255), adjust_colors='avg', filter_size=50, n_clips=1, do_debug=False):

    mask_original = 1*(small[:, :] == background_color)
    mask = mask_original.copy()
    mask = numpy.array(numpy.min(mask,axis=2),dtype=numpy.float)

    if do_debug: cv2.imwrite('./images/output/mask0.png', 255 * mask)

    if n_clips>0:
        mask = ndimage.uniform_filter(mask, size=(filter_size,filter_size), mode='reflect')
        #mask = cv2.GaussianBlur(mask, (int(filter_size), int(filter_size)), 0)

    if do_debug: cv2.imwrite('./images/output/mask1.png', 255 * mask)

    for c in range(n_clips):
        mask = numpy.clip(2 * mask, 0, 1.0)
        if do_debug == 1: cv2.imwrite('./images/output/mask2.png', 255 * mask)

    if adjust_colors is not None:
        large = large.astype(numpy.float)
        small = small.astype(numpy.float)

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
    mask_bin = numpy.array(numpy.min(mask_bin,axis=2),dtype=numpy.int)

    if do_debug: cv2.imwrite('./images/output/mask0.png', 255 * mask_bin)

    mask = sliding_2d(mask_bin,-filter_size//2,filter_size//2,-filter_size//2,filter_size//2,'avg')
    if do_debug: cv2.imwrite('./images/output/mask1.png', 255 * mask)

    mask = numpy.clip(2 * mask, 0, 1.0)
    if do_debug == 1: cv2.imwrite('./images/output/mask2.png', 255 * mask)

    if adjust_colors:
        large = large.astype(numpy.float)
        small = small.astype(numpy.float)

        cnt_small = sliding_2d(1-mask_bin,-filter_size,filter_size,-filter_size,filter_size,'cnt')
        for c in range(3):

            avg_large = sliding_2d(large[:, :, c],-filter_size,filter_size,-filter_size,filter_size,'avg')

            sum_small = sliding_2d(small[:, :, c],-filter_size,filter_size,-filter_size,filter_size,'cnt')
            avg_small = sum_small/cnt_small
            if do_debug: cv2.imwrite('./images/output/avg_large.png', avg_large)
            if do_debug: cv2.imwrite('./images/output/avg_small.png', avg_small)
            if do_debug: cv2.imwrite('./images/output/cnt_small.png', cnt_small)

            scale = avg_large/avg_small
            scale = numpy.nan_to_num(scale)
            small[:,:,c]=small[:,:,c]*scale

    if do_debug: cv2.imwrite('./images/output/small_corrected.png', small)

    mask = numpy.stack((mask,mask,mask),axis=2)

    result = do_blend(large,small,mask)

    return result
#----------------------------------------------------------------------------------------------------------------------
def do_blend(large,small,mask):

    if len(mask.shape)==2:
        mask = numpy.array([mask,mask,mask]).transpose([1,2,0])

    if mask.max()>1:
        mask = (mask.astype(numpy.float)/255.0)

    background = numpy.multiply(mask, large)
    background = numpy.clip(background, 0, 255)


    foreground = numpy.multiply(1 - mask, small)
    foreground = numpy.clip(foreground, 0, 255)

    result = cv2.add(background, foreground)
    result = numpy.clip(result, 0, 255)

    result = numpy.array(result).astype(numpy.uint8)
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
def convolve_with_mask(image255, mask_pn,min_value=None,max_value=None):

    res = cv2.filter2D(image255.astype(numpy.long), -1, mask_pn)

    if min_value is None or max_value is None:
        #min_value = 255*(mask_pn[mask_pn<=0]).sum()
        #max_value = 255*(mask_pn[mask_pn>0]).sum()
        min_value = -255*mask_pn.shape[0]*mask_pn.shape[1]
        max_value = +255*mask_pn.shape[0]*mask_pn.shape[1]

    res -= min_value
    res = 255*res/(max_value-min_value)

    return res.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
def convolve_with_mask_corners(image255, corners_p, corners_n):



    return
# ----------------------------------------------------------------------------------------------------------------------

def auto_corel(image):
    return
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

    B = numpy.pad(A,((-h_neg,h_pos),(-w_neg,w_pos)),mode)
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

    image_resized = M*resize(image, (dsize[1],dsize[0]),anti_aliasing=True)
    if image_resized.max() <= 1:
        image_resized*= 255

    image_resized = numpy.clip(0,255,image_resized).astype(numpy.uint8)


    return image_resized
# --------------------------------------------------------------------------------------------------------------------
def do_rescale(image,scale,anti_aliasing=True):
    image_rescaled = 255*rescale(image, scale, anti_aliasing=anti_aliasing)
    return image_rescaled.astype(numpy.uint8)
# --------------------------------------------------------------------------------------------------------------------
def put_color_by_mask(image, mask2d, color):

    idx = numpy.where(mask2d > 0)
    for r,c in zip(idx[0], idx[1]):
        image[r,c,[0,1,2]]=image[r,c,[0,1,2]]*(1-mask2d[r,c]/255) +numpy.array(color)[[0,1,2]]*mask2d[r,c]/255

    return image
# --------------------------------------------------------------------------------------------------------------------