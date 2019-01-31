#----------------------------------------------------------------------------------------------------------------------
import time
import os
import numpy
import math
import cv2
import matplotlib.pyplot as plt
from scipy.misc import toimage
#----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_draw_numpy
import tools_calibrate
import tools_alg_match
import tools_image
import tools_alg_grid_templates
#----------------------------------------------------------------------------------------------------------------------
def blend_by_coord(image,coord1,coord2,window_size,fill_declines=False, background_color=(128, 128, 128)):

    reproject1, reproject2,m1,m2 = tools_alg_match.reproject_matches(image, image, coord1, coord2, fill_declines=fill_declines, window_size=window_size,background_color=background_color)
    base1 = image.copy()
    mask1 = tools_image.get_mask(reproject1,background_color)
    base1[mask1==0]=background_color
    result = tools_image.blend_avg(reproject1, base1, background_color=background_color)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def estimate_pattern_size(coord1, coord2):

    d = coord1[:, 0] - coord2[:, 0]
    hist, bin_edges = numpy.histogram(d)
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.show()

    i=0

    return
# ----------------------------------------------------------------------------------------------------------------------
def generate_input(image_base,coordinates,image_small,border_color=(255,0,0)):

    image_out = image_base.copy()
    dx = image_small.shape[0]
    dy = image_small.shape[1]

    for i in range(0, coordinates.shape[0]):
        x = coordinates[i][0]
        y = coordinates[i][1]

        dst_left   = x - int(dx/2)
        dst_right  = x - int(dx / 2) + dx
        dst_top    = y - int(dy/2)
        dst_bottom = y - int(dy / 2)+ dy
        x_left =0
        x_right = image_small.shape[0]
        x_top=0
        x_bottom = image_small.shape[1]

        if dst_left<0:
            x_left = -dst_left
            dst_left=0

        if dst_right>=image_out.shape[0]:
            x_right -= dst_right-(image_out.shape[0])
            dst_right=image_out.shape[0]

        if dst_top<0:
            x_top = -dst_top
            dst_top=0

        if dst_bottom>=image_out.shape[1]:
            x_bottom -= dst_bottom-(image_out.shape[1])
            dst_bottom=image_out.shape[1]

        image_out[dst_left:dst_right, dst_top:dst_bottom, :] = image_small[x_left:x_right, x_top:x_bottom,:]
        #image_out[dst_left          , dst_top:dst_bottom, :] = numpy.array(border_color)
        #image_out[dst_right-1       , dst_top:dst_bottom, :] = numpy.array(border_color)
        #image_out[dst_left:dst_right, dst_top           , :] = numpy.array(border_color)
        #image_out[dst_left:dst_right, dst_bottom-1      , :] = numpy.array(border_color)


    return image_out
# ----------------------------------------------------------------------------------------------------------------------



def analyze_matches(folder_in,filename_in,folder_out,disp_v1, disp_v2, disp_h1, disp_h2,window_size=5,step=10):

    #if not os.path.exists(folder_out):
    #    os.makedirs(folder_out)
    #else:
    #    tools_IO.remove_files(folder_out)


    image1 = cv2.imread(folder_in+filename_in)
    pattern_width = 200

    print('Calculating matches ...')
    start_time = time.time()
    coord1, coord2, quality = tools_alg_match.get_best_matches(image1,image1,disp_v1, disp_v2, disp_h1, disp_h2, window_size,step)
    print('%s sec\n\n' % (time.time() - start_time))

    quality_image1 = tools_alg_match.interpolate_image_by_matches(image1.shape[0],image1.shape[1],coord2,quality)
    cv2.imwrite(folder_out + '%d-Q1.png' % window_size, quality_image1)
    quality_image2 = tools_alg_match.interpolate_image_by_matches(image1.shape[0],image1.shape[1],coord1,quality)
    cv2.imwrite(folder_out + '%d-Q2.png' % window_size, quality_image2)

    idx = numpy.where(quality>=240)
    coord1,coord2  = coord1[idx], coord2[idx]

    idx = numpy.random.choice(coord1.shape[0], int(coord1.shape[0] * 0.3))
    gray1_rgb = tools_image.desaturate(image1)
    gray2_rgb = tools_image.desaturate(image1)
    gray1_rgb, gray2_rgb = tools_alg_match.visualize_matches_coord(coord1[idx, :],coord2[idx, :],gray1_rgb,gray2_rgb)
    cv2.imwrite(folder_out + '%d-points1.png' % window_size, gray1_rgb)
    cv2.imwrite(folder_out + '%d-points2.png' % window_size, gray2_rgb)

    blend1 = blend_by_coord(image1, coord1, coord2, window_size, fill_declines=False, background_color=(128, 128, 128))
    blend2 = blend_by_coord(image1, coord2, coord1, window_size, fill_declines=False, background_color=(128, 128, 128))
    cv2.imwrite(folder_out + '%d-blend1.png' % window_size, blend1)
    cv2.imwrite(folder_out + '%d-blend2.png' % window_size, blend2)

    return
# ----------------------------------------------------------------------------------------------------------------------
def filter_best_matches(coord1, coord2, quality,th=50):
    idx = numpy.where(quality >= numpy.percentile(quality, th))
    return coord1[idx], coord2[idx], quality[idx]

# ----------------------------------------------------------------------------------------------------------------------
def filter_best_matches_v2(coord1, coord2, quality,th=60):
    R = 1 + numpy.max(coord1[:, 0])
    dR = int(R * 0.05)
    th = numpy.full(R, 0)

    for r in range(0, R):
        set1 = set(numpy.where((coord1[:, 0] > r - dR))[0])
        set2 = set(numpy.where((coord1[:, 0] < r + dR))[0])
        idx = list(set1.intersection(set2))
        if len(idx) > 0:
            th[r] = numpy.percentile(quality[idx], 80)

    idx = []
    for i in range(0, coord1.shape[0]):
        if quality[i] > th[coord1[i, 0]]:
            idx.append(i)

    return coord1[idx], coord2[idx]

# ----------------------------------------------------------------------------------------------------------------------
def build_hitmap(folder_in, filename_in_field,filename_in_pattern, folder_out):
    image_field = cv2.imread(folder_in + filename_in_field)
    img_pattern = cv2.imread(folder_in + filename_in_pattern)

    hitmap_2d = tools_alg_match.calc_hit_field(image_field, img_pattern)
    cv2.imwrite(folder_out + 'hitmap_advanced.png', hitmap_2d)
    cv2.imwrite(folder_out + 'hitmap_advanced_jet.png', tools_image.hitmap2d_to_jet(hitmap_2d))
    cv2.imwrite(folder_out + 'hitmap_advanced_viridis.png', tools_image.hitmap2d_to_viridis(hitmap_2d))

    return
# ----------------------------------------------------------------------------------------------------------------------
def get_self_hits(image, max_period, pattern_height, pattern_width,window_size=25,step=10,folder_debug=None):

    tol = 20

    mask1 = numpy.zeros((max_period, image.shape[0], image.shape[1]))
    mask2 = numpy.zeros((max_period, image.shape[0], image.shape[1]))
    acc_color = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.float64)

    for period in range (0,max_period):
        print(period)
        disp_v1, disp_v2, disp_h1, disp_h2 = -tol, +tol, (pattern_width) * (1+period) - tol, (pattern_width) * (1+period) + tol

        coord1, coord2, quality = tools_alg_match.get_best_matches(image,image,disp_v1, disp_v2, disp_h1, disp_h2, window_size,step)
        coord1, coord2, quality = filter_best_matches(coord1, coord2, quality,80)
        reproject1, reproject2, mask1[period], mask2[period] = tools_alg_match.reproject_matches(image, image, coord1,coord2, window_size,False, (128, 128, 128))
        acc_color[:, :] += reproject1[:, :].astype(numpy.float32) * mask1[period].reshape((mask1.shape[1],mask1.shape[2],1))
        acc_color[:, :] += reproject2[:, :].astype(numpy.float32) * mask2[period].reshape((mask1.shape[1],mask1.shape[2],1))

        if folder_debug is not None:
            cv2.imwrite(folder_debug + '%d-r1_%02d.png' % (window_size, 1+period), reproject1)
            cv2.imwrite(folder_debug + '%d-r2_%02d.png' % (window_size, 1+period), reproject2)

    if folder_debug is not None:
        sum = numpy.sum(mask1 + mask2, axis=0)
        sum += 1
        acc_color += image
        idx = numpy.where(sum > 0)
        acc_color[idx[0], idx[1], 0] /= sum[idx]
        acc_color[idx[0], idx[1], 1] /= sum[idx]
        acc_color[idx[0], idx[1], 2] /= sum[idx]
        idx = numpy.where(sum == 0)
        acc_color[idx[0], idx[1], :] = (128, 128, 128)
        cv2.imwrite(folder_debug + '%d-reproject.png' % (window_size), acc_color)


    hitmap = numpy.average(mask1 + mask2, axis=0)
    hitmap = hitmap * 255 / numpy.max(hitmap)

    return hitmap, pattern_height, pattern_width
# ----------------------------------------------------------------------------------------------------------------------
def get_best_pattern(images, weights, folder_debug=None):

    one = numpy.full((images.shape[1], images.shape[2]), 1, dtype=numpy.uint8)
    Q = numpy.zeros((images.shape[0], images.shape[0]))
    for i in range(0,images.shape[0]-1):

        acc_color = numpy.zeros((images.shape[1], images.shape[2], 3), dtype=numpy.float64)
        sum = numpy.zeros((acc_color.shape[0], acc_color.shape[1]), dtype=numpy.uint8)

        for j in range(i+1, images.shape[0]):
            #im1,im2,q = tools_calibrate.align_two_images_translation(images[j],images[i],detector='SIFT',matchtype='xxx')
            im1,im2,q = tools_calibrate.align_two_images_translation(images[j],images[i],detector='SIFT',matchtype='xxx',borderMode=cv2.BORDER_CONSTANT, background_color=(0, 255, 255))
            Q[i,j] = q
            Q[j,i] = q
            if q > 100:
                acc_color+=im1
                sum+=one
                if folder_debug is not None:
                    cv2.imwrite(folder_debug + 'A_%02d_%02d.png' % (i,j), im1)


        acc_color[:, :, 0] /= sum[:, :]
        acc_color[:, :, 1] /= sum[:, :]
        acc_color[:, :, 2] /= sum[:, :]
        if folder_debug is not None:
            q = int(numpy.average(Q[i,:], axis=0))
            cv2.imwrite(folder_debug + 'acc_color_%03d_%03d.png' % (q,i), acc_color)


    best = numpy.argmax(numpy.average(Q, axis=0))

    acc_color = numpy.zeros((images.shape[1], images.shape[2], 3), dtype=numpy.float64)
    sum       = numpy.zeros((acc_color.shape[0],acc_color.shape[1]), dtype=numpy.uint8)


    for j in range(0, images.shape[0]):
        im1, im2, q = tools_calibrate.align_two_images_translation(images[j], images[best], detector='SIFT', matchtype='xxx',borderMode=cv2.BORDER_CONSTANT, background_color=(0, 255, 255))
        if q>100:
            s = one.copy()
            s[numpy.all(im1 == (0, 255, 255), axis=-1)] = 0
            a = im1.copy()
            a[numpy.all(im1 == (0, 255, 255),axis=-1)]=0

            acc_color+=a
            sum+=s

            if folder_debug is not None:
                cv2.imwrite(folder_debug + 'A_%02d.png' % (j), (a))

    acc_color[:, :, 0] /= sum[:,:]
    acc_color[:, :, 1] /= sum[:, :]
    acc_color[:, :, 2] /= sum[:, :]

    if folder_debug is not None:
        cv2.imwrite(folder_debug + 'acc_color.png', acc_color)


    return acc_color.astype(numpy.uint8)
# ---------------------------------------------------------------------------------------------------------------------
def E2E_detect_patterns(folder_in, filename_in_field,folder_out,pattern_height, pattern_width,max_period, window_size=25,step=10):

    print(folder_in)

    image = cv2.imread(folder_in + filename_in_field)
    image_gray = tools_image.desaturate(image)

    peak_factor = 0.25
    cut_factor = 0.95

    #pairs hitmap
    if os.path.isfile(folder_out+'hitmap.png'):
        hitmap = cv2.imread(folder_out+'hitmap.png',0)
        tools_IO.remove_files(folder_out)
        cv2.imwrite(folder_out + '%d-hitmap_00.png' % (window_size), tools_image.hitmap2d_to_jet(hitmap))
        cv2.imwrite(folder_out + 'hitmap.png', hitmap)
    else:
        tools_IO.remove_files(folder_out)
        hitmap,pattern_height, pattern_width  = get_self_hits(image, max_period,pattern_height, pattern_width,window_size,step,folder_debug=folder_out)
        cv2.imwrite(folder_out + '%d-hitmap_00.png' % (window_size), tools_image.hitmap2d_to_jet(hitmap))
        cv2.imwrite(folder_out + 'hitmap.png' , hitmap)

    #candidates to patterns
    coord = tools_alg_grid_templates.find_local_peaks(hitmap, int(pattern_height*peak_factor),int(pattern_width*peak_factor))
    for i in range(0, coord.shape[0]):
        image_gray = tools_draw_numpy.draw_circle(image_gray, coord[i, 0], coord[i, 1], 5, (0, 64, 255))
    cv2.imwrite(folder_out + '%d-hits_00.png' % window_size, image_gray)
    patterns_candidates = tools_alg_grid_templates.coordinates_to_images(image, coord, cut_factor * pattern_height,cut_factor * pattern_width)
    hitmap_candidates = tools_alg_grid_templates.coordinates_to_images(hitmap, coord, cut_factor * pattern_height,cut_factor * pattern_width)

    #best pattern
    pattern = get_best_pattern(patterns_candidates,hitmap_candidates)
    cv2.imwrite(folder_out + '%d-pattern_00.png' % (window_size), pattern)


    #template match with pattern
    hitmap = tools_alg_match.calc_hit_field_basic(image, pattern,rotation_tol = 10, rotation_step=0.5)
    cv2.imwrite(folder_out + '%d-hitmap_01_jet.png' % (window_size),tools_image.hitmap2d_to_jet(hitmap) )
    cv2.imwrite(folder_out + '%d-hitmap_01_vrd.png' % (window_size), tools_image.hitmap2d_to_viridis(hitmap))

    #visualize hits
    min_value = 200
    coord, image_miss = tools_alg_grid_templates.find_local_peaks_greedy(hitmap, int(1.0*pattern_height), int(1.0*pattern_width), min_value)
    image_gray = tools_image.desaturate(image)
    for i in range(0, coord.shape[0]):
        value  = int(85*(hitmap[coord[i, 0], coord[i, 1]] - min_value)/(255-min_value))
        image_gray = tools_draw_numpy.draw_circle(image_gray, coord[i, 0], coord[i, 1], 9, tools_image.hsv2bgr((value,255,255)))

    image_gray_and_miss = tools_image.put_layer_on_image(image_gray,image_miss,background_color=(0,0,0))
    image_gray_and_miss = tools_image.blend_avg(image_gray, image_gray_and_miss, weight=0.5)
    cv2.imwrite(folder_out + '%d-hits_01.png' % window_size, image_gray_and_miss)



    #reproject pattern
    image_gen = generate_input(image,coord,pattern)
    cv2.imwrite(folder_out + '%d-gen.png' % window_size, image_gen)


    return
# ----------------------------------------------------------------------------------------------------------------------