#https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
import cv2
import numpy
import tools_draw_numpy
# ---------------------------------------------------------------------------------------------------------------------
from scipy.spatial import Delaunay
import detector_landmarks
import tools_calibrate
import tools_image
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
def apply_affine_transform(src, src_tri, target_tri, size):
    warp_mat = cv2.getAffineTransform(numpy.float32(src_tri), numpy.float32(target_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return dst
# ---------------------------------------------------------------------------------------------------------------------
def morph_triangle(img1, img2, img, t1, t2, t, alpha):


    if t1[0] == t1[1] or t1[2] == t1[1] or t1[0] == t1[2]: return
    if t2[0] == t2[1] or t2[2] == t2[1] or t2[0] == t2[2]: return
    if t[0] == t[1] or t[2] == t[1] or t[0] == t[2]: return

    '''
    flag=0
    for i in [0,1,2]:
        if t1[i][0] < 0 or t1[i][0] >= img1.shape[1]: flag = 1
        if t1[i][1] < 0 or t1[i][1] >= img1.shape[0]: flag = 1
        if t2[i][0] < 0 or t2[i][0] >= img2.shape[1]: flag = 1
        if t2[i][1] < 0 or t2[i][1] >= img2.shape[0]: flag = 1
        if  t[i][0] < 0 or  t[i][0] >= img.shape[1]: flag = 1
        if  t[i][1] < 0 or  t[i][1] >= img.shape[0]: flag = 1
    if flag==1:
        return
    '''

    for i in [0,1,2]:
        t1[i][0] = numpy.clip(t1[i][0], 0, img1.shape[1])
        t1[i][1] = numpy.clip(t1[i][1], 0, img1.shape[0])

        t2[i][0] = numpy.clip(t2[i][0], 0, img2.shape[1])
        t2[i][1] = numpy.clip(t2[i][1], 0, img2.shape[0])

        t[i][0] = numpy.clip(t[i][0], 0, img.shape[1])
        t[i][1] = numpy.clip(t[i][1], 0, img.shape[0])


    val1 = (t1[2][0] - t1[1][0]) * (t1[0][1] - t1[2][1]) - (t1[2][1] - t1[1][1]) * (t1[0][0] - t1[2][0])
    val2 = (t2[2][0] - t2[1][0]) * (t2[0][1] - t2[2][1]) - (t2[2][1] - t2[1][1]) * (t2[0][0] - t2[2][0])

    if val1*val2<0:
        return


    r1 = cv2.boundingRect(numpy.float32([t1]))
    r2 = cv2.boundingRect(numpy.float32([t2]))
    r =  cv2.boundingRect(numpy.float32([t]))

    if r[2]<=1 or r[3]<=1:return

    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = numpy.zeros((r[3], r[2], 3), dtype=numpy.float32)
    cv2.fillConvexPoly(mask, numpy.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)



    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2
    if r[1] + r[3] < img.shape[0] and r[0] + r[2]<img.shape[1]:
        before = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]].copy()
        non_empty_mask_shift = numpy.where(before>0)
        xxx = before * (1 - mask) + img_rect * mask
        xxx[non_empty_mask_shift] = before[non_empty_mask_shift]
        img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = xxx



    return
# ---------------------------------------------------------------------------------------------------------------------
def draw_trianges(image,src_points,del_triangles):
    result = tools_image.desaturate(image, 0.8)
    for triangle in del_triangles:
        t = [src_points[triangle[0]], src_points[triangle[1]], src_points[triangle[2]]]
        t = numpy.array(t,dtype=numpy.int)
        cv2.line(result, (t[0][0], t[0][1]), (t[1][0], t[1][1]), (0, 0, 255))
        cv2.line(result, (t[0][0], t[0][1]), (t[2][0], t[2][1]), (0, 0, 255))
        cv2.line(result, (t[2][0], t[2][1]), (t[1][0], t[1][1]), (0, 0, 255))
    return result
# ---------------------------------------------------------------------------------------------------------------------
def get_morph(src_img,target_img,src_points,target_points,del_triangles,alpha=0.5,keep_src_colors=True):

    debug_mode = 0

    weighted_pts = []
    for i in range(0, len(src_points)):
        x = (1 - alpha) * src_points[i][0] + alpha * target_points[i][0]
        y = (1 - alpha) * src_points[i][1] + alpha * target_points[i][1]
        weighted_pts.append([x, y])

    img_morph = numpy.full(src_img.shape,0, dtype=src_img.dtype)

    for i,triangle in enumerate(del_triangles):

        x, y, z = triangle
        t1 = [[src_points[x][0],src_points[x][1]], [src_points[y][0],src_points[y][1]], [src_points[z][0],src_points[z][1]]]
        t2 = [[target_points[x][0], target_points[x][1]], [target_points[y][0], target_points[y][1]],[target_points[z][0], target_points[z][1]]]
        t = [weighted_pts[x], weighted_pts[y], weighted_pts[z]]

        if keep_src_colors:
            morph_triangle(src_img, target_img, img_morph, t1, t2, t, 0)
        else:
            morph_triangle(src_img, target_img, img_morph, t1, t2, t, alpha)
        if debug_mode == 1:
            cv2.imwrite('./images/output/img_morph_%03d.png'%i, img_morph)

    if debug_mode==1:
        src_img_debug    = draw_trianges(src_img   ,src_points,del_triangles)
        target_img_debug = draw_trianges(target_img,target_points, del_triangles)

        #cv2.imwrite('./images/output/src_img_debug.png',src_img_debug)
        #cv2.imwrite('./images/output/target_img_debug.png', target_img_debug)

    return img_morph
# ---------------------------------------------------------------------------------------------------------------------
def do_transfer(image1, image2, L1_original, L2_original, del_triangles):
    H = tools_calibrate.get_transform_by_keypoints(L1_original, L2_original)
    if H is None:
        return image2
    aligned1, aligned2 = tools_calibrate.get_stitched_images_using_translation(image1, image2, H, keep_shape=True)
    L1_aligned, L2_aligned = tools_calibrate.translate_coordinates(image1, image2, H, L1_original, L2_original)

    # H = tools_calibrate.get_homography_by_keypoints(L1_original,L2_original)
    # aligned1, aligned2= tools_calibrate.get_stitched_images_using_homography(image1, image2, H)
    # L1_aligned, L2_aligned = tools_calibrate.homography_coordinates(image1, image2, H, L1_original, L2_original)

    face = get_morph(aligned1, aligned2, L1_aligned, L2_aligned, del_triangles, alpha=1, keep_src_colors=True)
    result2 = tools_image.blend_multi_band_large_small(aligned2, face, (0, 0, 0))

    return result2
# ---------------------------------------------------------------------------------------------------------------------
def do_reenackement(image1, image2, L1_original, L2_original, idx_mouth,del_triangles_all,del_triangles_mouth):

    idx_head = numpy.arange(0, 27, 1).tolist()

    H = tools_calibrate.get_transform_by_keypoints(L2_original[idx_head], L1_original[idx_head])
    if H is None:
        return image1
    aligned2, aligned1 = tools_calibrate.get_stitched_images_using_translation(image2, image1, H, keep_shape=True)
    L2_aligned, L1_aligned = tools_calibrate.translate_coordinates(image2, image1, H, L2_original, L1_original)

    face12 = get_morph(aligned1, aligned1, L1_aligned, L2_aligned, del_triangles_all, alpha=1)
    face21 = get_morph(face12, face12, L2_aligned[idx_mouth], L1_aligned[idx_mouth], del_triangles_mouth, alpha=1)

    filter_size = int(face12.shape[0] * 0.07)
    result2 = tools_image.blend_multi_band_large_small(aligned1, face21, (0, 0, 0), filter_size=filter_size)

    return result2
# ---------------------------------------------------------------------------------------------------------------------
def transferface_first_to_second_manual(filename_image_first, filename_image_second,file_annotations):

    image1 = cv2.imread(filename_image_first)
    image2 = cv2.imread(filename_image_second)

    delim = ' '
    with open(file_annotations) as f: lines = f.readlines()[1:]
    boxes_xyxy = numpy.array([line.split(delim)[1:5] for line in lines], dtype=numpy.int)
    filenames = numpy.array([line.split(delim)[0] for line in lines])

    L1_original = boxes_xyxy[filenames==filename_image_first.split('/')[-1]][:,:2]
    L2_original = boxes_xyxy[filenames==filename_image_second.split('/')[-1]][:,:2]
    L1_original = L1_original.astype(numpy.float)
    L2_original = L2_original.astype(numpy.float)

    del_triangles = Delaunay(L1_original).vertices

    result2 = do_transfer(image1, image2, L1_original, L2_original, del_triangles)

    return result2
# ---------------------------------------------------------------------------------------------------------------------
def transferface_first_to_second(D,filename_image_first, filename_image_second,folder_out=None):

    do_debug = True
    swap = False

    if do_debug and folder_out is not None:
        tools_IO.remove_files(folder_out, create=True)

    image1 = cv2.imread(filename_image_first)
    image2 = cv2.imread(filename_image_second)
    if swap:
        image1,image2 = image2,image1

    L1_original = D.get_landmarks(image1)
    L2_original = D.get_landmarks(image2)

    del_triangles = Delaunay(L1_original).vertices
    idx = numpy.arange(0,len(L1_original),1).tolist()

    H = tools_calibrate.get_transform_by_keypoints(L1_original[idx],L2_original[idx])
    aligned1, aligned2= tools_calibrate.get_stitched_images_using_translation(image1, image2, H,keep_shape=True)
    L1_aligned, L2_aligned = tools_calibrate.translate_coordinates(image1, image2, H, L1_original, L2_original)


    if do_debug and folder_out is not None:

        cv2.imwrite(folder_out+'s01-original1.jpg', image1)
        cv2.imwrite(folder_out+'s05-original2.jpg', image2)
        #for lmark in L1_aligned:aligned1 = tools_draw_numpy.draw_circle(aligned1, lmark[1], lmark[0], 3, [0,0,255])
        #for lmark in L2_aligned:aligned2 = tools_draw_numpy.draw_circle(aligned2, lmark[1], lmark[0], 3, [0, 0, 255])
        cv2.imwrite(folder_out+'s02-aligned1.jpg', aligned1)
        cv2.imwrite(folder_out+'s02-aligned2.jpg', aligned2)

    face = get_morph(aligned1, aligned2, L1_aligned, L2_aligned, del_triangles, alpha=1,keep_src_colors=True)
    if do_debug and folder_out is not None:
        for alpha in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
            #aligned1 = D.draw_landmarks_v2(aligned1, L1_aligned, del_triangles)
            #aligned2 = D.draw_landmarks_v2(aligned2, L2_aligned, del_triangles)
            temp_face = get_morph(aligned1, aligned2, L1_aligned, L2_aligned, del_triangles, alpha=alpha, keep_src_colors=True)
            cv2.imwrite(folder_out + 's03-temp_face_%02d.jpg'%int(alpha*10), temp_face)

    filter_size = int(face.shape[0]*0.07)
    result2 = tools_image.blend_multi_band_large_small(aligned2, face, (0, 0, 0),filter_size=filter_size)

    if do_debug and folder_out is not None:
        cv2.imwrite(folder_out+'s04-result2.jpg', result2)

    return result2
# ---------------------------------------------------------------------------------------------------------------------
def transfer_emo(D,filename_image_src1, filename_image_src2,folder_out=None):

    do_debug = True
    swap = True
    if do_debug and folder_out is not None:
        tools_IO.remove_files(folder_out, create=True)

    image1 = cv2.imread(filename_image_src1)
    image2 = cv2.imread(filename_image_src2)

    if swap:
        image1,image2 = image2,image1

    L1_original = D.get_landmarks(image1)
    L2_original = D.get_landmarks(image2)

    H = tools_calibrate.get_transform_by_keypoints(L2_original[D.idx_head], L1_original[D.idx_head])
    aligned2, aligned1 = tools_calibrate.get_stitched_images_using_translation(image2, image1, H, keep_shape=True)
    L2_aligned, L1_aligned = tools_calibrate.translate_coordinates(image2, image1, H, L2_original, L1_original)

    if do_debug:
        for lmark in L1_aligned :aligned1 = tools_draw_numpy.draw_circle(aligned1, lmark[1], lmark[0], 2, [0, 0, 255])
        for lmark in L2_aligned: aligned2 = tools_draw_numpy.draw_circle(aligned2, lmark[1], lmark[0], 2, [255, 0, 0])

    del_triangles = Delaunay(L1_aligned).vertices
    face12 = get_morph(aligned1,aligned1,L1_aligned,L2_aligned,del_triangles,alpha=1)

    idx = D.idx_head + D.idx_nose + D.idx_eyes
    del_triangles = Delaunay(L2_aligned[idx]).vertices
    face21 = get_morph(face12, face12, L2_aligned[idx], L1_aligned[idx], del_triangles, alpha=1)

    filter_size = int(face12.shape[0] * 0.07)
    result2 = tools_image.blend_multi_band_large_small(aligned1, face21, (0, 0, 0), filter_size=filter_size)

    if do_debug:
        cv2.imwrite(folder_out + 'aligned1.jpg', aligned1)
        cv2.imwrite(folder_out + 'aligned2.jpg', aligned2)
        cv2.imwrite(folder_out + 'face12.jpg', face12)
        cv2.imwrite(folder_out + 'face21.jpg', face21)
        cv2.imwrite(folder_out + 'result2.jpg', result2)

    return result2
# ---------------------------------------------------------------------------------------------------------------------
def transferface_folder(D,filename_candidate, folder_in,folder_out):

    tools_IO.remove_files(folder_out,create=True)

    local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')

    image1 = cv2.imread(filename_candidate)
    L1_original = D.get_landmarks(image1)
    del_triangles = Delaunay(L1_original).vertices
    #idx = [1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 8, 9, 10, 32, 33, 34, 35, 36, 37, 38, 38,40, 41, 42, 43, 44, 45, 46, 47, 48]
    #idx = numpy.arange(0, 68, 1).tolist()
    idx = numpy.arange(0, 48, 1).tolist()

    for local_filename in local_filenames:

        image2 = cv2.imread(folder_in+local_filename)
        L2_original = D.get_landmarks(image2)

        H = tools_calibrate.get_transform_by_keypoints(L1_original[idx],L2_original[idx])
        aligned1, aligned2= tools_calibrate.get_stitched_images_using_translation(image1, image2, H,keep_shape=True)
        L1_aligned, L2_aligned = tools_calibrate.translate_coordinates(image1, image2, H, L1_original, L2_original)

        face = get_morph(aligned1, aligned2, L1_aligned, L2_aligned, del_triangles, alpha=1,keep_src_colors=True)


        result = tools_image.blend_multi_band_large_small(aligned2, face, (0, 0, 0))
        cv2.imwrite(folder_out + local_filename, result)
        print(local_filename)

    return
# ---------------------------------------------------------------------------------------------------------------------
def morph_first_to_second(D,filename_image_first, filename_image_second,folder_out,weight_array):

    stage1 = cv2.imread(filename_image_second)
    stage2 = transferface_first_to_second(D,filename_image_first, filename_image_second)

    for weight in weight_array:
        result = cv2.add(stage1*(1-weight), stage2*(weight))
        cv2.imwrite(folder_out+'result_%03d.jpg'%(weight*100), result)

    cv2.imwrite(folder_out + 'result_%03d.jpg' % (0 * 100), stage1)
    cv2.imwrite(folder_out + 'result_%03d.jpg' % (1 * 100), stage2)

    return
# ---------------------------------------------------------------------------------------------------------------------
def morph_first_to_second_manual(D,filename_image_first, filename_image_second,file_annotations,folder_out,weight_array):

    stage1 = cv2.imread(filename_image_second)
    stage2 = transferface_first_to_second_manual(filename_image_first, filename_image_second,file_annotations)

    for weight in weight_array:
        result = cv2.add(stage1*(1-weight), stage2*(weight))
        cv2.imwrite(folder_out + 'r_%03d.jpg'%(weight*100), result)
        cv2.imwrite(folder_out + 'r_%03d.jpg'%(100 + (100-weight * 100)), result)

    cv2.imwrite(folder_out + 'r_%03d.jpg' % (0 * 100), stage1)
    cv2.imwrite(folder_out + 'r_%03d.jpg' % (1 * 100), stage2)

    return
# ---------------------------------------------------------------------------------------------------------------------