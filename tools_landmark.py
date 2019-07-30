import cv2
import numpy
# ---------------------------------------------------------------------------------------------------------------------
from scipy.spatial import Delaunay
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
    r1 = cv2.boundingRect(numpy.float32([t1]))
    r2 = cv2.boundingRect(numpy.float32([t2]))
    r =  cv2.boundingRect(numpy.float32([t]))

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


    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask
# ---------------------------------------------------------------------------------------------------------------------
def get_morph(src_img,target_img,src_points,target_points,del_triangles,alpha=0.5,keep_src_colors=True):

    weighted_pts = []
    for i in range(0, len(src_points)):
        x = (1 - alpha) * src_points[i][0] + alpha * target_points[i][0]
        y = (1 - alpha) * src_points[i][1] + alpha * target_points[i][1]
        weighted_pts.append((x, y))

    img_morph = numpy.zeros(src_img.shape, dtype=src_img.dtype)

    for triangle in del_triangles:
        x, y, z = triangle
        t1 = [src_points[x], src_points[y], src_points[z]]
        t2 = [target_points[x], target_points[y], target_points[z]]
        t = [weighted_pts[x], weighted_pts[y], weighted_pts[z]]
        if keep_src_colors:
            morph_triangle(src_img, target_img, img_morph, t1, t2, t, 0)
        else:
            morph_triangle(src_img, target_img, img_morph, t1, t2, t, alpha)

    return img_morph
# ---------------------------------------------------------------------------------------------------------------------
def transferface_first_to_second_manual(filename_image_first, filename_image_second,file_annotations):


    image1 = cv2.imread(filename_image_first)
    image2 = cv2.imread(filename_image_second)

    delim = ' '
    with open(file_annotations) as f: lines = f.readlines()[1:]
    boxes_xyxy = numpy.array([line.split(delim)[1:5] for line in lines], dtype=numpy.int)
    filenames = numpy.array([line.split(delim)[0] for line in lines])
    class_IDs = numpy.array([line.split(delim)[5] for line in lines], dtype=numpy.int)

    L1_original = boxes_xyxy[filenames==filename_image_first.split('/')[-1]][:,:2]
    L2_original = boxes_xyxy[filenames==filename_image_second.split('/')[-1]][:,:2]
    del_triangles = Delaunay(L1_original).vertices

    landmarks = L1_original

    for t in del_triangles:
        p0 = (landmarks[t[0], 0], landmarks[t[0], 1])
        p1 = (landmarks[t[1], 0], landmarks[t[1], 1])
        p2 = (landmarks[t[2], 0], landmarks[t[2], 1])
        #cv2.line(image1,p0,p1,(0,0,255))
        #cv2.line(image1,p0,p2,(0, 0, 255))
        #cv2.line(image1,p2,p1,(0, 0, 255))

    #for i in range(len(L1_original)):cv2.putText(image1, '{0:d}'.format(i), (L1_original[i,0], L1_original[i,1]),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255))
    #for i in range(len(L2_original)):cv2.putText(image2, '{0:d}'.format(i), (L2_original[i, 0], L2_original[i, 1]),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
    #cv2.imwrite('./images/output/tri1.png',image1)
    #cv2.imwrite('./images/output/tri2.png', image2)

    L1_original = L1_original.astype(numpy.float)
    L2_original = L2_original.astype(numpy.float)


    H = tools_calibrate.get_transform_by_keypoints(L1_original,L2_original)
    aligned1, aligned2= tools_calibrate.get_stitched_images_using_translation(image1, image2, H,keep_shape=True)
    L1_aligned, L2_aligned = tools_calibrate.translate_coordinates(image1, image2, H, L1_original, L2_original)

    #H = tools_calibrate.get_homography_by_keypoints(L1_original,L2_original)
    #aligned1, aligned2= tools_calibrate.get_stitched_images_using_homography(image1, image2, H)
    #L1_aligned, L2_aligned = tools_calibrate.homography_coordinates(image1, image2, H, L1_original, L2_original)

    face = get_morph(aligned1, aligned2, L1_aligned, L2_aligned, del_triangles, alpha=1,keep_src_colors=True)
    result2 = tools_image.blend_multi_band_large_small(aligned2, face, (0, 0, 0))

    cv2.imwrite('./images/output/res.png', result2)

    return result2
# ---------------------------------------------------------------------------------------------------------------------
def transferface_first_to_second(D,filename_image_first, filename_image_second,folder_out=None):

    do_debug = False
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

    result = [('filename', 'x_right', 'y_top', 'x_left', 'y_bottom', 'class_ID', 'confidence')]
    for l1,l2 in zip(L1_original,L2_original):
        result.append([filename_image_first.split('/')[-1],  l1[0],l1[1],l1[0],l1[1],0,1])
        result.append([filename_image_second.split('/')[-1], l2[0],l2[1],l2[0],l2[1],0,1])

    tools_IO.save_mat(result,'./images/markup.txt',delim=' ')


    idx = [1,2,3,4,14,15,16,17,18,19,20,21,22,23,24,25,26,27,8,9,10,32,33,34,35,36,37,38,38,40,41,42,43,44,45,46,47,48]
    idx = numpy.arange(0,68,1).tolist()

    H = tools_calibrate.get_transform_by_keypoints(L1_original[idx],L2_original[idx])
    aligned1, aligned2= tools_calibrate.get_stitched_images_using_translation(image1, image2, H,keep_shape=True)
    L1_aligned, L2_aligned = tools_calibrate.translate_coordinates(image1, image2, H, L1_original, L2_original)

    #H = tools_calibrate.get_homography_by_keypoints(L1_original[idx],L2_original[idx])
    #aligned1, aligned2= tools_calibrate.get_stitched_images_using_homography(image1, image2, H)
    #L1_aligned, L2_aligned = tools_calibrate.homography_coordinates(image1, image2, H, L1_original, L2_original)

    if do_debug and folder_out is not None:
        cv2.imwrite(folder_out+'original1.jpg', image1)
        cv2.imwrite(folder_out+'original2.jpg', image2)
        cv2.imwrite(folder_out+'aligned1.jpg', aligned1)
        cv2.imwrite(folder_out+'aligned2.jpg', aligned2)

    face = get_morph(aligned1, aligned2, L1_aligned, L2_aligned, del_triangles, alpha=1,keep_src_colors=True)
    #face = get_morph(aligned1, aligned2, L1_aligned, L2_aligned, del_triangles, alpha=0,keep_src_colors=True)
    if do_debug and folder_out is not None:
        cv2.imwrite(folder_out + 'face1.jpg', face)


    result2 = tools_image.blend_multi_band_large_small(aligned2, face, (0, 0, 0))

    if do_debug and folder_out is not None:
        cv2.imwrite(folder_out+'result2.jpg', result2)

    return result2
# ---------------------------------------------------------------------------------------------------------------------
def transferface_folder(D,filename_candidate, folder_in,folder_out):

    tools_IO.remove_files(folder_out,create=True)

    local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')

    image1 = cv2.imread(filename_candidate)
    L1_original = D.get_landmarks(image1)
    del_triangles = Delaunay(L1_original).vertices
    idx = [1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 8, 9, 10, 32, 33, 34, 35, 36, 37, 38, 38,40, 41, 42, 43, 44, 45, 46, 47, 48]
    idx = numpy.arange(0, 68, 1).tolist()

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
        cv2.imwrite(folder_out+'result_%03d.jpg'%(weight*100), result)

    cv2.imwrite(folder_out + 'result_%03d.jpg' % (0 * 100), stage1)
    cv2.imwrite(folder_out + 'result_%03d.jpg' % (1 * 100), stage2)

    return
# ---------------------------------------------------------------------------------------------------------------------