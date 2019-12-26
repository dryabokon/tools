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
import tools_GL
import tools_filter
# ---------------------------------------------------------------------------------------------------------------------
def do_transfer(R_c,R_a,image_clbrt, image_actor, L_clbrt, L_actor, del_triangles):

    H = tools_calibrate.get_transform_by_keypoints(L_clbrt, L_actor)
    if H is None:
        return image_actor

    L1_aligned, L2_aligned = tools_calibrate.translate_coordinates(image_clbrt, image_actor, H, L_clbrt, L_actor)
    face = R_c.morph_mesh(image_actor.shape[0], image_actor.shape[1], L2_aligned, L_clbrt, del_triangles)

    L2_aligned_mouth = L2_aligned[numpy.arange(48, 61, 1).tolist()]
    del_mouth = Delaunay(L2_aligned_mouth).vertices
    temp_mouth = R_a.morph_mesh(image_actor.shape[0], image_actor.shape[1], L2_aligned_mouth, L2_aligned_mouth,del_mouth)

    filter_size = int(face.shape[0] * 0.07)
    result2 = tools_image.blend_multi_band_large_small(image_actor, face, (0, 0, 0), filter_size=filter_size)
    result2 = tools_image.blend_multi_band_large_small(result2, temp_mouth, (0, 0, 0), do_color_balance=False,filter_size=filter_size // 2)

    return result2
# ---------------------------------------------------------------------------------------------------------------------
def transferface_first_to_second(D,filename_image_clbrt, filename_image_actor,folder_out=None):

    do_debug = True
    swap = False


    if do_debug and folder_out is not None:
        tools_IO.remove_files(folder_out, create=True)

    image_clbrt = cv2.imread(filename_image_clbrt)
    image_actor = cv2.imread(filename_image_actor)
    if swap:
        image_clbrt,image_actor = image_actor,image_clbrt

    R_c = tools_GL.render_GL(image_clbrt)
    R_a = tools_GL.render_GL(image_actor)

    L_clbrt = D.get_landmarks_augm(image_clbrt)
    L_actor = D.get_landmarks_augm(image_actor)
    del_triangles = Delaunay(L_actor).vertices

    H = tools_calibrate.get_transform_by_keypoints(L_clbrt,L_actor)
    L1_aligned, L2_aligned = tools_calibrate.translate_coordinates(image_clbrt, image_actor, H, L_clbrt, L_actor)

    if do_debug and folder_out is not None:
        cv2.imwrite(folder_out+'s01-original1.jpg', image_clbrt)
        cv2.imwrite(folder_out+'s05-original2.jpg', image_actor)

    face = R_c.morph_mesh(image_actor.shape[0],image_actor.shape[1],L2_aligned,L_clbrt,del_triangles)

    if do_debug and folder_out is not None:
        cv2.imwrite(folder_out + 's03-face.jpg', face)

    L2_aligned_mouth = L2_aligned[numpy.arange(48, 61, 1).tolist()]
    del_mouth = Delaunay(L2_aligned_mouth).vertices

    temp_mouth = R_a.morph_mesh(image_actor.shape[0],image_actor.shape[1],L2_aligned_mouth,L2_aligned_mouth, del_mouth)

    if do_debug:cv2.imwrite(folder_out + 's04-mouth.jpg', temp_mouth)

    filter_size = 0.1*L2_aligned[:,0].max()
    result2 = tools_image.blend_multi_band_large_small(image_actor, face, (0, 0, 0),filter_size=filter_size,n_clips=2)
    if do_debug:cv2.imwrite(folder_out + 's04-mouth_face.jpg', result2)

    filter_size = 0.05*L2_aligned_mouth[:,0].max()
    result2 = tools_image.blend_multi_band_large_small(result2, temp_mouth, (0, 0, 0),do_color_balance=False, filter_size=filter_size,n_clips=1)

    if do_debug and folder_out is not None:
        cv2.imwrite(folder_out+'s04-result2.jpg', result2)


    return result2
# ---------------------------------------------------------------------------------------------------------------------
def transferface_folder(D, filename_celebrity, folder_in, folder_out):

    tools_IO.remove_files(folder_out,create=True)

    local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')

    image_clbrt = cv2.imread(filename_celebrity)
    L_clbrt = D.get_landmarks_augm(image_clbrt)
    del_triangles_C = Delaunay(L_clbrt).vertices
    R_c = tools_GL.render_GL(image_clbrt)
    image_actor = cv2.imread(folder_in + local_filenames[0])
    R_a = tools_GL.render_GL(image_actor)

    M = 20
    L_actor_hist = numpy.zeros((M,68,2))

    for local_filename in local_filenames:

        image_actor = cv2.imread(folder_in+local_filename)
        L_actor = D.get_landmarks_augm(image_actor)
        #L_actor_hist = numpy.roll(L_actor_hist,1,axis=0)
        #L_actor_hist[0] = L_actor
        #L_actor = tools_filter.from_fistorical(L_actor_hist)

        R_a.update_texture(image_actor)

        H = tools_calibrate.get_transform_by_keypoints(L_clbrt, L_actor)
        if H is None:
            return image_actor

        L1_aligned, L2_aligned = tools_calibrate.translate_coordinates(image_clbrt, image_actor, H, L_clbrt, L_actor)
        face = R_c.morph_mesh(image_actor.shape[0], image_actor.shape[1], L2_aligned, L_clbrt, del_triangles_C)

        L2_aligned_mouth = L2_aligned[numpy.arange(48, 61, 1).tolist()]
        del_mouth = Delaunay(L2_aligned_mouth).vertices
        temp_mouth = R_a.morph_mesh(image_actor.shape[0], image_actor.shape[1], L2_aligned_mouth, L2_aligned_mouth,del_mouth)

        filter_size = int(face.shape[0] * 0.07)
        result = tools_image.blend_multi_band_large_small(image_actor, face, (0, 0, 0), filter_size=filter_size,do_color_balance=False)
        result = tools_image.blend_multi_band_large_small(result, temp_mouth, (0, 0, 0), do_color_balance=False,filter_size=filter_size // 2)

        cv2.imwrite(folder_out + local_filename, result)
        print(local_filename)

    return
# ---------------------------------------------------------------------------------------------------------------------
def landmarks_folder(D,folder_in, folder_out):
    tools_IO.remove_files(folder_out, create=True)

    local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')

    M = 20
    L_actor_hist = numpy.zeros((M, 68, 2))

    for local_filename in local_filenames:

        image_actor = cv2.imread(folder_in + local_filename)
        L_actor = D.get_landmarks_augm(image_actor)
        del_triangles_A = Delaunay(L_actor).vertices
        #L_actor_hist = numpy.roll(L_actor_hist, 1, axis=0)
        #L_actor_hist[0] = L_actor
        #L_actor = tools_filter.from_fistorical(L_actor_hist)

        result = D.draw_landmarks_v2(image_actor, L_actor, color=(0, 192, 255),del_triangles=del_triangles_A)


        cv2.imwrite(folder_out + local_filename, result)
        print(local_filename)
    return
# ---------------------------------------------------------------------------------------------------------------------