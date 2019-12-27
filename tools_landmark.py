#https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
import os
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
def transferface_first_to_second(R_c,R_a,image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C,folder_out='./images/output/',do_debug=False):

    swap = False

    if do_debug and folder_out is not None:
        tools_IO.remove_files(folder_out, create=True)

    if swap:
        image_clbrt,image_actor = image_actor,image_clbrt

    H = tools_calibrate.get_transform_by_keypoints(L_clbrt,L_actor)
    L1_aligned, L2_aligned = tools_calibrate.translate_coordinates(image_clbrt, image_actor, H, L_clbrt, L_actor)

    # face
    face = R_c.morph_mesh(image_actor.shape[0],image_actor.shape[1],L2_aligned,L_clbrt,del_triangles_C)
    filter_size = 0.1*L2_aligned[:,0].max()
    result = tools_image.blend_multi_band_large_small(image_actor, face, (0, 0, 0),filter_size=filter_size,n_clips=1)

    if do_debug: cv2.imwrite(folder_out + 's03-face.jpg', face)
    if do_debug: cv2.imwrite(folder_out + 's03-result.jpg', result)

    # mouth
    L2_aligned_mouth = L2_aligned[numpy.arange(48, 61, 1).tolist()]
    del_mouth = Delaunay(L2_aligned_mouth).vertices
    temp_mouth = R_a.morph_mesh(image_actor.shape[0], image_actor.shape[1], L2_aligned_mouth, L2_aligned_mouth,del_mouth)
    filter_size = 0.05*L2_aligned_mouth[:,0].max()
    result = tools_image.blend_multi_band_large_small(result, temp_mouth, (0, 0, 0), do_color_balance=False, filter_size=filter_size, n_clips=1,do_debug=do_debug)

    if do_debug: cv2.imwrite(folder_out + 's03-temp_mouth.jpg', temp_mouth)

    if do_debug:
        cv2.imwrite(folder_out+'s06-result2.jpg', result)
        cv2.imwrite(folder_out + 's06-original.jpg', image_actor)

    return result
# ---------------------------------------------------------------------------------------------------------------------
def process_folder(D, folder_in, folder_out, write_images=True, write_annotation=True,delim='\t'):

    tools_IO.remove_files(folder_out, create=True)
    local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')

    myfile = None

    if write_annotation:
        myfile = open(folder_out+"Landmarks.txt", "w")
        myfile.close()

    for local_filename in local_filenames:

        image = cv2.imread(folder_in + local_filename)
        L_actor = D.get_landmarks_augm(image)

        if write_images:
            del_triangles_A = Delaunay(L_actor).vertices
            result = D.draw_landmarks_v2(image, L_actor, color=(0, 192, 255),del_triangles=del_triangles_A)
            cv2.imwrite(folder_out + local_filename, result)

        if write_annotation:
            myfile = open(folder_out + "Landmarks.txt", "a")
            data = numpy.vstack((numpy.array([local_filename, local_filename]),(L_actor).astype(numpy.chararray))).T
            numpy.savetxt(myfile,data.astype(numpy.str),fmt='%s',encoding='str',delimiter=delim)
            myfile.close()

        print(local_filename)
    return
# ---------------------------------------------------------------------------------------------------------------------
def draw_landmarks(D,folder_in, filaname_landmarks, folder_out,delim='\t'):

    with open(filaname_landmarks) as f:
        lines = f.readlines()

    filenames_dict = sorted(set([line.split(delim)[0] for line in lines]))

    for local_filename in filenames_dict:
        if not os.path.isfile(folder_in + local_filename): continue

        L = []
        for i,line in enumerate(lines):
            if local_filename == line.split(delim)[0]:
                L.append(line.split(delim))

        if len(L)==2:
            L = (numpy.array(L)[:,1:].T).astype(numpy.float)
            image = cv2.imread(folder_in + local_filename)
            image = D.draw_landmarks_v2(image,L)
            cv2.imwrite(folder_out+local_filename,image)

    return
# ---------------------------------------------------------------------------------------------------------------------