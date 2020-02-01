#https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
import os
import cv2
import numpy
import tools_draw_numpy
import tools_GL
import tools_signal
import pandas as pd
from scipy import ndimage
# ---------------------------------------------------------------------------------------------------------------------
from scipy.spatial import Delaunay
import detector_landmarks
import tools_calibrate
import tools_image
import tools_IO
import tools_filter
# ---------------------------------------------------------------------------------------------------------------------
def narrow_face(image, landmarks):

    mask = numpy.zeros(image.shape[:2], dtype=numpy.float64)
    idx = numpy.arange(4, 13, 1).tolist() + numpy.arange(17, 68, 1).tolist()
    mask = tools_draw_numpy.draw_convex_hull(mask,numpy.array(landmarks[idx]),color=(1,1,1))
    mask = numpy.array([mask,mask,mask]).transpose((1,2,0))
    res = numpy.multiply(mask, image)
    return res
# ---------------------------------------------------------------------------------------------------------------------
def get_mask_eye1(LA_aligned,image_actor,R_a):

    LA_aligned_eye1= LA_aligned[numpy.arange(36, 42, 1).tolist()]
    del_eye1= Delaunay(LA_aligned_eye1).vertices
    mask = R_a.morph_mesh(image_actor.shape[0], image_actor.shape[1], LA_aligned_eye1, LA_aligned_eye1,del_eye1)
    mask = 255*(mask[:, :] != (0,0,0))

    return mask
# ---------------------------------------------------------------------------------------------------------------------
def get_mask_eye2(LA_aligned,image_actor,R_a):

    LA_aligned_eye2= LA_aligned[numpy.arange(42, 48, 1).tolist()]
    del_eye2= Delaunay(LA_aligned_eye2).vertices
    mask = R_a.morph_mesh(image_actor.shape[0], image_actor.shape[1], LA_aligned_eye2, LA_aligned_eye2,del_eye2)
    mask = 255*(mask[:, :] != (0,0,0))
    return mask
# ---------------------------------------------------------------------------------------------------------------------
def process_folder_extract_landmarks(D, folder_in, folder_out, write_images=True, write_annotation=True, delim='\t'):

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
def process_folder_faceswap(D,filename_clbrt,folder_in, folder_out):

    tools_IO.remove_files(folder_out, create=True)
    local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')
    image_clbrt = cv2.imread(filename_clbrt)
    L_clbrt = D.get_landmarks_augm(image_clbrt)
    del_triangles_C = Delaunay(L_clbrt).vertices
    R_c = tools_GL.render_GL(image_clbrt)

    image_actor = cv2.imread(folder_in + local_filenames[0])
    R_a = tools_GL.render_GL(image_actor)

    for local_filename in local_filenames:

        image_actor = cv2.imread(folder_in + local_filename)
        L_actor = D.get_landmarks_augm(image_actor)
        R_a.update_texture(image_actor)

        result = do_faceswap(R_c, R_a, image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C)
        cv2.imwrite(folder_out + local_filename, result)

        print(local_filename)
    return
# ---------------------------------------------------------------------------------------------------------------------
def process_folder_draw_landmarks(D, folder_in, list_of_filaname_landmarks, folder_out, delim='\t'):

    lines=[]

    for i,filaname_landmarks in enumerate(list_of_filaname_landmarks):
        with open(filaname_landmarks) as f:
            lines.append(f.readlines())

    filenames_dict = sorted(set([line.split(delim)[0] for line in lines[0]]))

    color = [(255,0,0),(0,255,0)]

    for local_filename in filenames_dict:
        if not os.path.isfile(folder_in + local_filename): continue
        image = cv2.imread(folder_in + local_filename)



        for i in range(len(list_of_filaname_landmarks)):

            L = []
            for line in lines[i]:
                if local_filename == line.split(delim)[0]:
                    L.append(line.split(delim))

            if len(L)==2:
                L = (numpy.array(L)[:,1:].T).astype(numpy.float)
                image = D.draw_landmarks_v2(image,L,color=color[i])

        cv2.imwrite(folder_out+local_filename,image)
        print(local_filename)
    return
# ---------------------------------------------------------------------------------------------------------------------
def process_folder_faceswap_by_landmarks(D, filename_clbrt,folder_in, filaname_landmarks, folder_out, delim='\t'):

    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    image_clbrt = cv2.imread(filename_clbrt)
    L_clbrt = D.get_landmarks_augm(image_clbrt)
    del_triangles_C = Delaunay(L_clbrt).vertices
    R_c = tools_GL.render_GL(image_clbrt)
    local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')
    image_actor = cv2.imread(folder_in + local_filenames[0])
    R_a = tools_GL.render_GL(image_actor)

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
            L_actor = (numpy.array(L)[:,1:].T).astype(numpy.float)
            image_actor = cv2.imread(folder_in + local_filename)
            R_a.update_texture(image_actor)

            result = do_faceswap(R_c, R_a, image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C,do_debug=False)
            cv2.imwrite(folder_out + local_filename, result)

            print(local_filename)


    return
# ---------------------------------------------------------------------------------------------------------------------
def filter_landmarks(filename_in, filename_out,N=5,delim='\t'):

    dataset = tools_IO.load_mat(filename_in, delim=delim, dtype=numpy.str)
    result = []
    idx_x = numpy.arange(0,dataset.shape[0], 2)
    idx_y = numpy.arange(1,dataset.shape[0], 2)

    for c in range(1, dataset.shape[1]):
        D = numpy.array(dataset[:,c],dtype=numpy.float)
        if True:#c in [0,1,2,3,14,15,16,17]:
            X = D[idx_x]
            Y = D[idx_y]
            Rx = tools_filter.do_filter_median(X,N)
            Ry = tools_filter.do_filter_median(Y,N)
            D[0::2] = Rx
            D[1::2] = Ry
        result.append(D)

    result = (numpy.array(result).T).astype(numpy.str)

    names = dataset[:, 0]
    result = numpy.insert(result,0,names,axis=1)

    myfile = open(filename_out, "w")
    numpy.savetxt(myfile, result, fmt='%s', encoding='str', delimiter=delim)
    myfile.close()

    return
# ---------------------------------------------------------------------------------------------------------------------
def interpolate(filename_in, filename_out,N=5,delim='\t'):

    dataset = tools_IO.load_mat(filename_in, delim=delim, dtype=numpy.str)
    result = []
    idx_x = numpy.arange(0,dataset.shape[0], 2)
    idx_y = numpy.arange(1,dataset.shape[0], 2)

    for c in range(1, dataset.shape[1]):
        D = numpy.array(dataset[:,c],dtype=numpy.float)

        X = D[idx_x]
        Y = D[idx_y]
        Rx = tools_filter.fill_zeros(X)
        Ry = tools_filter.fill_zeros(Y)
        D[0::2] = Rx
        D[1::2] = Ry
        result.append(D)

    result = (numpy.array(result).T).astype(numpy.str)

    names = dataset[:, 0]
    result = numpy.insert(result,0,names,axis=1)

    myfile = open(filename_out, "w")
    numpy.savetxt(myfile, result, fmt='%s', encoding='str', delimiter=delim)
    myfile.close()

    return
# ---------------------------------------------------------------------------------------------------------------------
def do_faceswap(R_c, R_a, image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C, folder_out='./images/output/', do_debug=False):

    swap = False

    if do_debug and folder_out is not None:
        tools_IO.remove_files(folder_out, create=True)

    if swap:
        image_clbrt,image_actor = image_actor,image_clbrt

    if L_actor.min()==L_actor.max()==0 or L_clbrt.min()==L_clbrt.max()==0:
        return image_actor

    H = tools_calibrate.get_transform_by_keypoints(L_clbrt,L_actor)
    LC_aligned, LA_aligned = tools_calibrate.translate_coordinates(image_clbrt, image_actor, H, L_clbrt, L_actor)

    idx_removed_eyes = numpy.arange(0, 68, 1).tolist()
    for each in [37, 38, 40, 41, 43, 44, 46, 47]:idx_removed_eyes.remove(each)

    # face
    del_face = Delaunay(L_clbrt[idx_removed_eyes]).vertices
    face = R_c.morph_mesh(image_actor.shape[0],image_actor.shape[1],LA_aligned[idx_removed_eyes],L_clbrt[idx_removed_eyes],del_face)
    if do_debug: cv2.imwrite(folder_out + 's03-face.jpg', face)

    if do_debug: cv2.imwrite(folder_out + 's03-face_masked.jpg', face)
    filter_face_size = int(0.2*(LA_aligned[:,0].max()-LA_aligned[:,0].min()))
    result = tools_image.blend_multi_band_large_small(image_actor, face, (0, 0, 0),adjust_colors='xxx',filter_size=filter_face_size,do_debug=do_debug)
    if do_debug: cv2.imwrite(folder_out + 's03-result.jpg', result)

    # mouth
    LA_aligned_mouth = LA_aligned[numpy.arange(48, 61, 1).tolist()]
    del_mouth = Delaunay(LA_aligned_mouth).vertices
    temp_mouth = R_a.morph_mesh(image_actor.shape[0], image_actor.shape[1], LA_aligned_mouth, LA_aligned_mouth,del_mouth)
    if do_debug: cv2.imwrite(folder_out + 's03-temp_mouth.jpg', temp_mouth)
    filter_mouth_size = filter_face_size//2
    result = tools_image.blend_multi_band_large_small(result, temp_mouth, (0, 0, 0), adjust_colors=None, filter_size=filter_mouth_size, n_clips=1)
    if do_debug: cv2.imwrite(folder_out + 's03-result-mouth.jpg', result)


    if do_debug:
        cv2.imwrite(folder_out+'s06-result2.jpg', result)
        cv2.imwrite(folder_out + 's06-original.jpg', image_actor)

    return result
# ---------------------------------------------------------------------------------------------------------------------
