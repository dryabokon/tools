import os
import cv2
import numpy
import tools_draw_numpy
import tools_GL
import tools_cupy
# ---------------------------------------------------------------------------------------------------------------------
from scipy.spatial import Delaunay
import tools_calibrate
import tools_image
import tools_IO
import tools_filter
# ---------------------------------------------------------------------------------------------------------------------
class Face_Swaper(object):
    def __init__(self,D,image_clbrt,image_actor,device = 'cpu',adjust_every_frame=False,do_narrow_face=False):
        self.device = device
        self.adjust_every_frame = adjust_every_frame
        self.do_narrow_face = do_narrow_face
        self.folder_out = './images/output/'
        self.D = D
        self.image_clbrt = image_clbrt
        self.image_actor = image_actor
        self.L_clbrt = self.D.get_landmarks(image_clbrt)
        if not (self.L_clbrt.min() == self.L_clbrt.max() == 0):
            self.del_triangles_C = Delaunay(self.L_clbrt[self.D.idx_removed_eyes]).vertices
        self.L_actor = self.D.get_landmarks(image_actor)
        self.R_c = tools_GL.render_GL(image_clbrt)
        self.R_a = tools_GL.render_GL(image_actor)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def update_clbrt(self, image_clbrt):
        self.image_clbrt = image_clbrt
        self.L_clbrt = self.D.get_landmarks(image_clbrt)

        if self.L_clbrt.min() == self.L_clbrt.max() == 0:
            print('Landmarks for clbrt image not found')
        else:
            self.del_triangles_C = Delaunay(self.L_clbrt[self.D.idx_removed_eyes]).vertices
            self.R_c.update_texture(self.image_clbrt)
            self.adjust_colors_clbrt()


        return
# ---------------------------------------------------------------------------------------------------------------------
    def update_actor(self, image_actor):
        self.image_actor = image_actor
        self.L_actor = self.D.get_landmarks(image_actor)
        self.R_a.update_texture(image_actor)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def narrow_face(self,image, landmarks):

        mask = numpy.zeros(image.shape[:2], dtype=numpy.float64)
        idx = numpy.arange(4, 13, 1).tolist() + numpy.arange(17, 68, 1).tolist()
        mask = tools_draw_numpy.draw_convex_hull(mask,numpy.array(landmarks[idx]),color=(1,1,1))
        mask = numpy.array([mask,mask,mask]).transpose((1,2,0))
        res = numpy.multiply(mask, image)
        return res
# ---------------------------------------------------------------------------------------------------------------------
    def process_folder_extract_landmarks(self,folder_in, folder_out, write_images=True, write_annotation=True, delim='\t'):

        tools_IO.remove_files(folder_out, create=True)
        local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')

        if write_annotation:
            myfile = open(folder_out+"Landmarks.txt", "w")
            myfile.close()

        for local_filename in local_filenames:

            image = cv2.imread(folder_in + local_filename)
            L_actor = self.D.get_landmarks(image)

            if write_images:
                del_triangles_A = Delaunay(L_actor).vertices
                result = self.D.draw_landmarks_v2(image, L_actor, color=(0, 192, 255),del_triangles=del_triangles_A)
                cv2.imwrite(folder_out + local_filename, result)

            if write_annotation:
                myfile = open(folder_out + "Landmarks.txt", "a")
                data = numpy.vstack((numpy.array([local_filename, local_filename]),(L_actor).astype(numpy.chararray))).T
                numpy.savetxt(myfile,data.astype(numpy.str),fmt='%s',encoding='str',delimiter=delim)
                myfile.close()

            print(local_filename)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def process_folder_faceswap(self,filename_clbrt,folder_in, folder_out):

        tools_IO.remove_files(folder_out, create=True)
        local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')
        image_clbrt = cv2.imread(filename_clbrt)
        L_clbrt = self.D.get_landmarks(image_clbrt)
        del_triangles_C = Delaunay(L_clbrt).vertices
        R_c = tools_GL.render_GL(image_clbrt)

        image_actor = cv2.imread(folder_in + local_filenames[0])
        R_a = tools_GL.render_GL(image_actor)

        for local_filename in local_filenames:

            image_actor = cv2.imread(folder_in + local_filename)
            L_actor = self.D.get_landmarks(image_actor)
            R_a.update_texture(image_actor)

            result = self.do_faceswap(R_c, R_a, image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C)
            cv2.imwrite(folder_out + local_filename, result)

            print(local_filename)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def process_folder_draw_landmarks(self,folder_in, list_of_filaname_landmarks, folder_out, delim='\t'):

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
                    image = self.D.draw_landmarks_v2(image,L,color=color[i])

            cv2.imwrite(folder_out+local_filename,image)
            print(local_filename)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def process_folder_faceswap_by_landmarks(self,folder_in, filaname_landmarks, folder_out, delim='\t'):

        if not os.path.exists(folder_out):
            os.mkdir(folder_out)

        with open(filaname_landmarks) as f:lines = f.readlines()
        filenames_dict = sorted(set([line.split(delim)[0] for line in lines]))

        flag =0

        for local_filename in filenames_dict:
            if not os.path.isfile(folder_in + local_filename): continue

            L = []
            for i,line in enumerate(lines):
                if local_filename == line.split(delim)[0]:
                    L.append(line.split(delim))

            if len(L)==2:
                image_actor = cv2.imread(folder_in + local_filename)
                self.update_actor(image_actor)

                if flag==0:
                    flag=1
                    self.adjust_colors_clbrt()

                result = self.do_faceswap()
                cv2.imwrite(folder_out + local_filename, result)
                print(local_filename)


        return
# ---------------------------------------------------------------------------------------------------------------------
    def filter_landmarks(self,filename_in, filename_out,N=5,delim='\t'):

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
    def interpolate(self,filename_in, filename_out,N=5,delim='\t'):

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
    def adjust_colors_clbrt(self,avg_size=50):

        if self.L_actor.min() == self.L_actor.max() == 0 or self.L_clbrt.min() == self.L_clbrt.max() == 0:
            return

        H = tools_calibrate.get_transform_by_keypoints(self.L_clbrt,self.L_actor)
        LC_aligned, LA_aligned = tools_calibrate.translate_coordinates(self.image_clbrt, self.image_actor, H, self.L_clbrt, self.L_actor)

        face = self.R_c.morph_mesh(self.image_actor.shape[0],self.image_actor.shape[1],LA_aligned[self.D.idx_removed_eyes],self.L_clbrt[self.D.idx_removed_eyes],self.del_triangles_C)
        self.filter_face_size = int(0.2*(LA_aligned[:,0].max()-LA_aligned[:,0].min()))

        mask_bin = 1 * (face[:, :] == (0,0,0))
        mask_bin = numpy.array(numpy.min(mask_bin, axis=2), dtype=numpy.int)
        cnt_face = tools_filter.sliding_2d(1 - mask_bin, avg_size, avg_size, 'cnt')
        scale = numpy.zeros(face.shape)

        face = face.astype(numpy.long)
        for c in range(3):

            avg_actor = tools_filter.sliding_2d(self.image_actor[:, :, c], avg_size, avg_size, 'avg')
            sum_face  = tools_filter.sliding_2d(face[:, :, c], avg_size, avg_size, 'cnt')
            avg_face = sum_face/ cnt_face
            s  = avg_actor / avg_face
            s = numpy.nan_to_num(s)
            s[s==0]=1
            scale[:,:,c] = s

        #scale -> im
        #[]
        scale = numpy.clip(scale,0,3)
        minvalue = scale.min()
        scale-=minvalue
        maxvalue = scale.max()
        scale*=255/maxvalue
        scale = scale.astype(numpy.uint8)
        #cv2.imwrite(self.folder_out + 'scale.png', scale)

        R = tools_GL.render_GL(scale)

        reverce_scale = R.morph_mesh(self.image_clbrt.shape[0], self.image_clbrt.shape[1],
                                   self.L_clbrt[self.D.idx_removed_eyes],LA_aligned[self.D.idx_removed_eyes],
                                   self.del_triangles_C)


        mask = 1 * (reverce_scale[:, :] == (0,0,0))



        #cv2.imwrite(self.folder_out + 'scale_inv.png', reverce_scale)


        reverce_scale=reverce_scale.astype(float)

        reverce_scale /= (255 / maxvalue)
        reverce_scale += minvalue
        reverce_scale[numpy.where(mask==1)]=1


        self.image_clbrt = self.image_clbrt.astype(float)
        self.image_clbrt*= reverce_scale
        self.image_clbrt = numpy.clip(self.image_clbrt,0,255)
        self.image_clbrt =self.image_clbrt.astype(numpy.uint8)
        #cv2.imwrite(self.folder_out + 'image_clbrt_scaled.png', self.image_clbrt)

        self.R_c.update_texture(self.image_clbrt)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def do_faceswap(self,folder_out='./images/output/', do_debug=False):
        if self.device=='cpu':
            return self.do_faceswap_cpu(folder_out,do_debug)
        else:
            return self.do_faceswap_gpu()
#----------------------------------------------------------------------------------------------------------------------
    def do_faceswap_cpu(self,folder_out='./images/output/', do_debug=False):

        if do_debug and folder_out is not None:
            tools_IO.remove_files(folder_out, create=True)

        if self.L_actor.min()==self.L_actor.max()==0 or self.L_clbrt.min()==self.L_clbrt.max()==0:
            return self.image_actor

        H = tools_calibrate.get_transform_by_keypoints(self.L_clbrt,self.L_actor)
        LC_aligned, LA_aligned = tools_calibrate.translate_coordinates(self.image_clbrt, self.image_actor, H, self.L_clbrt, self.L_actor)

        # face
        face = self.R_c.morph_mesh(self.image_actor.shape[0],self.image_actor.shape[1],LA_aligned[self.D.idx_removed_eyes],self.L_clbrt[self.D.idx_removed_eyes],self.del_triangles_C)
        if self.do_narrow_face:
            face = self.narrow_face(face,LA_aligned)
        if do_debug: cv2.imwrite(folder_out + 's03-face.jpg', face)

        if do_debug: cv2.imwrite(folder_out + 's03-face_masked.jpg', face)
        filter_face_size = int(0.2*(LA_aligned[:,0].max()-LA_aligned[:,0].min()))
        result = tools_image.blend_multi_band_large_small(self.image_actor, face, (0, 0, 0),adjust_colors=self.adjust_every_frame,filter_size=filter_face_size,do_debug=do_debug)
        if do_debug: cv2.imwrite(folder_out + 's03-result.jpg', result)

        # mouth
        LA_aligned_mouth = LA_aligned[numpy.arange(48, 61, 1).tolist()]
        del_mouth = Delaunay(LA_aligned_mouth).vertices
        temp_mouth = self.R_a.morph_mesh(self.image_actor.shape[0], self.image_actor.shape[1], LA_aligned_mouth, LA_aligned_mouth,del_mouth)
        if do_debug: cv2.imwrite(folder_out + 's03-temp_mouth.jpg', temp_mouth)
        filter_mouth_size = filter_face_size//2
        result = tools_image.blend_multi_band_large_small(result, temp_mouth, (0, 0, 0),adjust_colors=False,filter_size=filter_mouth_size)
        if do_debug: cv2.imwrite(folder_out + 's03-result-mouth.jpg', result)


        if do_debug:
            cv2.imwrite(folder_out+'s06-result2.jpg', result)
            cv2.imwrite(folder_out + 's06-original.jpg', self.image_actor)

        return result
# ---------------------------------------------------------------------------------------------------------------------
    def do_faceswap_gpu(self):

        if self.L_actor.min() == self.L_actor.max() == 0 or self.L_clbrt.min() == self.L_clbrt.max() == 0:
            return self.image_actor

        H = tools_calibrate.get_transform_by_keypoints(self.L_clbrt, self.L_actor)
        LC_aligned, LA_aligned = tools_calibrate.translate_coordinates(self.image_clbrt, self.image_actor, H,self.L_clbrt, self.L_actor)

        face = self.R_c.morph_mesh(self.image_actor.shape[0], self.image_actor.shape[1],LA_aligned[self.D.idx_removed_eyes], self.L_clbrt[self.D.idx_removed_eyes],self.del_triangles_C)
        if self.do_narrow_face:
            face = self.narrow_face(face,LA_aligned)

        filter_face_size = int(0.2 * (LA_aligned[:, 0].max() - LA_aligned[:, 0].min()))


        result = tools_cupy.blend_multi_band_large_small(self.image_actor, face, (0, 0, 0),adjust_colors=self.adjust_every_frame,filter_size=filter_face_size)

        LA_aligned_mouth = LA_aligned[numpy.arange(48, 61, 1).tolist()]
        del_mouth = Delaunay(LA_aligned_mouth).vertices
        temp_mouth = self.R_a.morph_mesh(self.image_actor.shape[0], self.image_actor.shape[1], LA_aligned_mouth,LA_aligned_mouth, del_mouth)
        result = tools_cupy.blend_multi_band_large_small(result, temp_mouth, (0, 0, 0),adjust_colors=False, filter_size=filter_face_size // 2)

        return tools_cupy.asnumpy(result)
# ---------------------------------------------------------------------------------------------------------------------