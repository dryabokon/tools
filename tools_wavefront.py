import os

import cv2
import numpy
#import pyvista
#----------------------------------------------------------------------------------------------------------------------
class ObjLoader:
    def __init__(self,filename_obj=None,do_autoscale=False,target_object_name=None):
        if filename_obj is not None:
            self.load_mesh(filename_obj,do_autoscale=do_autoscale,target_object_name=target_object_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_mesh(self, filename_obj, do_autoscale=False,target_object_name=None):

        folder_name=''
        split = filename_obj.split('/')
        for s in split[:-1]:folder_name+=s+'/'

        self.mat_color = []
        self.filename_texture = []
        self.filename_mat = None
        self.mat_name = None
        self.dct_textures = {}

        self.coord_vert = []
        self.coord_texture = []
        self.coord_norm = []
        self.idx_vertex = []
        self.idx_texture = []
        self.idx_normal = []


        object_id=-1
        self.dct_obj_id = {}
        flag_target = False
        self.idx_target_start = None
        self.idx_target_end = None

        for line in open(filename_obj, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            values = numpy.array(values)

            if values[0] == 'o':
                if target_object_name is not None:
                    flag_target = (values[1]==target_object_name)

                object_id+=1
                self.mat_color.append(None)
                self.filename_texture.append(None)


            if values[0] == 'v':
                self.coord_vert.append([float(v) for v in values[1:4]])
                if flag_target:
                    if self.idx_target_start is None:
                        self.idx_target_start = len(self.coord_vert)-1
                    self.idx_target_end = len(self.coord_vert) - 1

            if values[0] == 'vt': self.coord_texture.append([float(v) for v in values[1:3]])
            if values[0] == 'vn': self.coord_norm.append([float(v) for v in values[1:4]])


            if values[0] == 'f':
                face_i = []
                text_i = []
                norm_i = []

                if len(values[1:]) == 3:
                    for v in values[1:4]:
                        w = v.split('/')
                        face_i.append(int(w[0]) - 1)
                        if len(w[1])>0:
                            text_i.append(int(w[1]) - 1)
                        else:
                            text_i.append(0)

                        norm_i.append(int(w[2]) - 1)
                elif len(values[1:]) == 4:
                    for v in values[[1, 2, 3]]:
                        w = v.split('/')
                        face_i.append(int(w[0]) - 1)
                        text_i.append(int(w[1]) - 1)
                        norm_i.append(int(w[2]) - 1)

                    for v in values[[1, 2, 4]]:
                        w = v.split('/')
                        face_i.append(int(w[0]) - 1)
                        text_i.append(int(w[1]) - 1)
                        norm_i.append(int(w[2]) - 1)

                    for v in values[[1, 3, 4]]:
                        w = v.split('/')
                        face_i.append(int(w[0]) - 1)
                        text_i.append(int(w[1]) - 1)
                        norm_i.append(int(w[2]) - 1)

                    for v in values[[2, 3, 4]]:
                        w = v.split('/')
                        face_i.append(int(w[0]) - 1)
                        text_i.append(int(w[1]) - 1)
                        norm_i.append(int(w[2]) - 1)

                    for v in values[[1, 3, 4]]:
                        w = v.split('/')
                        face_i.append(int(w[0]) - 1)
                        text_i.append(int(w[1]) - 1)
                        norm_i.append(int(w[2]) - 1)

                face_i = [f for f in face_i]
                text_i = [t for t in text_i]
                self.idx_vertex.append(face_i)
                self.idx_texture.append(text_i)
                self.idx_normal.append(norm_i)
                for f in face_i:
                    self.dct_obj_id[f]=object_id

            if values[0]=='mtllib':
                self.filename_mat = folder_name+values[1]
            if values[0]=='usemtl':
                self.mat_name = values[1]
                mat,texture = self.get_material(self.filename_mat,self.mat_name)
                self.mat_color[-1]=mat
                if texture is not None:
                    self.filename_texture[-1]=texture
                    self.dct_textures[self.mat_name] = texture

        if len(self.coord_texture) == 0: self.coord_texture.append([0,0])

        self.coord_texture = numpy.array(self.coord_texture)
        self.coord_vert = numpy.array(self.coord_vert)
        self.coord_norm = numpy.array(self.coord_norm)

        if do_autoscale:
            min_value = numpy.min(self.coord_vert,axis=0)
            max_value = numpy.max(self.coord_vert,axis=0)
            shift = (max_value+min_value)/2.0
            self.coord_vert = self.coord_vert - shift
            S = numpy.max(numpy.abs(self.coord_vert))
            self.coord_vert = self.coord_vert / S

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_material(self,filename_mat,mat_name=None):
        mat_color,filename_texture = None,None
        ready_to_read = mat_name is None

        if not os.path.exists(filename_mat):
            return mat_color, filename_texture

        for line in open(filename_mat, 'r'):
            values = line.split()
            if not values: continue
            if ready_to_read and values[0] == 'newmtl':break
            if (mat_name is not None) and not ready_to_read:
                if mat_name in values[1:]:
                    ready_to_read = True
            else:
                if ready_to_read:
                    if values[0] == 'Kd' and mat_color is None:
                        mat_color=tuple([float(v) for v in values[1:4]])
                    if values[0] == 'map_Kd' and filename_texture is None:
                        filename_texture = values[1]

        if filename_texture is not None:
            filename_texture = os.path.dirname(filename_mat) + '/' + filename_texture
            if os.path.exists(filename_texture):
                im = cv2.imread(filename_texture)
                mat_color = im[:,:,[2,1,0]].mean(axis=(0, 1))/255.0

        return mat_color, filename_texture
# ----------------------------------------------------------------------------------------------------------------------
    def scale_mesh(self,svec):
        for i in range(self.coord_vert.shape[0]):
            self.coord_vert[i]=numpy.multiply(self.coord_vert[i],svec)
        return
# ----------------------------------------------------------------------------------------------------------------------
#     def rotate_mesh_rvec(self, rvec):
#
#         R = pyrr.matrix44.create_from_eulers(rvec)
#
#         X = numpy.array(self.coord_vert)
#         X4D = numpy.hstack((X, numpy.full((X.shape[0], 1), 1)))
#         X = R.dot(X4D.T).T
#         self.coord_vert = X[:, :3]
#
#         return
# # ----------------------------------------------------------------------------------------------------------------------
#     def rotate_mesh(self, R):
#
#         X = numpy.array(self.coord_vert)
#         X4D = numpy.hstack((X, numpy.full((X.shape[0], 1), 1)))
#         X = R.dot(X4D.T).T
#         self.coord_vert = X[:, :3]
#
#         return
# ----------------------------------------------------------------------------------------------------------------------
#     def translate_mesh(self, tvec):
#         self.coord_vert += tvec
#         return
# ----------------------------------------------------------------------------------------------------------------------
    def transform_mesh(self,M):
        if M is None:return 
        
        if M.shape[0] == 3:
            Mtemp = numpy.eye(4)
            Mtemp[:3, :3] = M
            M = Mtemp

        X4D = numpy.column_stack((self.coord_vert, numpy.ones(len(self.coord_vert))))
        X = X4D.dot(M)
        self.coord_vert = X[:, :3]
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_trianges(self, X):
        cloud = pyvista.PolyData(X)

        surf = cloud.delaunay_2d()
        F = numpy.array(surf.faces)
        if len(F) > 0 and F[0] == 3:

            del_triangles = F.reshape((len(F) // 4, 4))
            del_triangles = del_triangles[:, 1:]
            normals = numpy.array(surf.face_normals)
        else:
            del_triangles, normals = [], []
        return del_triangles, normals
# ----------------------------------------------------------------------------------------------------------------------
    def convert_v0(self, filename_in, filename_out,filename_material=None,bias_vertex=0,bias_texture=0,bias_normal=0,center=None,max_size=None):
        coord_vert = []
        coord_texture = []
        coord_norm = []

        idx_vertex = []
        idx_texture = []
        idx_normal = []

        lines = open(filename_in).readlines()

        for line in lines:
            values = line.split()
            if not values: continue
            values = numpy.array(values)
            if values[0] == 'v': coord_vert.append([float(v) for v in values[1:4]])
            if values[0] == 'vt': coord_texture.append([float(v) for v in values[1:3]])
            if values[0] == 'vn': coord_norm.append([float(v) for v in values[1:4]])

        coord_vert = numpy.array(coord_vert)
        coord_texture = numpy.array(coord_texture)
        coord_norm = numpy.array(coord_norm)

        for line in lines:
            values = line.split()
            if not values: continue
            values = numpy.array(values)
            if values[0] == 'f':
                X = []
                Iv,It,In = [],[],[]
                for face in values[1:]:
                    Iv.append(int(face.split('/')[0]) - 1)
                    It.append(int(face.split('/')[1]) - 1)
                    In.append(int(face.split('/')[2]) - 1)
                    X.append(coord_vert[Iv[-1]])
                X = numpy.array(X)
                Iv = numpy.array(Iv)
                if len(Iv) == 3:
                    idx_vertex.append(Iv)
                    idx_texture.append(It)
                    idx_normal.append(In)

                else:
                    triangles, normals = self.get_trianges(X)
                    for triangle in triangles:
                        idx_vertex.append(I[triangle])

        max_size = 1 if max_size is None else max_size

        for c in [0,1,2]:
            coord_vert[:, c] -= coord_vert[:,c].mean()
            coord_vert[:, c]/= (coord_vert[:,c].max()/max_size)

        if center is not None:
            coord_vert+=center

        idx_vertex = numpy.array(idx_vertex)+bias_vertex
        idx_texture = numpy.array(idx_texture)+bias_texture
        idx_normal = numpy.array(idx_normal)+bias_normal

        self.export_mesh(filename_out, coord_vert, coord_texture=coord_texture,coord_norm=coord_norm,idx_vertex=idx_vertex,idx_texture=idx_texture,idx_normal=idx_normal,filename_material=filename_material)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def convert(self, filename_in, filename_out):

        lines = open(filename_in).readlines()
        with open(filename_out, "w+") as f_handle:
            for line in lines:
                values = line.split()
                if not values: continue

                if values[0] == 'f':
                    nodes = values[1:]
                    if len(nodes)==3:
                        f_handle.write(line)
                    elif len(nodes)==4:
                        f_handle.write('f %s %s %s\n' % (nodes[0], nodes[1], nodes[2]))
                        f_handle.write('f %s %s %s\n' % (nodes[0], nodes[2], nodes[3]))
                    else:
                        pass

                else:
                    f_handle.write(line)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_mesh(self, filename_out, X, coord_texture, coord_norm,idx_vertex,idx_texture,idx_normal,filename_material=None,material_name=None,shared_material=False,mode='w+'):

        fmt = '%1.3f'

        if isinstance(filename_out, (str, os.PathLike)):
            f_handle = open(filename_out, mode, newline='\n')  # writing to a real file
            need_close = True
        else:
            f_handle = filename_out
            need_close = False

        f_handle.write("# Obj file\n")
        f_handle.write("o %s\n" % (material_name if material_name is not None else 'Object'))
        if not shared_material:
            if filename_material is not None:f_handle.write('mtllib %s\n' % filename_material.split('/')[-1])
        if material_name     is not None:f_handle.write('usemtl %s\n' % material_name)
        for x in X: f_handle.write(f'v {fmt} {fmt} {fmt}\n' % (x[0], x[1], x[2]))

        if coord_texture is None:
            f_handle.write("vt 0 0\n")
        else:
            for t in coord_texture: f_handle.write("vt %1.2f %1.2f\n" % (t[0], t[1]))

        for n in coord_norm:f_handle.write("vn %1.2f %1.2f %1.2f\n" % (n[0], n[1], n[2]))

        for iv,it,inr in zip(idx_vertex,idx_texture,idx_normal):
            f_handle.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (iv[0] + 1, it[0]+1, inr[0] + 1, iv[1] + 1, it[1]+1, inr[1] + 1, iv[2] + 1, it[2]+1, inr[2] + 1))

        if need_close:
            f_handle.close()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_material(self,filename_out,mat_name,color255,transparency=0,filename_texture=None,mode='w+'):

        with open(filename_out, mode) as f:
            f.write('# Material\n')
            f.write('newmtl %s\n'%mat_name)
            f.write('Kd %1.4f %1.4f %1.4f\n' % (color255[0] / 255.0, color255[1] / 255.0, color255[2] / 255.0))
            if transparency>0:
                f.write('d %.1f\n'%(1-transparency))
            if filename_texture is not None:
                f.write('map_Kd %s\n'%filename_texture.split('/')[-1])

        return
# ----------------------------------------------------------------------------------------------------------------------
