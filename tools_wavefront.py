import os
import numpy
#----------------------------------------------------------------------------------------------------------------------
class ObjLoader:
    def __init__(self):
        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_mesh(self, filename_obj, do_autoscale=True):

        folder_name=''
        split = filename_obj.split('/')
        for s in split[:-1]:folder_name+=s+'/'

        self.mat_color = []
        self.filename_texture = []

        self.coord_vert = []
        self.coord_texture = []
        self.coord_norm = []
        self.idx_vertex = []
        self.idx_texture = []
        self.idx_normal = []
        offset_vert=0
        offset_text=0
        object_id=-1
        self.dct_obj_id = {}

        for line in open(filename_obj, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            values = numpy.array(values)

            if values[0] == 'o':
                offset_vert=len(self.coord_vert)
                offset_text=len(self.coord_texture)
                object_id+=1
                self.mat_color.append(None)
                self.filename_texture.append(None)


            if values[0] == 'v':  self.coord_vert.append([float(v) for v in values[1:4]])
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

                face_i = [f+offset_vert for f in face_i]
                text_i = [t+offset_text for t in text_i]
                self.idx_vertex.append(face_i)
                self.idx_texture.append(text_i)
                self.idx_normal.append(norm_i)
                for f in face_i:
                    self.dct_obj_id[f]=object_id

            if values[0]=='mtllib':
                mat,texture = self.get_material(folder_name+values[1])
                self.mat_color[-1]=mat
                if texture is not None:
                    self.filename_texture[-1]=(folder_name + texture)

        if len(self.coord_texture) == 0: self.coord_texture.append([0,0])

        self.coord_texture = numpy.array(self.coord_texture)
        self.coord_vert = numpy.array(self.coord_vert)
        self.coord_norm = numpy.array(self.coord_norm)

        if do_autoscale:
            max_value = self.coord_vert.max()
            self.coord_vert/= max_value

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_material(self,filename_mat):
        mat_color,filename_texture = None,None

        if not os.path.exists(filename_mat):
            return mat_color, filename_texture

        for line in open(filename_mat, 'r'):
            values = line.split()
            if not values: continue
            values = numpy.array(values)
            if values[0] == 'Ka':mat_color=tuple([float(v) for v in values[1:4]])
            if values[0] == 'map_Kd': filename_texture = values[1]

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

        X4D = numpy.hstack((numpy.array(self.coord_vert), numpy.full((len(self.coord_vert), 1), 1)))
        X = ((M.T).dot(X4D.T)).T
        self.coord_vert = X[:, :3]
        return
# ----------------------------------------------------------------------------------------------------------------------
#     def get_trianges(self, X):
#         cloud = pyvista.PolyData(X)
#
#         surf = cloud.delaunay_2d()
#         F = numpy.array(surf.faces)
#         if len(F) > 0 and F[0] == 3:
#
#             del_triangles = F.reshape((len(F) // 4, 4))
#             del_triangles = del_triangles[:, 1:]
#             normals = numpy.array(surf.face_normals)
#         else:
#             del_triangles, normals = [], []
#         return del_triangles, normals
# ----------------------------------------------------------------------------------------------------------------------
#     def convert_v0(self, filename_in, filename_out,filename_material=None,bias_vertex=0,bias_texture=0,bias_normal=0,center=None,max_size=None):
#         coord_vert = []
#         coord_texture = []
#         coord_norm = []
#
#         idx_vertex = []
#         idx_texture = []
#         idx_normal = []
#
#         lines = open(filename_in).readlines()
#
#         for line in lines:
#             values = line.split()
#             if not values: continue
#             values = numpy.array(values)
#             if values[0] == 'v': coord_vert.append([float(v) for v in values[1:4]])
#             if values[0] == 'vt': coord_texture.append([float(v) for v in values[1:3]])
#             if values[0] == 'vn': coord_norm.append([float(v) for v in values[1:4]])
#
#         coord_vert = numpy.array(coord_vert)
#         coord_texture = numpy.array(coord_texture)
#         coord_norm = numpy.array(coord_norm)
#
#         for line in lines:
#             values = line.split()
#             if not values: continue
#             values = numpy.array(values)
#             if values[0] == 'f':
#                 X = []
#                 Iv,It,In = [],[],[]
#                 for face in values[1:]:
#                     Iv.append(int(face.split('/')[0]) - 1)
#                     It.append(int(face.split('/')[1]) - 1)
#                     In.append(int(face.split('/')[2]) - 1)
#                     X.append(coord_vert[Iv[-1]])
#                 X = numpy.array(X)
#                 Iv = numpy.array(Iv)
#                 if len(Iv) == 3:
#                     idx_vertex.append(Iv)
#                     idx_texture.append(It)
#                     idx_normal.append(In)
#
#                 else:
#                     triangles, normals = self.get_trianges(X)
#                     for triangle in triangles:
#                         idx_vertex.append(I[triangle])
#
#         max_size = 1 if max_size is None else max_size
#
#         for c in [0,1,2]:
#             coord_vert[:, c] -= coord_vert[:,c].mean()
#             coord_vert[:, c]/= (coord_vert[:,c].max()/max_size)
#
#         if center is not None:
#             coord_vert+=center
#
#         idx_vertex = numpy.array(idx_vertex)+bias_vertex
#         idx_texture = numpy.array(idx_texture)+bias_texture
#         idx_normal = numpy.array(idx_normal)+bias_normal
#
#         self.export_mesh(filename_out, coord_vert, coord_texture=coord_texture,coord_norm=coord_norm,idx_vertex=idx_vertex,idx_texture=idx_texture,idx_normal=idx_normal,filename_material=filename_material)
#
#         return
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
    def export_mesh(self, filename_out, X, coord_texture, coord_norm,idx_vertex,idx_texture,idx_normal,filename_material=None,material_name=None,mode='w+'):

        fmt = '%1.3f'

        with open(filename_out, mode) as f_handle:
            f_handle.write("# Obj file\n")
            f_handle.write("o %s\n" % (material_name if material_name is not None else 'Object'))
            if filename_material is not None:f_handle.write('mtllib %s\n' % filename_material)
            if material_name     is not None:f_handle.write('usemtl %s\n' % material_name)
            for x in X: f_handle.write(f'v {fmt} {fmt} {fmt}\n' % (x[0], x[1], x[2]))

            if coord_texture is None:
                f_handle.write("vt 0 0\n")
            else:
                for t in coord_texture: f_handle.write("vt %1.2f %1.2f\n" % (t[0], t[1]))

            for n in coord_norm:f_handle.write("vn %1.2f %1.2f %1.2f\n" % (n[0], n[1], n[2]))

            for iv,it,inr in zip(idx_vertex,idx_texture,idx_normal):
                f_handle.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (iv[0] + 1, it[0]+1, inr[0] + 1, iv[1] + 1, it[1]+1, inr[1] + 1, iv[2] + 1, it[2]+1, inr[2] + 1))

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
                f.write('map_Kd %s\n'%filename_texture)

        return
# ----------------------------------------------------------------------------------------------------------------------
