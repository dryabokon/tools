import os
import numpy
import pyvista
import pyrr
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
    def convert_v0(self, filename_in, filename_out,do_normalize=True,cutoff=None):
        coord_vert = []
        idx_vertex = []
        coord_texture = []
        lines = open(filename_in).readlines()

        for line in lines:
            values = line.split()
            if not values: continue
            values = numpy.array(values)
            if values[0] == 'v': coord_vert.append([float(v) for v in values[1:4]])
            if values[0] == 'vt': coord_texture.append([float(v) for v in values[1:4]])

        coord_vert = numpy.array(coord_vert)
        coord_texture = numpy.array(coord_texture)

        for line in lines:
            values = line.split()
            if not values: continue
            values = numpy.array(values)
            if values[0] == 'f':
                X = []
                I = []
                for face in values[1:]:
                    idx = int(face.split('/')[0]) - 1
                    X.append(coord_vert[idx])
                    I.append(idx)
                X = numpy.array(X)
                I = numpy.array(I)
                if len(I) == 3:
                    idx_vertex.append(I)

                else:
                    triangles, normals = self.get_trianges(X)
                    for triangle in triangles:
                        idx_vertex.append(I[triangle])

        if do_normalize:
            coord_vert[:, 0] -= coord_vert[:,0].mean()
            coord_vert[:, 1] -= coord_vert[:,1].mean()
            coord_vert[:, 2] -= coord_vert[:,2].mean()
            coord_vert[:, 0]/= coord_vert[:, 0].max()
            coord_vert[:, 1]/= coord_vert[:, 1].max()
            coord_vert[:, 2]/= coord_vert[:, 2].max()

        self.export_mesh(filename_out, coord_vert, coord_texture=coord_texture,idx_vertex=idx_vertex,do_transform=False,cutoff=cutoff)

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
    def export_mesh(self, filename_out, X, coord_texture=None, idx_vertex=None, do_transform=False, cutoff=None, replace_zero_normals=False,filename_material=None):

        if do_transform:
            X[:,1]=0 - X[:,1]

        f_handle = open(filename_out, "w+")
        f_handle.write("# Obj file\n")
        f_handle.write("o Object\n")
        for x in X: f_handle.write("v %1.2f %1.2f %1.2f\n" % (x[0], x[1], x[2]))

        if coord_texture is None:
            f_handle.write("vt 0 0\n")
        else:
            for t in coord_texture: f_handle.write("vt %1.2f %1.2f\n" % (t[0], t[1]))

        if idx_vertex is None:
            del_triangles, normals = self.get_trianges(X)
        else:
            del_triangles = idx_vertex

        for i, t in enumerate(del_triangles):
            if idx_vertex is None:
                n = normals[i]
            else:
                A = X[t[1]] - X[t[0]]
                B = X[t[2]] - X[t[0]]
                Nx = A[1] * B[2] - A[2] * B[1]
                Ny = A[2] * B[0] - A[0] * B[2]
                Nz = A[0] * B[1] - A[1] * B[0]
                n = -numpy.array((Nx, Ny, Nz), dtype=numpy.float32)
                if numpy.sqrt((n ** 2).sum())>0:
                    n = n / numpy.sqrt((n ** 2).sum())

            if replace_zero_normals and (n ** 2).sum()==0:
                n=(1,1,1)
            f_handle.write("vn %1.2f %1.2f %1.2f\n" % (n[0], n[1], n[2]))

        for n, t in enumerate(del_triangles):
            if cutoff is not None:
                x = numpy.array([X[t[0]],X[t[1]],X[t[2]]])
                if x[:,2].mean()<cutoff:
                    continue

            tx=(1,1,1)
            if coord_texture is not None:tx=(t[0]+1,t[1]+1,t[2]+1)

            f_handle.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (t[0]+1, tx[0], n + 1, t[1] + 1, tx[1], n + 1, t[2] + 1, tx[2], n + 1))

        if filename_material is not None:
            f_handle.write('mtllib %s\n' % filename_material)

        f_handle.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_material(self,filename_out,color255,filename_texture=None):

        # with open(filename_out, "w+") as f:
        #     f.write('Ka %1.4f %1.4f %1.4f\n' % (color255[0]/255.0, color255[1]/255.0, color255[2]/255.0))
        #     if filename_texture is not None:
        #         f.write('map_Kd %s\n'%filename_texture)

        with open(filename_out, "w+") as f:
            f.write('# Material\n')
            f.write('newmtl Material\n')
            f.write('Ns 96.078431\n')
            f.write('Ka %1.4f %1.4f %1.4f\n' % (color255[0] / 255.0, color255[1] / 255.0, color255[2] / 255.0))
            f.write('Kd 0.640000 0.640000 0.640000\n')
            f.write('Ks 0.500000 0.500000 0.500000\n')
            f.write('Ke 0.600000 0.000000 0.000000\n')
            f.write('Ni 1.00000\n')
            f.write('d 1.00000\n')
            f.write('illum 20\n')


        return
# ----------------------------------------------------------------------------------------------------------------------