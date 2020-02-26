import numpy
import pyvista
#----------------------------------------------------------------------------------------------------------------------
class ObjLoader:
    def __init__(self):
        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_mesh(self, file, mat_color=(0.5, 0.5, 0.5), do_autoscale=True):
        self.mat_color = mat_color

        self.coord_vert = []
        self.coord_texture = []
        self.coord_norm = []
        self.idx_vertex = []
        self.idx_texture = []
        self.idx_normal = []


        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            values = numpy.array(values)

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

                self.idx_vertex.append(face_i)
                self.idx_texture.append(text_i)
                self.idx_normal.append(norm_i)

        if len(self.coord_texture) == 0: self.coord_texture.append([0,0])

        coord_vert = numpy.array(self.coord_vert)

        #if do_autoscale: #min max -> -1..1
            #min_value = coord_vert.min()
            #coord_vert-= min_value
            #max_value = coord_vert.max()
            #coord_vert/= max_value
            #coord_vert*=2
            #coord_vert-=1

        if do_autoscale:
            max_value = coord_vert.max()
            coord_vert/= max_value

        coord_norm = numpy.array(self.coord_norm)
        self.coord_vert = coord_vert
        self.coord_norm = coord_norm
        return
# ----------------------------------------------------------------------------------------------------------------------
    def remove_triangele(self):

        return
# ----------------------------------------------------------------------------------------------------------------------
    def scale_mesh(self,svec):
        for i in range(self.coord_vert.shape[0]):self.coord_vert[i]=numpy.multiply(self.coord_vert[i],svec)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def translate_mesh(self, tvec):
        for i in range(self.coord_vert.shape[0]):self.coord_vert[i]+= tvec
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
    def convert(self, filename_in, filename_out,do_normalize=True,cutoff=None):
        coord_vert = []
        idx_vertex = []
        lines = open(filename_in).readlines()

        for line in lines:
            values = line.split()
            if not values: continue
            values = numpy.array(values)
            if values[0] == 'v': coord_vert.append([float(v) for v in values[1:4]])

        coord_vert = numpy.array(coord_vert)

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

        self.export_mesh(filename_out, coord_vert, idx_vertex,do_transform=False,cutoff=cutoff)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_mesh(self, filename_out, X=None, idx_vertex=None,do_transform=False,cutoff=None):

        if do_transform:
            X[:,1]=0 - X[:,1]

        f_handle = open(filename_out, "w+")
        f_handle.write("# Obj file\n")
        for x in X: f_handle.write("v %1.2f %1.2f %1.2f\n" % (x[0], x[1], x[2]))
        f_handle.write("vt 0 0\n")

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
                n = -numpy.array((Nx, Ny, Nz), dtype=numpy.float)
                if numpy.sqrt((n ** 2).sum())>0:
                    n = n / numpy.sqrt((n ** 2).sum())

            f_handle.write("vn %1.2f %1.2f %1.2f\n" % (n[0], n[1], n[2]))

        for n, t in enumerate(del_triangles):
            if cutoff is not None:
                x = numpy.array([X[t[0]],X[t[1]],X[t[2]]])
                if x[:,2].mean()<cutoff:
                    continue


            f_handle.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (t[0] + 1, 1, n + 1, t[1] + 1, 1, n + 1, t[2] + 1, 1, n + 1))

        f_handle.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
