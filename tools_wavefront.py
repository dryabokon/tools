import numpy
import pyvista
#----------------------------------------------------------------------------------------------------------------------
class ObjLoader:
    def __init__(self):
        self.coord_vert = []
        self.model = []
        self.scale = 1
        self.mat_color = (1,1,1)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def load_model(self, file,mat_color=(0.5,0.5,0.5),do_normalize=True):
        self.mat_color = mat_color

        coord_vert = []
        coord_texture = []
        coord_norm = []
        idx_vertex = []
        idx_texture = []
        idx_normal = []


        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            values = numpy.array(values)

            if values[0] == 'v':  coord_vert.append([float(v) for v in values[1:4]])
            if values[0] == 'vt': coord_texture.append([float(v) for v in values[1:3]])
            if values[0] == 'vn': coord_norm.append([float(v) for v in values[1:4]])

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

                idx_vertex.append(face_i)
                idx_texture.append(text_i)
                idx_normal.append(norm_i)

        if len(coord_texture) == 0: coord_texture.append([0,0])

        idx_vertex = [y for x in idx_vertex for y in x]
        idx_texture = [y for x in idx_texture for y in x]
        idx_normal = [y for x in idx_normal for y in x]

        coord_vert = numpy.array(coord_vert)
        if do_normalize:
            self.scale = coord_vert.max()
            coord_vert /= self.scale


        coord_norm = numpy.array(coord_norm)

        self.model = []
        for i in idx_vertex: self.model.extend(coord_vert[i])
        for i in idx_vertex: self.model.extend(mat_color)
        for i in idx_normal: self.model.extend(coord_norm[i])

        self.model = numpy.array(self.model, dtype='float32')
        self.n_vertex = len(idx_vertex)
        self.idx_vertex = idx_vertex
        self.idx_normal = idx_normal
        self.coord_vert = coord_vert
        self.coord_norm = coord_norm


        self.color_offset = self.n_vertex * 3*4
        self.normal_offset = (self.color_offset + self.n_vertex * 3*4)

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
    def convert(self, filename_in, filename_out):
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

        self.export_mesh(filename_out, coord_vert, idx_vertex)

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def export_mesh(self, filename_out, X, idx_vertex=None):
        f_handle = open(filename_out, "w+")
        f_handle.write("# Obj file\n")
        for x in X: f_handle.write("v %1.2f %1.2f %1.2f\n" % (x[0], x[1], x[2]))
        f_handle.write("vt 0 0\n")

        if idx_vertex is None:
            del_triangeles, normals = self.get_trianges(X)
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
                n = n / numpy.sqrt((n ** 2).sum())

            f_handle.write("vn %1.2f %1.2f %1.2f\n" % (n[0], n[1], n[2]))

        for n, t in enumerate(del_triangles):
            f_handle.write(
                "f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (t[0] + 1, 1, n + 1, t[1] + 1, 1, n + 1, t[2] + 1, 1, n + 1))

        f_handle.close()
        return
# ----------------------------------------------------------------------------------------------------------------------