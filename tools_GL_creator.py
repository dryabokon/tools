import base64
import numpy
import tools_wavefront
from CV import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
class Mesh:
    def __init__(self):
        self.X = None
        self.coord_texture = None
        self.idx_vertex = None
        self.coord_norm = None
        self.idx_normal = None
        self.idx_texture = None
        self.material_name = None
        return
# ----------------------------------------------------------------------------------------------------------------------
class Material:
    def __init__(self):
        self.name = None
        self.color = None
        self.transparency = None
        return
# ----------------------------------------------------------------------------------------------------------------------
class OBJ_Utils:

    def __init__(self,folder_out):
        self.folder_out = folder_out
        self.ObjLoader = tools_wavefront.ObjLoader()
        self.bias_vertex, self.bias_texture, self.bias_normal = 0,0,0
        self.do_transpose = True

        return
# ----------------------------------------------------------------------------------------------------------------------
    def construct_material(self,name,color,transparency=0.0):
        mat = Material()
        mat.name = name
        mat.color = color
        mat.transparency = transparency
        return mat
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cube(self,dim, rvec, tvec,material):

        mesh = Mesh()
        mesh.material_name = material.name
        mesh.X = self.construct_cuboid_RT(dim, rvec, tvec)
        mesh.idx_vertex = self.construct_cuboid_index()
        mesh.coord_norm = self.construct_cuboid_normals(rvec, tvec)
        mesh.idx_normal = self.construct_cuboid_idx_normals()
        mesh.coord_texture = self.construct_cuboid_texture_coord()
        mesh.idx_texture = self.construct_cuboid_texture_index()

        return mesh
# ----------------------------------------------------------------------------------------------------------------------
    def construct_circle(self, dim,rvec, tvec, material):
        mesh = Mesh()
        mesh.material_name = material.name
        mesh.X = self.construct_circle_RT(dim, rvec, tvec)
        mesh.idx_vertex = self.construct_circle_index()
        mesh.coord_norm = self.construct_circle_normals(rvec, tvec)
        mesh.idx_normal = self.construct_circle_idx_normals()
        mesh.coord_texture = numpy.array([(0, 0)])
        mesh.idx_texture = numpy.array([(0, 0, 0) for i in range(len(mesh.idx_vertex))])

        return mesh
# ----------------------------------------------------------------------------------------------------------------------
    def construct_dodecahedron(self,dim, rvec, tvec,material):
        mesh = Mesh()
        mesh.material_name = material.name
        mesh.X = self.construct_dodecahedron_RT(dim, rvec, tvec)
        mesh.idx_vertex = self.generate_dodecahedron_index()
        mesh.coord_norm = self.generate_dodecahedron_normals(rvec, tvec)
        mesh.idx_normal = self.generate_dodecahedron_idx_normals()
        mesh.coord_texture = numpy.array([(0, 0)])
        mesh.idx_texture = numpy.array([(0, 0, 0) for i in range(len(mesh.idx_vertex))])

        return mesh
# ----------------------------------------------------------------------------------------------------------------------
    def compute_rotation_vector(self,direction_normalized):
        x_axis = numpy.array([1, 0, 0])
        rotation_axis = numpy.cross(x_axis, direction_normalized)
        angle = numpy.arccos(numpy.clip(numpy.dot(x_axis, direction_normalized), -1.0, 1.0))
        if numpy.linalg.norm(rotation_axis) < 1e-6:
            return numpy.array([0.0, 0.0, 0.0])
        rotation_axis_normalized = rotation_axis / numpy.linalg.norm(rotation_axis)
        rvec = rotation_axis_normalized * angle
        return rvec
# ----------------------------------------------------------------------------------------------------------------------
    def construct_line(self, p1, p2, material):
        dim = (numpy.linalg.norm(numpy.array(p2) - numpy.array(p1)), 0.05, 0.05)
        direction_normalized = (numpy.array(p2) - numpy.array(p1)) / numpy.linalg.norm(numpy.array(p2) - numpy.array(p1))
        rvec = self.compute_rotation_vector(direction_normalized)
        tvec = (numpy.array(p2) + numpy.array(p1)) / 2
        mesh = self.construct_cube(dim, rvec, tvec, material)
        return mesh
# ----------------------------------------------------------------------------------------------------------------------
    def transpose(self,mesh):
        mesh.X = mesh.X[:,[0, 2, 1]]*numpy.array([1, 1, -1]) if self.do_transpose else mesh.X*numpy.array([1, +1, 1])
        return mesh
# ----------------------------------------------------------------------------------------------------------------------
    def export_mesh(self,filename_obj,filename_material,mesh):
        mesh = self.transpose(mesh)

        self.ObjLoader.export_mesh(self.folder_out+filename_obj,
                                   mesh.X, coord_texture=mesh.coord_texture,
                                   coord_norm=mesh.coord_norm,idx_vertex=mesh.idx_vertex, idx_texture=mesh.idx_texture, idx_normal=mesh.idx_normal,
                                   filename_material=filename_material, material_name=mesh.material_name)

        self.bias_vertex, self.bias_texture, self.bias_normal = mesh.X.shape[0], mesh.coord_texture.shape[0], mesh.coord_norm.shape[0]
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_bias_data(self,filname_in):

        self.ObjLoader.load_mesh(filname_in, do_autoscale=False)
        bias_vertex = self.ObjLoader.coord_vert.shape[0]
        bias_texture = self.ObjLoader.coord_texture.shape[0]
        bias_normal = self.ObjLoader.coord_norm.shape[0]

        cx, cy, cz = self.ObjLoader.coord_vert.mean(axis=0).tolist()
        max_size = (self.ObjLoader.coord_vert.max(axis=0) - self.ObjLoader.coord_vert.min(axis=0)).max()
        return bias_vertex, bias_texture, bias_normal, cx, cy, cz, max_size
# ----------------------------------------------------------------------------------------------------------------------
    def append_mesh(self,filename_obj,filename_material,mesh):

        mesh = self.transpose(mesh)

        self.ObjLoader.export_mesh(self.folder_out+filename_obj,
                                   mesh.X, coord_texture=mesh.coord_texture,
                                   coord_norm=mesh.coord_norm,idx_vertex=mesh.idx_vertex+self.bias_vertex, idx_texture=mesh.idx_texture+self.bias_texture,idx_normal=mesh.idx_normal+self.bias_normal,
                                   filename_material=filename_material, material_name=mesh.material_name,mode='a+')
        self.bias_vertex+=mesh.X.shape[0]
        self.bias_texture+=mesh.coord_texture.shape[0]
        self.bias_normal+=mesh.coord_norm.shape[0]

        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_material(self,filename_material,material):
        self.ObjLoader.export_material(self.folder_out + filename_material, material.name, material.color, transparency=material.transparency,mode='w+')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def append_material(self,filename_material, material):
        self.ObjLoader.export_material(self.folder_out + filename_material, material.name, material.color, transparency=material.transparency,mode='a+')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def construct_circle_RT(self,dim, rvec, tvec):

        #N = 16*2
        N = 8*2
        X = numpy.zeros((N, 3), dtype=numpy.float32)
        for i in range(N):
            X[i, 0] = dim[0] * numpy.cos(2 * numpy.pi * i / N)
            X[i, 1] = dim[1] * numpy.sin(2 * numpy.pi * i / N)
            X[i, 2] = 0

        X = numpy.concatenate((numpy.array([[0, 0, 0]]),X), axis=0)

        RT = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=False, GL_style=False)
        Xt = tools_pr_geom.apply_matrix_GL(RT, X)[:, :3]
        return Xt
# ----------------------------------------------------------------------------------------------------------------------
    def construct_circle_index(self):
        # N = 16*2
        N = 8 * 2
        idx = numpy.zeros((N, 3), dtype=numpy.int32)
        for i in range(N):
            idx[i, 0] = 0
            idx[i, 1] = i + 1
            idx[i, 2] = i + 2 if i < N - 1 else 1

        return idx
# ----------------------------------------------------------------------------------------------------------------------
    def construct_circle_normals(self,rvec,tvec):
        # N = 16*2
        N = 8 * 2
        normals = numpy.zeros((N, 3), dtype=numpy.float32)
        for i in range(N):
            normals[i, 0] = 0
            normals[i, 1] = 0
            normals[i, 2] = -1

        RT = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=False, GL_style=False)
        normals = tools_pr_geom.apply_matrix_GL(RT, normals)[:, :3]
        return normals
# ----------------------------------------------------------------------------------------------------------------------
    def construct_circle_idx_normals(self):
        # N = 16*2
        N = 8 * 2
        idx = numpy.zeros((N, 3), dtype=numpy.int32)
        for i in range(N):
            idx[i, 0] = 1
            idx[i, 1] = 1
            idx[i, 2] = 1

        idx = -1+idx
        return idx
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid_RT(self,dim, rvec, tvec):
        d0, d1, d2 = dim[0], dim[1], dim[2]

        x_corners = [-d0 / 2, -d0 / 2, +d0 / 2, +d0 / 2, -d0 / 2, -d0 / 2, +d0 / 2, +d0 / 2]
        y_corners = [-d1 / 2, +d1 / 2, +d1 / 2, -d1 / 2, -d1 / 2, +d1 / 2, +d1 / 2, -d1 / 2]
        z_corners = [-d2 / 2, -d2 / 2, -d2 / 2, -d2 / 2, +d2 / 2, +d2 / 2, +d2 / 2, +d2 / 2]

        X = numpy.array([x_corners, y_corners, z_corners], dtype=numpy.float32).T

        RT = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=False, GL_style=False)
        Xt = tools_pr_geom.apply_matrix_GL(RT, X)[:, :3]

        return Xt
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid_index(self):
        idx = -1+numpy.array([[1, 2, 3], [3, 4, 1], [5, 7, 6], [7, 5, 8], [1, 6, 2], [5, 6, 1], [3, 7, 4], [7, 8, 4], [1, 4, 5],[8, 5, 4], [2, 6, 3], [7, 3, 6]])
        return idx
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid_texture_coord(self):
        coord_texture = numpy.array([[0.25,0.25],[0.50,0.25],[0.50,0.50],[0.25,0.50],[0.25,1.00],[0.50,1.00],[0.50,0.75],[0.25,0.75],[1.00,0.50],[0.75,0.75],[0.75,0.50],[1.00,0.75],[0.50,0.50],[0.50,0.75],[0.25,0.50],[0.25,0.75],[0.00,0.50],[0.25,0.50],[0.00,0.75],[0.25,0.75],[0.75,0.50],[0.75,0.75],[0.50,0.50],[0.50,0.75]])
        return coord_texture
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid_texture_index(self):
        idx=-1+numpy.array([[1,2,3],[3,4,1],[5,7,6],[7,5,8],[9,10,11],[12,10,9],[14,14,15],[14,16,15],[17,18,19],[20,19,18],[21,22,23],[24,23,22]])
        return idx
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid_normals(self,rvec,tvec):
        normals = numpy.array([[0,0,-1],[0,0,+1],[0,-1,0],[0,+1,0],[-1,0,0],[+1,0,0]],dtype=numpy.float32)
        RT = tools_pr_geom.compose_RT_mat(rvec, (0,0,0), do_rodriges=True, do_flip=False, GL_style=False)
        normals = tools_pr_geom.apply_matrix_GL(RT, normals)[:, :3]
        return normals
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid_idx_normals(self):
        idx = -1+numpy.array([[1,1,1],[1,1,1],[2,2,2],[2,2,2],[5,5,5],[5,5,5],[6,6,6],[6,6,6],[3,3,3],[3,3,3],[4,4,4],[4,4,4]],dtype=numpy.float32)
        return idx
# ----------------------------------------------------------------------------------------------------------------------
    def construct_dodecahedron_RT(self, dim=(1, 1, 1), rvec=(0, 0, 0), tvec=(0, 0, 0)):
        X =([[-1.113516, -0.262866, -0.809017],[-0.425325, 0.262866, -1.309017],[-0.688191, -1.113516, -0.500000],[-0.850651, 1.113516, 0.000000],[0.850651, -1.113516, 0.000000],[1.113516, 0.262866, 0.809017],[0.262866, -1.113516, 0.809017],[1.113516, 0.262866, -0.809017],[0.262866, -1.113516, -0.809017],[0.425325, -0.262866, -1.309017],[-0.688191, -1.113516, 0.500000],[-1.376382, 0.262866, 0.000000],[0.425325, -0.262866, 1.309017],[-1.113516, -0.262866, 0.809017],[1.376382, -0.262866, 0.000000],[0.688191, 1.113516, 0.500000],[0.688191, 1.113516, -0.500000],[-0.425325, 0.262866, 1.309017],[-0.262866, 1.113516, -0.809017],[-0.262866, 1.113516, 0.809017]])
        X = numpy.array(X, dtype=numpy.float32)
        d0, d1, d2 = dim[0], dim[1], dim[2]
        X = X * numpy.array([d0, d1, d2], dtype=numpy.float32)
        RT = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=False, GL_style=False)
        X = tools_pr_geom.apply_matrix_GL(RT, X)[:, :3]
        return X
# ----------------------------------------------------------------------------------------------------------------------
    def generate_dodecahedron_index(self):
        faces = [[1, 2, 3],[2, 1, 4],[5, 6, 7],[8, 9, 10],[10, 9, 3],[3, 7, 11],[3, 12, 1],[1, 12, 4],[13, 7, 6],[7, 13, 11],[5, 9, 8],[9, 5, 7],[14, 12, 11],[12, 14, 4],[6, 15, 8],[15, 6, 5],[8, 15, 5],[16, 8, 17],[16, 17, 4],[9, 7, 3],[4, 14, 18],[14, 11, 18],[8, 10, 19],[6, 8, 16],[6, 16, 20],[16, 4, 20],[10, 2, 19],[2, 4, 19],[11, 13, 18],[18, 13, 20],[4, 18, 20],[4, 17, 19],[17, 8, 19],[3, 2, 10],[13, 6, 20],[11, 12, 3]]
        return -1+numpy.array(faces, dtype=numpy.int32)
# ----------------------------------------------------------------------------------------------------------------------
    def generate_dodecahedron_normals(self,rvec,tvec):
        normals = [[ -0.2764, -0.4472, -0.8507],[ -0.7236, 0.4472, -0.5257],[ 0.7236, -0.4472, 0.5257],[ 0.7236, -0.4472, -0.5257],[ -0.0000, -1.0000, -0.0000],[ -0.8944, -0.4472, -0.0000],[ -0.2764, -0.4472, 0.8507],[ -0.7236, 0.4472, 0.5257],[ 0.8944, 0.4472, -0.0000],[ -0.0000, 1.0000, -0.0000],[ 0.2764, 0.4472, -0.8507],[ 0.2764, 0.4472, 0.8507]]
        normals = numpy.array(normals)
        RT = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=False, GL_style=False)
        normals = tools_pr_geom.apply_matrix_GL(RT, normals)[:, :3]
        return normals
# ----------------------------------------------------------------------------------------------------------------------
    def generate_dodecahedron_idx_normals(self):
        idx_normals = [[1, 1, 1],[2, 2, 2],[3, 3, 3],[4, 4, 4],[1, 1, 1],[5, 5, 5],[6, 6, 6],[2, 2, 2],[3, 3, 3],[7, 7, 7],[4, 4, 4],[5, 5, 5],[6, 6, 6],[8, 8, 8],[9, 9, 9],[3, 3, 3],[4, 4, 4],[9, 9, 9],[10, 10, 10],[5, 5, 5],[8, 8, 8],[7, 7, 7],[11, 11, 11],[9, 9, 9],[12, 12, 12],[10, 10, 10],[11, 11, 11],[2, 2, 2],[7, 7, 7],[12, 12, 12],[8, 8, 8],[10, 10, 10],[11, 11, 11],[1, 1, 1],[12, 12, 12],[6, 6, 6]]
        idx_normals = -1 + numpy.array(idx_normals, dtype=numpy.int32)
        return idx_normals
# ----------------------------------------------------------------------------------------------------------------------
    def test_00(self,filename_obj='All.obj', filename_mat='mat.mtl'):

        mat_plane = self.construct_material('plane', (64, 64, 64))
        mat_line = self.construct_material('line', (0, 100, 192))
        mat_cube = self.construct_material('cube', (200, 0, 0), 0.1)
        mat_dode = self.construct_material('dode', (100, 200, 0), 0.2)

        self.export_material(filename_mat, mat_plane)
        self.append_material(filename_mat, mat_dode)
        self.append_material(filename_mat, mat_line)
        self.append_material(filename_mat, mat_cube)

        mesh_circle = self.construct_circle((100, 100, 1), (0, 0, 0), (10, 20, -10), mat_plane)
        mesh_cube  = self.construct_cube((1, 1, 1), (0, 0, 0), (10, 20, -10), mat_cube)
        mesh_dode  = self.construct_dodecahedron((1, 1, 1), (0, 0, 0), (0, 0, 0), mat_dode)
        mesh_line  = self.construct_line((0, 0, 0), (10, 20, -10), mat_line)

        self.export_mesh(filename_obj, filename_mat, mesh_circle)
        self.append_mesh(filename_obj, filename_mat, mesh_cube)
        self.append_mesh(filename_obj, filename_mat, mesh_dode)
        self.append_mesh(filename_obj, filename_mat, mesh_line)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def test_01_plane(self):
        filename_obj = 'plane.obj'
        filename_mat = 'plane.mtl'
        mat_plane = self.construct_material('plane', (64, 64, 64))
        mesh_plane = self.construct_circle((100, 100, 1), (0, 0, 0), (0, 0, -0), mat_plane)
        self.export_mesh(filename_obj, filename_mat, mesh_plane)
        self.export_material(filename_mat, mat_plane)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def test_04_cube(self,with_plane=False):
        filename_obj = 'cube.obj'
        filename_mat = 'cube.mtl'
        mat = self.construct_material('cube', (200, 0, 0), 0.1)
        mesh = self.construct_cube((1, 1, 1), (0, 0, 0), (20, 10, 0), mat)
        self.export_mesh(filename_obj, filename_mat, mesh)
        self.export_material(filename_mat, mat)

        if with_plane:
            mat_plane = self.construct_material('plane', (64, 64, 64))
            mesh_plane = self.construct_circle((100, 100, 1), (0, 0, 0), (0, 0, -0), mat_plane)
            self.append_mesh(filename_obj, filename_mat, mesh_plane)
            self.append_material(filename_mat, mat_plane)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def test_05_dodecahedron(self,with_plane=False):

        filename_obj = 'dodecahedron.obj'
        filename_mat = 'dodecahedron.mtl'

        mat = self.construct_material('dodecahedron', (200, 0, 0), 0.1)
        mesh = self.construct_dodecahedron((1, 1, 1), (0, 0, 0), (10, 20, 10), mat)
        self.export_mesh(filename_obj, filename_mat, mesh)
        self.export_material(filename_mat, mat)

        if with_plane:
            mat_plane = self.construct_material('plane', (64, 64, 64))
            mesh_plane = self.construct_circle((100, 100, 1), (0, 0, 0), (0, 0, -0), mat_plane)
            self.append_mesh(filename_obj, filename_mat, mesh_plane)
            self.append_material(filename_mat, mat_plane)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_base64Obj(self,filename_obj):
        with open(filename_obj, 'rb') as f:
            base64Obj = f.read()
        base64Obj = base64.b64encode(base64Obj).decode('utf-8')
        base64Obj = '"' + base64Obj + '"'
        return base64Obj
    # ----------------------------------------------------------------------------------------------------------------------
    def replace_last_substring(self,text, substring, replacement):
        idx = text.rfind(substring)
        return text[:idx] + replacement + text[idx + len(substring):]
    # ----------------------------------------------------------------------------------------------------------------------
    def obj_2_html(self,filename_obj, filename_mat, filename_img):
        with open('./images/ex_GL/renderer_self_contained.html') as f:
            txt_html = f.read()

        with open(filename_obj) as f:
            placeholder_objText = f.read()

        with open(filename_mat) as f:
            placeholder_mtlText = f.read()

        placeholder_base64Texture = f'data:image/png;base64,{self.get_base64Obj(filename_img)[1:-1]}'

        txt_html = self.replace_last_substring(txt_html, 'placeholder_objText', "`" + placeholder_objText + "`")
        txt_html = self.replace_last_substring(txt_html, 'placeholder_mtlText', "`" + placeholder_mtlText + "`")
        txt_html = self.replace_last_substring(txt_html, 'placeholder_base64Texture',"`" + placeholder_base64Texture + "`")

        return txt_html
    # ----------------------------------------------------------------------------------------------------------------------