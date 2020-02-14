#http://blog.db-in.com/cameras-on-opengl-es-2-x/
import numpy
import math
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GLUT import *
import glfw
import OpenGL.GL.shaders
import numpy
from scipy.spatial import Delaunay
import pyrr
import scipy
import tools_aruco
import pyvista
import tools_calibrate

numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=2)
# ----------------------------------------------------------------------------------------------------------------------
class ObjLoader:
    def __init__(self):
        self.coord_vert = []
        self.coord_texture = []
        self.coord_norm = []
        self.idx_vertex = []
        self.idx_texture = []
        self.idx_normal = []
        self.model = []
        self.scale = 1
# ----------------------------------------------------------------------------------------------------------------------
    def load_model(self, file):
        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v' :self.coord_vert.append([float(v) for v in values[1:4]])
            if values[0] == 'vt':self.coord_texture.append([float(v) for v in values[1:3]])
            if values[0] == 'vn':self.coord_norm.append([float(v) for v in values[1:4]])

            if values[0] == 'f':
                face_i = []
                text_i = []
                norm_i = []
                for v in values[1:4]:
                    w = v.split('/')
                    face_i.append(int(w[0])-1)
                    text_i.append(int(w[1])-1)
                    norm_i.append(int(w[2])-1)
                self.idx_vertex.append(face_i)
                self.idx_texture.append(text_i)
                self.idx_normal.append(norm_i)


        self.idx_vertex  = [y for x in self.idx_vertex for y in x]
        self.idx_texture = [y for x in self.idx_texture for y in x]
        self.idx_normal  = [y for x in self.idx_normal for y in x]

        self.coord_vert = numpy.array(self.coord_vert)
        self.scale = self.coord_vert.max()
        self.coord_vert/=self.scale

        for i in self.idx_vertex :self.model.extend(self.coord_vert[i])
        for i in self.idx_texture:self.model.extend(self.coord_texture[i])
        for i in self.idx_normal :self.model.extend(self.coord_norm[i])
        self.model = numpy.array(self.model, dtype='float32')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_model(self,X,filename_out):
        f_handle = open(filename_out, "w+")
        f_handle.write("# Obj file\n")
        for x in X:f_handle.write("v %1.2f %1.2f %1.2f\n"%(x[0],x[1],x[2]))
        f_handle.write("vt 0 0\n")
        Y = X[:,:2]

        cloud = pyvista.PolyData(X)
        surf = cloud.delaunay_2d()
        F = numpy.array(surf.faces)
        del_triangles = F.reshape((len(F)//4,4))
        del_triangles = del_triangles[:,1:]

        normals = numpy.array(surf.face_normals)

        for i,t in enumerate(del_triangles):
            A = X[t[1]] - X[t[0]]
            B = X[t[2]] - X[t[0]]
            Nx = A[1] * B[2] - A[2] * B[1]
            Ny = A[2] * B[0] - A[0] * B[2]
            Nz = A[0] * B[1] - A[1] * B[0]
            n = numpy.array((Nx,Ny,Nz),dtype=numpy.float)
            n = n/numpy.sqrt(n**2)
            n = normals[i]
            f_handle.write("vn %1.2f %1.2f %1.2f\n" % (n[0], n[1], n[2]))

        for n,t in enumerate(del_triangles):
            f_handle.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (t[0]+1,1,n+1,t[1]+1,1,n+1,t[2]+1,1,n+1))

        f_handle.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
class render_GL3D(object):

    def __init__(self,filename_obj,filename_texture=None, W=640, H=480,is_visible=False):

        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glfw.init()
        self.W = W
        self.H = H

        glfw.window_hint(glfw.VISIBLE, is_visible)
        self.window = glfw.create_window(self.W, self.H, "hidden window", None, None)
        glfw.make_context_current(self.window)
        self.mat_color = numpy.array([192,128,32])
        self.bg_color  = numpy.array([64, 64, 64,0])/255


        self.obj = ObjLoader()
        self.obj.load_model(filename_obj)
        self.filename_texture = filename_texture

        self.__init_shader()
        self.__init_texture(self.filename_texture)
        self.reset_view()

        self.__init_runtime()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_shader(self):
        frag_shader = """#version 330
                                    in vec2 newTexture;
                                    in vec3 fragNormal;
                                    out vec4 outColor;
                                    uniform sampler2D samplerTexture;
                                    void main()
                                    {
                                        vec3 ambientLightIntensity = vec3(0.4f, 0.4f, 0.4f);
                                        vec3 sunLightIntensity = vec3(0.9f, 0.9f, 0.9f);
                                        vec3 sunLightDirection = normalize(vec3(-2.0f, -2.0f, 0.0f));
                                        vec4 texel = texture(samplerTexture, newTexture);
                                        vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * max(dot(fragNormal, sunLightDirection), 0.0f);
                                        outColor = vec4(texel.rgb * lightIntensity, texel.a);
                                    }"""

        vert_shader = """#version 330
                                    in layout(location = 0) vec3 position;
                                    in layout(location = 1) vec2 textureCoords;
                                    in layout(location = 2) vec3 vertNormal;
                                    uniform mat4 transform,view,model,projection,light;
                                    out vec2 newTexture;
                                    out vec3 fragNormal;
                                    void main()
                                    {
                                        fragNormal = (light * vec4(vertNormal, 0.0f)).xyz;
                                        gl_Position = projection * view * model * transform * vec4(position, 1.0f);
                                        newTexture = textureCoords;
                                    }"""

        self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vert_shader, GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(frag_shader, GL_FRAGMENT_SHADER))
        glUseProgram(self.shader)
        return
# ----------------------------------------------------------------------------------------------------------------------
#http://duriansoftware.com/joe/An-intro-to-modern-OpenGL.-Chapter-4:-Rendering-a-Dynamic-3D-Scene-with-Phong-Shading.html
    def __init_texture(self,filename_texture=None):

        if filename_texture is not None:
            texture_image = cv2.imread(filename_texture)

        else:
            #texture_image = (numpy.random.rand(255, 255, 3)*255).astype(numpy.uint8)
            texture_image = numpy.zeros((255, 255, 3),dtype=numpy.uint8)
            texture_image[:,:,0]=self.mat_color[0]
            texture_image[:,:,1]=self.mat_color[1]
            texture_image[:,:,2]=self.mat_color[2]

        glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
        #glBindTexture(GL_TEXTURE_2D, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_image.shape[1], texture_image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, texture_image.flatten())
        glEnable(GL_TEXTURE_2D)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_projection(self):
        fx, fy = float(self.W), float( self.H)

        left, right, bottom, top = -0.5, (self.W - fx / 2) / fx, (fy / 2 - self.H) / fy, 0.5
        near, far = 1, 1000

        self.mat_camera = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])
        self.mat_projection = pyrr.matrix44.create_perspective_projection_from_bounds(left,right,bottom,top,near,far)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, self.mat_projection)
        return
# ------------------------------------------------------------d----------------------------------------------------------
    def __init_mat_view_RT(self, rvec,tvec):
        R = pyrr.matrix44.create_from_eulers(rvec)
        T = pyrr.matrix44.create_from_translation(tvec)
        self.mat_view = pyrr.matrix44.multiply(R,T)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        #self.mat_view = tools_aruco.compose_GL_MAT(numpy.array(rvec,dtype=numpy.float), numpy.array(tvec,dtype=numpy.float),do_flip=True)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_view_ETU(self, eye, target, up):
        self.mat_view = pyrr.matrix44.create_look_at(eye, target, up)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_model(self,rvec, tvec):
        self.mat_model = pyrr.matrix44.create_from_eulers(rvec)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.mat_model)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_transform(self,scale):
        self.mat_trns = pyrr.Matrix44.from_scale((scale,scale,scale,scale))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_light(self,r_vec):
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "light")    , 1, GL_FALSE, pyrr.matrix44.create_from_eulers(r_vec))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_runtime(self):
        texture_offset = len(self.obj.idx_vertex) * 12
        normal_offset = (texture_offset + len(self.obj.idx_texture) * 8)

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, self.obj.model.itemsize * len(self.obj.model), self.obj.model, GL_STATIC_DRAW)

        # positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.obj.model.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # textures
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.obj.model.itemsize * 2, ctypes.c_void_p(texture_offset))
        glEnableVertexAttribArray(1)

        # normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.obj.model.itemsize * 3, ctypes.c_void_p(normal_offset))
        glEnableVertexAttribArray(2)

        glClearColor(self.bg_color[0],self.bg_color[1],self.bg_color[2],self.bg_color[3])
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        #glShadeModel(GL_SMOOTH)
        glShadeModel(GL_FLAT)

        ## set material information
        ambient = [0.0215, 0.1745, 0.0215]
        diffuse = [0.07568, 0.61424, 0.07568]
        specular = [0.633, 0.727811, 0.633]
        shininess = 128 * 0.6

        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
        glMaterialfv(GL_FRONT, GL_SHININESS, shininess)

        return
# ----------------------------------------------------------------------------------------------------------------------
    #projection * view * model * transform
    # model maps from an object's local coordinate space into world space,
    # view from world space to camera space,
    # projection from camera to screen.
# ----------------------------------------------------------------------------------------------------------------------
    def __draw(self,rvec=(0,0,0),tvec=(0,0,0)):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.__init_mat_view_RT(rvec,tvec)
        glDrawArrays(GL_TRIANGLES, 0, len(self.obj.idx_vertex))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_GL(self,do_debug=False):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, len(self.obj.idx_vertex))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_image(self,rvec=None,tvec=None,do_debug=False):
        self.__draw(rvec,tvec)
        image_buffer = glReadPixels(0, 0, self.W,self.H, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(self.H, self.W, 3)
        image = cv2.flip(image,0)
        if do_debug:
            self.draw_mat(self.mat_projection, 20, 20,image)
            self.draw_mat(self.mat_view      , 20, 120, image)
            self.draw_mat(self.mat_model     , 20, 220, image)
            self.draw_mat(self.mat_camera    , 20, 350, image)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_mat(self, M, posx, posy, image):
        for row in range(M.shape[0]):
            if M.shape[1]==4:
                string1 = '%+1.2f %+1.2f %+1.2f %+1.2f' % (M[row, 0], M[row, 1], M[row, 2], M[row, 3])
            else:
                string1 = '%+1.2f %+1.2f %+1.2f' % (M[row, 0], M[row, 1], M[row, 2])
            image = cv2.putText(image, '{0}'.format(string1), (posx, posy + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(128, 128, 0), 1, cv2.LINE_AA)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def start_rotation(self):
        self.on_rotate = True
        self.mat_model_checkpoint = self.mat_model
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def stop_rotation(self):
        self.on_rotate = False
        self.mat_model_checkpoint = None
        return
# ----------------------------------------------------------------------------------------------------------------------
    def translate_view(self, delta_translate):
        eye, target, up = tools_calibrate.calculate_eye_target_up(self.mat_view)
        eye *= delta_translate
        self.__init_mat_view_ETU(eye, target, up)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def scale_model(self, scale_factor):
        self.mat_trns*=scale_factor
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def transform_model(self,mode):

        M = pyrr.matrix44.create_identity()
        if   mode == 'XY':M = pyrr.matrix44.create_from_z_rotation(-math.pi/2)
        elif mode == 'XZ':M = pyrr.matrix44.create_from_y_rotation(-math.pi/2)
        elif mode == 'YZ':M = pyrr.matrix44.create_from_x_rotation(-math.pi/2)
        elif mode == 'xy':M = pyrr.matrix44.create_from_z_rotation(+math.pi/2)
        elif mode == 'yz':M = pyrr.matrix44.create_from_y_rotation(+math.pi/2)
        elif mode == 'xz':M = pyrr.matrix44.create_from_x_rotation(+math.pi/2)
        self.mat_trns = M * self.mat_trns.max()
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        self.reset_view(reset_transform=False)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __align_light(self, euler_model):
        vec_light = self.vec_initial_light + euler_model - self.vec_initial_model
        self.__init_mat_light(vec_light)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __display_info(self):

        S, Q, tvec = pyrr.matrix44.decompose(self.mat_model)
        rvec = tools_calibrate.quaternion_to_euler(Q)

        print('rvec', rvec * 180 / math.pi)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def rotate_model(self, delta_angle):

        if self.on_rotate and self.mat_model_checkpoint is not None:
            S, Q, tvec = pyrr.matrix44.decompose(self.mat_model_checkpoint)
        else:
            S, Q, tvec = pyrr.matrix44.decompose(self.mat_model)

        rvec  = tools_calibrate.quaternion_to_euler(Q) + numpy.array(delta_angle)
        rvec[0] = numpy.clip(rvec[0],0.01,math.pi-0.01)
        self.__init_mat_model(rvec,tvec)
        self.__align_light(rvec)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def reset_view(self,reset_transform=True):
        obj_min = numpy.array(self.obj.coord_vert).astype(numpy.float).min()
        obj_max = numpy.array(self.obj.coord_vert).astype(numpy.float).max()
        self.vec_initial_light = (0,math.pi/2,-math.pi/2)
        self.vec_initial_model = (math.pi/2,math.pi/2,0)
        self.__init_mat_view_ETU(eye=(0, 0, -5 * (obj_max - obj_min)), target=(0, 0, 0), up=(0, -1, 0))
        self.__init_mat_model(self.vec_initial_model,(0,0,0))
        if reset_transform:
            self.__init_mat_transform(1)
        self.__init_mat_light(self.vec_initial_light)
        self.__init_mat_projection()
        self.stop_rotation()
        return
# ----------------------------------------------------------------------------------------------------------------------
