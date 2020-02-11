import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GLUT import *
from PIL import Image,ImageDraw
import glfw
import OpenGL.GL.shaders
import numpy
from pywavefront import Wavefront
from pywavefront import visualization
import pyrr
import scipy
import tools_aruco
# ----------------------------------------------------------------------------------------------------------------------
class ObjLoader:
    def __init__(self):
        self.vert_coords = []
        self.text_coords = []
        self.norm_coords = []
        self.vertex_index = []
        self.texture_index = []
        self.normal_index = []
        self.model = []
# ----------------------------------------------------------------------------------------------------------------------
    def load_model(self, file):
        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':self.vert_coords.append(values[1:4])
            if values[0] == 'vt':self.text_coords.append(values[1:3])
            if values[0] == 'vn':self.norm_coords.append(values[1:4])

            if values[0] == 'f':
                face_i = []
                text_i = []
                norm_i = []
                for v in values[1:4]:
                    w = v.split('/')
                    face_i.append(int(w[0])-1)
                    text_i.append(int(w[1])-1)
                    norm_i.append(int(w[2])-1)
                self.vertex_index.append(face_i)
                self.texture_index.append(text_i)
                self.normal_index.append(norm_i)

        self.vertex_index = [y for x in self.vertex_index for y in x]
        self.texture_index = [y for x in self.texture_index for y in x]
        self.normal_index = [y for x in self.normal_index for y in x]

        for i in self.vertex_index:self.model.extend(self.vert_coords[i])
        for i in self.texture_index:self.model.extend(self.text_coords[i])
        for i in self.normal_index:self.model.extend(self.norm_coords[i])
        self.model = numpy.array(self.model, dtype='float32')
        return
# ----------------------------------------------------------------------------------------------------------------------
class render_GL3D(object):

    def __init__(self,filename_obj,filename_texture=None, W=640, H=480):

        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glfw.init()
        self.W = W
        self.H = H
        self.bg_color = (0.3, 0.3, 0.3, 0.5)
        glfw.window_hint(glfw.VISIBLE, False)
        self.window = glfw.create_window(self.W, self.H, "hidden window", None, None)
        glfw.make_context_current(self.window)

        self.obj = ObjLoader()
        self.obj.load_model(filename_obj)
        self.filename_texture = filename_texture

        self.__init_shader()
        self.__init_texture(self.filename_texture)
        self.init_mat_projection()
        self.__init_mat_view((0,0*0.6,0),(0,0,-1),flip=False)
        self.__init_mat_model()
        self.__init_mat_transform((0.1,0.1,0.1))
        self.__init_mat_light()
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
    def __init_texture(self,filename_texture=None):

        if filename_texture is not None:
            texture_image = cv2.imread(filename_texture)

        else:
            #texture_image = (numpy.random.rand(255, 255, 3)*255).astype(numpy.uint8)
            texture_image = numpy.full((255, 255, 3),192,dtype=numpy.uint8)
            texture_image[:,:,1]=128
            texture_image[:,:,2]=0

        #glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_image.shape[1], texture_image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, texture_image.flatten())
        glEnable(GL_TEXTURE_2D)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_mat_projection0(self):
        near,far  = 1,1000
        fx, fy = float(self.W), float(self.H)
        principalX, principalY = fx / 2, fy / 2
        left = -principalX / fx
        right = (self.W - principalX) / fx
        bottom = (principalY - self.H) / fy
        top = principalY / fy
        self.cameraMatrix = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])

        mat_projection2 = self.convert_hz_intrinsic_to_opengl_projection(self.cameraMatrix, fx / 2, fy / 2, self.W,self.H, near, far)
        self.mat_projection = pyrr.matrix44.create_perspective_projection_from_bounds(left,right,bottom,top,near,far)

        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, self.mat_projection)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_mat_projection(self):
        near, far = 1, 1000
        fx, fy = float(1*self.W), float(1*self.H)
        self.cameraMatrix = numpy.array([[fx, 0, fx/2], [0, fy, fy/2], [0, 0, 1]])
        self.mat_projection = self.convert_hz_intrinsic_to_opengl_projection(self.cameraMatrix, fx/2, fy/2, self.W, self.H, near, far)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, self.mat_projection)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_view(self,rvec=(0,0,0), tvec=(0,0,0),flip=True):
        self.mat_view = tools_aruco.compose_GL_MAT(numpy.array(rvec,dtype=numpy.float), numpy.array(tvec,dtype=numpy.float),flip=flip)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_model(self,rvec=(0,0,0), tvec=(0,0,0),flip=True):
        self.mat_model  = tools_aruco.compose_GL_MAT(numpy.array(rvec, dtype=numpy.float),numpy.array(tvec, dtype=numpy.float), flip=flip)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.mat_model)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_transform(self,scale=(1,1,1.0)):
        self.mat_trns = pyrr.Matrix44.from_scale((scale[0],scale[1],scale[2]))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_light(self):
        mat_light = pyrr.Matrix44.from_y_rotation(0.6)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "light")    , 1, GL_FALSE, mat_light)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_runtime(self):
        texture_offset = len(self.obj.vertex_index) * 12
        normal_offset = (texture_offset + len(self.obj.texture_index) * 8)

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
        return
# ----------------------------------------------------------------------------------------------------------------------
    #projection * view * model * transform
    # model maps from an object's local coordinate space into world space,
    # view from world space to camera space,
    # projection from camera to screen.
# ----------------------------------------------------------------------------------------------------------------------
    def __draw(self,rvec=(0,0,0),tvec=(0,0,0)):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.__init_mat_view(rvec,tvec)
        #self.__init_mat_model(rvec,tvec)
        glDrawArrays(GL_TRIANGLES, 0, len(self.obj.vertex_index))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_image(self,rvec=None,tvec=None):
        self.__draw(rvec,tvec)
        image_buffer = glReadPixels(0, 0, self.W,self.H, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(self.H, self.W, 3)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def convert_hz_intrinsic_to_opengl_projection(self,K,x0,y0,width,height,znear,zfar, window_coords='y down'):
        #https://gist.github.com/astraw/1341472#cameramatrix.txt
        znear = float(znear)
        zfar = float(zfar)
        depth = zfar - znear
        q = -(zfar + znear) / depth
        qn = -2 * (zfar * znear) / depth

        if window_coords=='y up':
            proj = numpy.array([[ 2*K[0,0]/width, -2*K[0,1]/width ,(-2*K[0,2]+width+2*x0)/width  , 0 ],
                                [ 0             , -2*K[1,1]/height,(-2*K[1,2]+height+2*y0)/height, 0 ],
                                [ 0             , 0               , q                            , qn],
                                [ 0             , 0               ,-1                            , 0 ]])
        else:
            assert window_coords=='y down'
            proj = numpy.array([[ 2*K[0,0]/width, -2*K[0,1]/width, (-2*K[0,2]+width+2*x0)/width, 0 ],
                             [  0,              2*K[1,1]/height,( 2*K[1,2]-height+2*y0)/height, 0],
                             [0,0,q,qn],  # This row is standard glPerspective and sets near and far planes.
                             [0,0,-1,0]]) # This row is also standard glPerspective.
        return proj
# ----------------------------------------------------------------------------------------------------------------------
    def RQ(self,M):

        R, K = scipy.linalg.rq(M)
        n = R.shape[0]
        for i in range(n):
            if R[i, i] < 0:
                R[:, i] = -R[:, i]
                K[i, :] = -K[i, :]
        return R, K
# ----------------------------------------------------------------------------------------------------------------------
    def pmat2cam_center(self,P):

        assert P.shape == (3, 4)
        determinant = numpy.linalg.det

        # camera center
        X = determinant([P[:, 1], P[:, 2], P[:, 3]])
        Y = -determinant([P[:, 0], P[:, 2], P[:, 3]])
        Z = determinant([P[:, 0], P[:, 1], P[:, 3]])
        T = -determinant([P[:, 0], P[:, 1], P[:, 2]])

        C_ = numpy.transpose(numpy.array([[X / T, Y / T, Z / T]]))
        return C_
# ----------------------------------------------------------------------------------------------------------------------
    def decompose(self,pmat):
        M = pmat[:,:3]
        K,R = self.RQ(M)
        K = K/K[2,2] # normalize intrinsic parameter matrix
        C_ = self.pmat2cam_center(pmat)
        t = numpy.dot( -R, C_)
        Rt = numpy.hstack((R, t ))

        return dict(intrinsic=K,rotation=R,cam_center=C_,t=t,extrinsic=Rt)
# ----------------------------------------------------------------------------------------------------------------------