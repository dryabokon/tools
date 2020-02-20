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
import pyrr
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_IO
import tools_render_CV
import tools_wavefront
import tools_calibrate
# ----------------------------------------------------------------------------------------------------------------------
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=2)
# ----------------------------------------------------------------------------------------------------------------------
class VBO(object):
    def __init__(self):
        self.n_vertex = 360000
        self.iterator = 0
        self.color_offset  = 3*self.n_vertex
        self.normal_offset = self.color_offset + 3*self.n_vertex
        self.normal_offset = self.color_offset + 3*self.n_vertex
        self.data = numpy.zeros(9*self.n_vertex,dtype='float32')
        self.id =   numpy.full(9*self.n_vertex,-1)
        self.cnt = 0
        self.marker_positions = []
        return
# ----------------------------------------------------------------------------------------------------------------------
    def append_object(self,filename_obj,mat_color,do_normalize_model_file,svec=(1,1,1),tvec=(0,0,0)):

        self.marker_positions.append(numpy.array(tvec))

        object = tools_wavefront.ObjLoader()
        object.load_mesh(filename_obj, mat_color, do_normalize_model_file)
        object.scale_mesh(svec)
        object.translate_mesh(tvec)

        clr = numpy.array([object.mat_color,object.mat_color,object.mat_color]).flatten()

        for idxv,idxn in zip(object.idx_vertex,object.idx_normal):
            self.data[self.iterator:                  self.iterator                  +9] = (object.coord_vert[idxv]).flatten()
            self.data[self.iterator+self.color_offset:self.iterator+self.color_offset+9] = clr
            self.data[self.iterator+self.normal_offset:self.iterator+self.normal_offset+9] = (object.coord_norm[idxn]).flatten()
            self.id[self.iterator:                  self.iterator + 9] = self.cnt
            self.iterator+=9

        self.cnt+=1

        return object
# ----------------------------------------------------------------------------------------------------------------------
    def remove_last_object(self):
        if self.cnt>1:
            self.remove_object(self.cnt-1)
            self.marker_positions.pop()
            return 0
        return 1
# ----------------------------------------------------------------------------------------------------------------------
    def remove_object(self,id):
        if id>0:
            idx = numpy.where(self.id==id)
            self.id[idx] = -1
            self.data[idx]=0
            self.data[[i + self.color_offset for i in idx]] = 0
            self.data[[i + self.normal_offset for i in idx]] = 0
        return
# ----------------------------------------------------------------------------------------------------------------------
class render_GL3D(object):

    def __init__(self,filename_obj,W=640, H=480,is_visible=True,do_normalize_model_file=True,scale=(1,1,1)):

        glfw.init()
        self.marker_scale = 0.015
        #self.marker_scale = 1
        self.scale = scale
        self.do_normalize_model_file = do_normalize_model_file
        self.bg_color  = numpy.array([76, 76, 76,1])/255

        self.W,self.H  = W,H
        glfw.window_hint(glfw.VISIBLE, is_visible)
        self.window = glfw.create_window(self.W, self.H, "GL viewer", None, None)
        glfw.make_context_current(self.window)

        self.__init_shader()

        self.my_VBO = VBO()
        self.object = self.my_VBO.append_object(filename_obj, numpy.array([188, 136, 113])/255.0,self.do_normalize_model_file)
        self.bind_VBO()
        self.reset_view()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_shader(self):
        # projection * view * model * transform
        # model maps from an object's local coordinate space into world space,
        # view from world space to camera space,
        # projection from camera to screen.
        vert_shader = """#version 330
                                    in layout(location = 0) vec3 position;
                                    in layout(location = 1) vec3 color;
                                    in layout(location = 2) vec3 vertNormal;
                                    uniform mat4 transform,view,model,projection,light;
                                    out vec3 inColor;
                                    out vec3 fragNormal;
                                    void main()
                                    {
                                        fragNormal = -abs((light * view * model * transform * vec4(vertNormal, 0.0f)).xyz);
                                        gl_Position = projection * view * model * transform * vec4(position, 1.0f);
                                        inColor = color;
                                    }"""

        frag_shader = """#version 330
                                    in vec3 inColor;
                                    in vec3 fragNormal;
                                    out vec4 outColor;
                                    void main()
                                    {
                                        
                                        vec3 ambientLightIntensity  = vec3(0.1f, 0.1f, 0.1f);
                                        vec3 sunLightIntensity      = vec3(1.0f, 1.0f, 1.0f);
                                        vec3 sunLightDirection      = normalize(vec3(+0.0f, -1.0f, +0.0f));

                                        



                                        vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * dot(fragNormal, sunLightDirection);
                                        outColor = vec4(inColor*lightIntensity, 1);
                                    }"""



        self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vert_shader, GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(frag_shader, GL_FRAGMENT_SHADER))
        glUseProgram(self.shader)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def bind_VBO(self):

        glBindBuffer(GL_ARRAY_BUFFER, glGenBuffers(1))
        glBufferData(GL_ARRAY_BUFFER, self.my_VBO.data.itemsize * len(self.my_VBO.data), self.my_VBO.data, GL_STATIC_DRAW)

        # positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.my_VBO.data.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.my_VBO.data.itemsize * 3, ctypes.c_void_p(self.my_VBO.data.itemsize*self.my_VBO.color_offset))
        glEnableVertexAttribArray(1)

        # normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.my_VBO.data.itemsize * 3, ctypes.c_void_p(self.my_VBO.data.itemsize*self.my_VBO.normal_offset))
        glEnableVertexAttribArray(2)

        glClearColor(self.bg_color[0], self.bg_color[1], self.bg_color[2], self.bg_color[3])
        glEnable(GL_DEPTH_TEST)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_projection(self):

        fx, fy = float(self.W), float(self.H)
        left, right, bottom, top = -0.5, (self.W - fx / 2) / fx, (fy / 2 - self.H) / fy, 0.5
        near, far = 1, 1000

        self.mat_camera = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])
        self.mat_projection = pyrr.matrix44.create_perspective_projection_from_bounds(left,right,bottom,top,near,far)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, self.mat_projection)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_view_RT(self, rvec,tvec,flip=True):
        self.mat_view = tools_render_CV.compose_GL_MAT(numpy.array(rvec, dtype=numpy.float),numpy.array(tvec, dtype=numpy.float),flip)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
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
    def __init_mat_transform(self,scale_vec):

        self.mat_trns = pyrr.Matrix44.from_scale(scale_vec)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_light(self,r_vec):
        self.mat_light = pyrr.matrix44.create_from_eulers(r_vec)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "light")    , 1, GL_FALSE,self.mat_light )
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_modelview(self,rvec,tvec):
        M = tools_render_CV.compose_GL_MAT(numpy.array(rvec, dtype=numpy.float), numpy.array(tvec, dtype=numpy.float),do_flip=True)
        S, Q, tvec_view = pyrr.matrix44.decompose(M)
        rvec_model = tools_calibrate.quaternion_to_euler(Q)
        self.__init_mat_model(rvec_model, (0, 0, 0))
        self.mat_view = pyrr.matrix44.create_from_translation(tvec_view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __init_modelview_ubstable(self, rvec, tvec):
        M0 = tools_render_CV.compose_GL_MAT(numpy.array(rvec, dtype=numpy.float),numpy.array(tvec, dtype=numpy.float), do_flip=True)
        M1 = tools_render_CV.compose_GL_MAT(numpy.array(rvec, dtype=numpy.float), numpy.array(tvec, dtype=numpy.float),do_flip=False)

        R = pyrr.matrix44.create_from_eulers(rvec)
        T = pyrr.matrix44.create_from_translation(tvec)
        M = pyrr.matrix44.multiply(R,T)

        S, Q, tvec_view = pyrr.matrix44.decompose(M)
        rvec_model = tools_calibrate.quaternion_to_euler(Q)
        self.__init_mat_model(rvec_model, (0, 0, 0))

        #self.mat_view = pyrr.matrix44.create_from_translation(tvec_view)
        eye = tvec_view
        target  = (0,0,0)
        up = (0,1,0)
        self.__init_mat_view_ETU(eye, target, up)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw(self,rvec=None,tvec=None):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        if rvec is not None and tvec is not None:
            self.init_modelview(rvec,tvec)
        glDrawArrays(GL_TRIANGLES, 0, self.my_VBO.iterator//3)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_image(self, rvec=None, tvec=None, do_debug=False):
        self.draw(rvec, tvec)
        image_buffer = glReadPixels(0, 0, self.W, self.H, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(self.H, self.W, 3)
        image = image[:,:,[2,1,0]]
        image = cv2.flip(image,0)

        if do_debug:
            S, Q, tvec_model = pyrr.matrix44.decompose(self.mat_model)
            rvec_model = tools_calibrate.quaternion_to_euler(Q)

            S, Q, tvec_view = pyrr.matrix44.decompose(self.mat_view)
            rvec_view = tools_calibrate.quaternion_to_euler(Q)


            self.draw_mat(self.mat_trns,       20, 20, image)
            self.draw_mat(self.mat_model,      20, 120, image)
            self.draw_mat(self.mat_view,       20, 220, image)
            self.draw_mat(self.mat_projection, 20, 320, image)
            self.draw_mat(self.mat_camera,     20, 420, image)
            self.draw_vec(rvec_model         , 20, 520, image)
            self.draw_vec(rvec_view,           20, 560, image)
            self.draw_vec(tvec_view,           20, 580, image)

            image = tools_render_CV.draw_points_numpy_MVP(self.object.coord_vert, image, self.mat_projection, self.mat_view, self.mat_model, self.mat_trns)

        if do_debug:
            M = pyrr.matrix44.multiply(self.mat_view,self.mat_model)
            S, Q, tvec = pyrr.matrix44.decompose(M)
            rvec = tools_calibrate.quaternion_to_euler(Q)

            image = tools_render_CV.draw_points_numpy_RT(self.object.coord_vert, image, self.mat_camera, numpy.zeros(4), rvec, tvec,self.mat_trns)


        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_vec(self, V, posx, posy, image):
        if V.shape[0] == 4:
            string1 = '%+1.2f %+1.2f %+1.2f %+1.2f' % (V[0], V[1], V[2], V[3])
        else:
            string1 = '%+1.2f %+1.2f %+1.2f' % (V[0], V[1], V[2])
        image = cv2.putText(image, '{0}'.format(string1), (posx, posy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(128, 128, 0), 1, cv2.LINE_AA)
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
    def stage_data(self,folder_out):
        cv2.imwrite(folder_out+'screenshot.png',self.get_image(do_debug=True))
        self.save_markers(folder_out+'markers.txt',do_transform=False)
        self.object.export_mesh(folder_out+'mesh.obj',self.object.coord_vert,self.object.idx_vertex,do_transform=False)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_markers(self,filename_out,do_transform = False):
        if do_transform:
            X = self.my_VBO.marker_positions[1:].copy()
            X = numpy.array(X)
            X[:,1] = 0-X[:,1]
            tools_IO.save_mat(X, filename_out,delim=',')
        else:
            tools_IO.save_mat(self.my_VBO.marker_positions[1:], filename_out,delim=',')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_markers(self, filename_in, filename_marker_obj,marker_scale=None):

        if marker_scale is None:
            self.marker_scale = marker_scale

        markers = tools_IO.load_mat(filename_in,dtype=numpy.float,delim=',')
        flag = self.my_VBO.remove_last_object()
        while flag==0:
            flag = self.my_VBO.remove_last_object()

        for marker in markers:
            self.my_VBO.append_object(filename_marker_obj, (0.7, 0.2, 0), do_normalize_model_file=True, svec=(self.marker_scale, self.marker_scale, self.marker_scale), tvec=marker)
        self.bind_VBO()
        return
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
    def start_append(self):
        self.on_append = True
        return
# ----------------------------------------------------------------------------------------------------------------------
    def stop_append(self):
        self.on_append = False
        return
# ----------------------------------------------------------------------------------------------------------------------
    def start_remove(self):
        self.on_remove = True
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def stop_remove(self):
        self.on_remove = False
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
    def inverce_transform_model(self,mode):
        I = pyrr.matrix44.create_identity()
        T = pyrr.matrix44.create_from_translation((0,0,0))
        if mode == 'X': I[0, 0] *= -1
        if mode == 'Y': I[1, 1] *= -1
        if mode == 'Z': I[2, 2] *= -1
        M = pyrr.matrix44.multiply(I, T)

        self.mat_trns = pyrr.matrix44.multiply(M, self.mat_trns)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        #self.reset_view(skip_transform=True)
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
        self.reset_view(skip_transform=True)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def rotate_model(self, delta_angle):

        if self.on_rotate and self.mat_model_checkpoint is not None:
            S, Q, tvec = pyrr.matrix44.decompose(self.mat_model_checkpoint)
        else:
            S, Q, tvec = pyrr.matrix44.decompose(self.mat_model)

        rvec  = tools_calibrate.quaternion_to_euler(Q) + numpy.array(delta_angle)

        rvec[0] = min(max(rvec[0],            0.01), math.pi   - 0.01)
        rvec[2] = min(max(rvec[2], -math.pi/2+0.02), math.pi/2 - 0.02)
        self.__init_mat_model(rvec,tvec)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def rotate_view(self, delta_angle):
        R = pyrr.matrix44.create_from_eulers((0,0,delta_angle))
        self.mat_view = pyrr.matrix44.multiply(self.mat_view,R)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def resize_window(self, W, H):
        self.W = W
        self.H = H
        glViewport(0, 0, W, H)
        self.transform_model('xz')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def reset_view(self,skip_transform=False):

        self.stop_rotation()
        self.stop_append()
        self.stop_remove()

        self.__init_mat_light(numpy.array((-math.pi / 2, -math.pi / 2, 0)))
        self.__init_mat_projection()
        self.__init_mat_model((0, 0, 0), (0, 0, 0))
        if not skip_transform:
            self.__init_mat_transform(self.scale)

        obj_min = self.object.coord_vert.min()
        obj_max = self.object.coord_vert.max()

        self.__init_mat_view_ETU(eye=(0, 0, -5 * (obj_max - obj_min)), target=(0, 0, 0), up=(0, -1, 0))

        return
# ----------------------------------------------------------------------------------------------------------------------
