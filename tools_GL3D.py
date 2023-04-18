# http://blog.db-in.com/cameras-on-opengl-es-2-x/
# https://3dviewer.net/
import math
import os.path

import cv2
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GLUT import *
import glfw
import OpenGL.GL.shaders
import numpy
import pyrr
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_render_GL
from CV import tools_pr_geom
import tools_wavefront
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=2)


# ----------------------------------------------------------------------------------------------------------------------
class VBO(object):
    def __init__(self):
        self.n_faces = 35000

        self.id = numpy.full(9 * self.n_faces, -1)
        self.cnt = 0
        self.iterator = 0
        self.color_offset = 3 * 3 * self.n_faces
        self.normal_offset = self.color_offset + 3 * 3 * self.n_faces
        self.texture_offset = self.normal_offset + 3 * 3 * self.n_faces

        self.data = numpy.full(self.texture_offset + 3 * 2 * self.n_faces, 0.0, dtype='float32')



        return

    # ----------------------------------------------------------------------------------------------------------------------
    def append_object(self, filename_obj, do_normalize_model_file=False, svec=(1, 1, 1), M=numpy.eye(1)):

        object = tools_wavefront.ObjLoader()
        object.load_mesh(filename_obj, do_normalize_model_file)
        object.scale_mesh(svec)

        object.transform_mesh(M)
        for idxv, idxn, idxt in zip(object.idx_vertex, object.idx_normal, object.idx_texture):
            idx_col = object.dct_obj_id[idxv[0]]
            c = (0.25, 0.75, 0.95)
            if len(object.mat_color) > 0 and (object.mat_color[idx_col] is not None):
                c = object.mat_color[idx_col]

            clr = numpy.array([c, c, c]).flatten()

            self.data[self.iterator:                    self.iterator + 9] = (object.coord_vert[idxv]).flatten()
            self.data[self.iterator + self.color_offset:self.iterator + self.color_offset + 9] = clr
            self.data[self.iterator + self.normal_offset:self.iterator + self.normal_offset + 9] = (object.coord_norm[idxn]).flatten()
            self.data[2*self.iterator // 3 + self.texture_offset:2 * self.iterator // 3 + self.texture_offset + 6] = (object.coord_texture[idxt]).flatten()

            self.id[self.iterator:                  self.iterator + 9] = self.cnt
            self.iterator += 9

        self.cnt += 1

        return object

# ----------------------------------------------------------------------------------------------------------------------
    def remove_total(self):

        self.cnt = 0
        self.iterator = 0
        self.data[:] = 0


        return
# ----------------------------------------------------------------------------------------------------------------------
    def remove_last_object(self):
        if self.cnt > 1:
            self.remove_object(self.cnt - 1)
            self.cnt -= 1
            return 0
        return 1

    # ----------------------------------------------------------------------------------------------------------------------
    def remove_object(self, id):
        if id > 0:
            idx = numpy.where(self.id == id)
            self.id[idx] = -1
            self.data[idx] = 0
            self.data[[i + self.color_offset for i in idx]] = 0
            self.data[[i + self.normal_offset for i in idx]] = 0
            self.iterator-=len(idx[0])

        return
# ----------------------------------------------------------------------------------------------------------------------
class render_GL3D(object):

    def __init__(self, filename_obj, W=640, H=480, is_visible=True,
                 do_normalize_model_file=True, projection_type='P',cam_fov_deg=90,
                 textured=True, scale=(1, 1, 1),
                 eye=None,target=(0,0,0),up=(0,0,1),
                 M_obj=numpy.eye(4)):

        glfw.init()
        self.projection_type = projection_type
        self.is_enabled_standardize_rvec = True
        self.marker_scale = 0.10
        self.scale = scale
        self.do_normalize_model_file = do_normalize_model_file
        self.bg_color = numpy.array([76, 76, 76, 1]) / 255
        #self.bg_color = numpy.array([26, 26, 26, 1]) / 255
        self.fg_color = numpy.array([0, 0, 0, 1]) / 255
        self.wired_mode = 0

        self.W, self.H = W, H
        self.cam_fov_deg = cam_fov_deg
        self.M_obj = M_obj

        glfw.window_hint(glfw.VISIBLE, is_visible)
        self.window = glfw.create_window(self.W, self.H, "GL viewer", None, None)
        glfw.make_context_current(self.window)

        self.my_VBO = VBO()

        self.object = self.init_object(filename_obj,M_obj)

        if (eye is None):
            self.eye_default, self.target_default, self.up_default = self.get_default_ETU()
        else:
            self.eye_default,self.target_default,self.up_default   = eye,target,up

        self.textured = textured
        self.update_texture()
        self.__init_shader()
        self.bind_VBO()
        self.reset_view()
        self.ctrl_pressed = False
        self.acc_pos = numpy.zeros(3)

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init_object(self,filename_obj,M_obj=numpy.eye(4)):
        self.object = None
        if filename_obj is not None:
            if os.path.isdir(filename_obj):
                for f in tools_IO.get_filenames(filename_obj,'*.obj'):
                    self.object = self.my_VBO.append_object(filename_obj+f, self.do_normalize_model_file, M=M_obj)
            else:
                if os.path.isfile(filename_obj):
                    self.object = self.my_VBO.append_object(filename_obj, self.do_normalize_model_file, M=M_obj)
        return self.object
    # ----------------------------------------------------------------------------------------------------------------------
    def __init_shader(self):
        # projection * view * model * transform
        # model maps from an object's local coordinate space into world space,
        # view from world space to camera space,
        # projection from camera to screen.

        vert_shader = """#version 400
                                    in layout(location = 0) vec3 position;
                                    in layout(location = 1) vec3 color;
                                    in layout(location = 2) vec3 vertNormal;
                                    in layout(location = 3) vec2 inTextureCoords;

                                    uniform mat4 transform,view,model,projection,light;
                                    out vec3 inColor;
                                    out vec3 fragNormal;
                                    out vec2 textureCoords;

                                    void main()
                                    {
                                        fragNormal = -abs((light * view * model * transform * vec4(vertNormal, 0.0f)).xyz);
                                        gl_Position = projection * view * model * transform * vec4(position, 1.0f);
                                        inColor = color;
                                        textureCoords = inTextureCoords;
                                    }"""

        frag_shader_texture = """#version 400
                                    in vec3 inColor;
                                    in vec3 fragNormal;
                                    in vec2 textureCoords;
                                    out vec4 outColor;
                                    uniform sampler2D samplerTexture;
                                    void main()
                                    {
                                        vec3 ambientLightIntensity  = vec3(0.1f, 0.1f, 0.1f);
                                        vec3 sunLightIntensity      = vec3(1.0f, 1.0f, 1.0f);
                                        vec3 sunLightDirection      = normalize(vec3(+0.0f, -1.0f, +0.0f));
                                        vec4 texel = texture(samplerTexture, textureCoords);

                                        vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * dot(fragNormal, sunLightDirection);
                                        vec4 materialColor = vec4(inColor*lightIntensity, 1);

                                        //outColor = materialColor;
                                        outColor = texel; //eg for soccer - texture without lighting effect 
                                        //outColor = texel* materialColor;//
                                        //outColor = vec4(texel.rgb * lightIntensity, 1);
                                        outColor.a = 1.00;                                        
                                    }"""

        frag_shader_color = """#version 400
                                    in vec3 inColor;
                                    in vec3 fragNormal;
                                    in vec2 textureCoords;
                                    out vec4 outColor;
                                    uniform sampler2D samplerTexture;
                                    void main()
                                    {
                                        vec3 ambientLightIntensity  = vec3(0.1f, 0.1f, 0.1f);
                                        vec3 sunLightIntensity      = vec3(1.0f, 1.0f, 1.0f);
                                        vec3 sunLightDirection      = normalize(vec3(+0.0f, -1.0f, +0.0f));
                                        vec4 texel = texture(samplerTexture, textureCoords);

                                        vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * dot(fragNormal, sunLightDirection);
                                        vec4 materialColor = vec4(inColor*lightIntensity, 1);
                                        

                                        outColor = materialColor; // color material with lighting and shaddow effect 
                                        //outColor = texel;//soccer
                                        //outColor = texel* materialColor;//
                                        //outColor = vec4(texel.rgb * lightIntensity, 1);
                                        outColor.a = 0.99;  //transparency                                        
                                    }"""





        if self.textured:
            self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vert_shader, GL_VERTEX_SHADER),OpenGL.GL.shaders.compileShader(frag_shader_texture, GL_FRAGMENT_SHADER))
        else:
            self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vert_shader, GL_VERTEX_SHADER),OpenGL.GL.shaders.compileShader(frag_shader_color, GL_FRAGMENT_SHADER))

        glUseProgram(self.shader)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def update_background(self,filename_obj):

        tg_half_fovx =1.0/ self.mat_projection[0][0]

        t0 = (0, 0.5 * self.W / tg_half_fovx, 0)
        M_obj0 = tools_pr_geom.compose_RT_mat((0, 0, 0), t0, do_rodriges=False, do_flip=False, GL_style=True)
        mat_view0 = tools_pr_geom.ETU_to_mat_view((0, 0, 0), (0, 1, 0), (0, 0, -1))
        mat_model0 = tools_render_GL.compose_RT_mat_GL((0,0,0), (0,0,0), do_rodriges=False, do_flip=False)

        M = tools_pr_geom.multiply([M_obj0,
                                    tools_pr_geom.multiply([self.mat_trns,mat_model0]),
                                    tools_pr_geom.multiply([mat_view0,numpy.linalg.pinv(self.mat_view)]),
                                    tools_pr_geom.multiply([numpy.linalg.pinv(self.mat_model),numpy.linalg.pinv(self.mat_trns)]),
                                    ])

        self.my_VBO.remove_last_object()
        if filename_obj is not None:
            self.my_VBO.append_object(filename_obj, do_normalize_model_file=False,M=M)
        self.bind_VBO(self.wired_mode)

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update_texture(self):

        if self.object is None:
            texture_image = numpy.full((10, 10, 3), 255)
            self.textured = False
        else:
            if (self.object.filename_texture[-1] is None) or (not os.path.exists(self.object.filename_texture[-1])):
                texture_image = numpy.full((10, 10, 3), 255)
                self.textured = False
            else:
                texture_image = cv2.cvtColor(cv2.imread(self.object.filename_texture[-1]), cv2.COLOR_BGR2RGB)

        glBindTexture(GL_TEXTURE_2D, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_image.shape[1], texture_image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, texture_image.flatten())
        glEnable(GL_TEXTURE_2D)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def bind_VBO(self, wired_mode=0):

        glBindBuffer(GL_ARRAY_BUFFER, glGenBuffers(1))

        item_size = self.my_VBO.data.itemsize

        glBufferData(GL_ARRAY_BUFFER, item_size * len(self.my_VBO.data), self.my_VBO.data, GL_STATIC_DRAW)

        # positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, item_size * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, item_size * 3,ctypes.c_void_p(item_size * self.my_VBO.color_offset))
        glEnableVertexAttribArray(1)

        # normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, item_size * 3,ctypes.c_void_p(item_size * self.my_VBO.normal_offset))
        glEnableVertexAttribArray(2)

        # textures
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, item_size * 2,ctypes.c_void_p(item_size * self.my_VBO.texture_offset))
        glEnableVertexAttribArray(3)

        glClearColor(self.bg_color[0], self.bg_color[1], self.bg_color[2], self.bg_color[3])
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

        if wired_mode==0:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        elif wired_mode==1:
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)


        return

    # ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_projection_perspective(self, a_fov_x=0.5, a_fov_y=0.5):
        self.mat_projection = tools_pr_geom.compose_projection_mat_4x4_GL(self.W, self.H, a_fov_x, a_fov_y)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, self.mat_projection)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_projection_ortho(self, scale_factor=(5, 5)):
        fx, fy = float(self.W), float(self.H)
        near, far = 0, 1000
        scale_factor_x, scale_factor_y = scale_factor[0], scale_factor[1]
        # left, right, bottom, top = (0.25*scale_factor_x*numpy.array([-1,1,-1,1])).tolist()
        left, right, bottom, top = -0.5 * scale_factor_x, +0.5 * scale_factor_x, -0.5 * scale_factor_y, +0.5 * scale_factor_y

        self.mat_projection = pyrr.matrix44.create_orthogonal_projection(left, right, bottom, top, near, far)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, self.mat_projection)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init_mat_projection(self, a_fov_x=0.5, a_fov_y=0.5):
        if self.projection_type == 'P':
            self.__init_mat_projection_perspective(a_fov_x, a_fov_y)
        else:
            self.__init_mat_projection_ortho()
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init_mat_view_direct(self, mat_view):
        self.mat_view = mat_view
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init_mat_view_ETU(self, eye, target, up):
        self.mat_view = tools_pr_geom.ETU_to_mat_view(numpy.array(eye), numpy.array(target), numpy.array(up))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init_mat_view_RT(self, rvec, tvec, do_rodriges=False, do_flip=True):
        self.mat_view = tools_render_GL.compose_RT_mat_GL(rvec, tvec, do_rodriges=do_rodriges, do_flip=do_flip)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init_mat_model_direct(self, mat_model):
        self.mat_model = mat_model
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.mat_model)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init_mat_model(self, rvec, tvec, do_rodriges=False, do_flip=False):
        self.mat_model = tools_render_GL.compose_RT_mat_GL(rvec, tvec, do_rodriges=do_rodriges, do_flip=do_flip)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.mat_model)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_transform(self, scale_vec):
        self.mat_trns = pyrr.Matrix44.from_scale(scale_vec)
        # !!!
        # self.mat_trns[1,1]*=-1
        # self.mat_trns[2,2]*=-1
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init_mat_transform_direct(self, mat_transform):
        self.mat_trns = mat_transform
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_light(self, r_vec):
        self.mat_light = pyrr.matrix44.create_from_eulers(r_vec)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "light"), 1, GL_FALSE, self.mat_light)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, self.my_VBO.iterator // 3)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def get_image(self, mat_proj=None, mat_view=None, mat_model=None, mat_trans=None, do_debug=False):

        if mat_proj is not None: self.mat_projection = mat_proj
        if mat_view is not None: self.mat_view = mat_view
        if mat_model is not None: self.mat_model = mat_model
        if mat_trans is not None: self.mat_trns = mat_trans

        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, self.mat_projection)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.mat_model)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)

        self.draw()
        image_buffer = glReadPixels(0, 0, self.W, self.H, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(self.H, self.W, 3)
        image = cv2.flip(image[:, :, [2, 1, 0]], 0)
        if do_debug: image = self.draw_debug_info(image)
        return image

    # ----------------------------------------------------------------------------------------------------------------------
    def init_perspective_view_1_mat_model(self, rvec, tvec, a_fov_x=0.5, a_fov_y=0.5, scale=(1, 1, 1)):
        # OK
        tvec = numpy.array(tvec, dtype=float)
        self.init_mat_view_RT(numpy.array(rvec, dtype=float), tvec, do_rodriges=True, do_flip=True)
        self.init_mat_model_direct(numpy.eye(4))
        self.__init_mat_transform(scale)
        self.__init_mat_projection_perspective(a_fov_x, a_fov_y)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init_perspective_view_1_mat_view(self, rvec, tvec, a_fov_x=0.5, a_fov_y=0.5, scale=(1, 1, 1)):
        tvec = numpy.array(tvec, dtype=float)
        self.init_mat_model(numpy.array(rvec, dtype=float), tvec, do_rodriges=True, do_flip=True)
        self.init_mat_view_direct(numpy.eye(4))
        self.__init_mat_transform(scale)
        self.__init_mat_projection_perspective(a_fov_x, a_fov_y)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def get_image_perspective(self, rvec, tvec, a_fov_x=0.5, a_fov_y=0.5, scale=(1, 1, 1), lookback=False, mat_view_to_1=True, do_debug=False):
        if mat_view_to_1:
            self.init_perspective_view_1_mat_view(rvec, tvec, a_fov_x, a_fov_y, scale)
            if lookback:
                self.mat_view *= -1
                self.mat_view[-1, -1] *= -1
        else:
            self.init_perspective_view_1_mat_model(rvec, tvec, a_fov_x, a_fov_y, scale)

        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.mat_model)
        return self.get_image(do_debug=do_debug)

    # ----------------------------------------------------------------------------------------------------------------------
    def get_image_perspective_M(self, mat_M, a_fov_x=0.5, a_fov_y=0.5, scale=(1, 1, 1), lookback=False, mat_view_to_1=True,do_debug=False):

        if mat_view_to_1:
            self.mat_model = mat_M
            self.init_mat_view_direct(numpy.eye(4))
            glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)
            glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.mat_model)

        else:
            self.mat_view = mat_M
            self.init_mat_model_direct(numpy.eye(4))
            glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)

        self.__init_mat_transform(scale)
        self.__init_mat_projection_perspective(a_fov_x, a_fov_y)

        return self.get_image(do_debug=do_debug)
    # ----------------------------------------------------------------------------------------------------------------------
    def init_ortho_view(self, rvec, tvec, scale_factor):
        rvec = numpy.array(rvec)
        self.__init_mat_projection_ortho(scale_factor)
        self.init_mat_model(rvec, tvec)
        self.__init_mat_view_RT((0, 0, 0), (0, 0, +1))
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def get_image_ortho(self, rvec, tvec, scale_factor, do_debug=False):
        self.init_ortho_view(rvec, tvec, scale_factor)
        return self.get_image(do_debug=do_debug)

    # ----------------------------------------------------------------------------------------------------------------------
    def draw_debug_info(self, image,draw_points=False):

        image_result = image.copy()
        if draw_points:
            image_result = tools_render_GL.draw_points_MVP_GL(self.object.coord_vert, image_result, self.mat_projection, self.mat_view, self.mat_model, self.mat_trns, w=6,do_debug=False)

        camera_matrix_3x3 = tools_pr_geom.compose_projection_mat_3x3(self.W, self.H, 1 / self.mat_projection[0][0], 1 / self.mat_projection[1][1])
        image_result = tools_draw_numpy.draw_mat(self.mat_trns      , 20, 20, image_result, color=255*self.fg_color,text='World')
        image_result = tools_draw_numpy.draw_mat(self.mat_model     , 20, 120, image_result,color=255*self.fg_color,text='Model')
        image_result = tools_draw_numpy.draw_mat(self.mat_view      , 20, 220, image_result,color=255*self.fg_color,text='View')
        image_result = tools_draw_numpy.draw_mat(self.mat_projection, 20, 320, image_result,color=255*self.fg_color,text='Projection')
        image_result = tools_draw_numpy.draw_mat(camera_matrix_3x3  , 20, 420, image_result,color=255*self.fg_color, text='Mat camera')

        Ym, Pm, Rm = tools_pr_geom.mat_view_to_YPR(self.mat_model)
        E, T, U = tools_pr_geom.mat_view_to_ETU(self.mat_view)
        Yc, Pc, Rc = tools_pr_geom.mat_view_to_YPR(self.mat_view)
        tg_half_fovx = 1.0 / (self.mat_projection[0][0])
        fovx_deg = 2 * numpy.arctan(tg_half_fovx)*180/numpy.pi

        image_result = tools_draw_numpy.draw_mat(numpy.array((Ym, Pm, Rm)) * 180 / numpy.pi, 20, 520, image_result,color=255*self.fg_color, text='Model Yaw Pitch Roll [deg]')
        image_result = tools_draw_numpy.draw_mat(E, 20, 560, image_result,color=255*self.fg_color,text='Camera Eye')
        image_result = tools_draw_numpy.draw_mat(T, 20, 580, image_result, color=255 * self.fg_color, text='Camera Target')
        image_result = tools_draw_numpy.draw_mat(U, 20, 600, image_result, color=255 * self.fg_color, text='Camera Up')

        image_result = tools_draw_numpy.draw_mat(numpy.array((Yc, Pc, Rc)) * 180 / numpy.pi, 20, 620, image_result, color=255 * self.fg_color, text='Camera Yaw Pitch Roll [deg]')
        image_result = tools_draw_numpy.draw_mat([fovx_deg], 20, 640, image_result,color=255*self.fg_color, text='Camera fov x deg')

        return image_result

    # ----------------------------------------------------------------------------------------------------------------------
    def stage_data(self, folder_out):
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        filenames = tools_IO.get_filenames(folder_out, 'screenshot*.png')
        ids = [(f.split('.')[0]).split('_')[1] for f in filenames]
        if len(ids) > 0:
            i = 1 + numpy.array(ids, dtype=int).max()
        else:
            i = 0

        cv2.imwrite(folder_out + 'screenshot_%03d.png' % i, self.get_image(do_debug=True))

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def load_markers(self, filename_in, filename_marker_obj, marker_scale=None):

        if marker_scale is not None:
            self.marker_scale = marker_scale

        markers = tools_IO.load_mat(filename_in, dtype=numpy.float, delim=' ')
        flag = self.my_VBO.remove_last_object()
        while flag == 0:
            flag = self.my_VBO.remove_last_object()

        for marker_t in markers:
            M = tools_pr_geom.compose_RT_mat((0,0,0),marker_t,do_rodriges=False,do_flip=False, GL_style=True)#?
            self.my_VBO.append_object(filename_marker_obj, do_normalize_model_file=True,svec=(self.marker_scale, self.marker_scale, self.marker_scale), M=M)
        self.bind_VBO(self.wired_mode)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def start_rotation(self):
        self.on_rotate = True
        self.mat_model_rotation_checkpoint = self.mat_model.copy()
        self.mat_view_rotation_checkpoint = self.mat_view.copy()
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def stop_rotation(self):
        self.on_rotate = False
        self.mat_model_rotation_checkpoint = None
        self.mat_view_rotation_checkpoint = None
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def start_translation(self):
        self.on_translate = True
        self.mat_model_translation_checkpoint = self.mat_model.copy()
        self.mat_view_translation_checkpoint = self.mat_view.copy()
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def stop_translation(self):
        self.on_translate = False
        self.mat_model_translation_checkpoint = None
        self.mat_view_translation_checkpoint = None
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def scale_projection(self, factor):
        scale_current = self.mat_projection[0][0] / 2
        a_fov_current = (0.5 / scale_current)

        self.init_mat_projection(factor * a_fov_current, factor * a_fov_current)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def translate_view_by_scale(self, scale):
        rvec, tvec = tools_pr_geom.decompose_to_rvec_tvec(self.mat_view, do_flip=True)
        tvec *= scale
        self.init_mat_view_RT(rvec, tvec)

        # E,T,U = tools_pr_geom.mat_view_to_ETU(self.mat_view)
        # F = T-E
        # E*=delta_translate
        # T = E+F
        # self.init_mat_view_ETU(E, T, U)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def translate_view(self, delta_translate):

        if self.on_translate and self.mat_view_translation_checkpoint is not None:
            self.mat_view = self.mat_view_translation_checkpoint

        T = pyrr.matrix44.create_from_translation(numpy.array(delta_translate))
        self.mat_view = pyrr.matrix44.multiply(self.mat_view, T)

        Yc, Pc, Rc = tools_pr_geom.mat_view_to_YPR(self.mat_view)
        self.mat_view  = tools_pr_geom.compose_RT_mat((Yc, Pc, 0), self.mat_view[3, :3], do_rodriges=False,do_flip=False,GL_style=True)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)


        return

    # ----------------------------------------------------------------------------------------------------------------------
    def rotate_view(self, delta_angle):
        if self.on_rotate and self.mat_view_rotation_checkpoint is not None:
            self.mat_view = self.mat_view_rotation_checkpoint

        R = pyrr.matrix44.create_from_eulers(numpy.array((delta_angle[0], delta_angle[1], delta_angle[2])))
        RR = pyrr.matrix44.multiply(pyrr.matrix44.multiply(self.mat_view, R), pyrr.matrix44.inverse(self.mat_view))
        self.mat_view = pyrr.matrix44.multiply(RR, self.mat_view)
        Yc, Pc, Rc = tools_pr_geom.mat_view_to_YPR(self.mat_view)
        self.mat_view = tools_pr_geom.compose_RT_mat((Yc, Pc, 0), self.mat_view[3, :3], do_rodriges=False,do_flip=False, GL_style=True)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, self.mat_view)

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def translate_ortho(self, delta_translate):
        factor = 2 / self.mat_projection[0, 0]
        factor *= delta_translate
        self.__init_mat_projection_ortho(factor)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def scale_model_vector(self, scale_vector):
        self.mat_trns[0, 0] *= scale_vector[0]
        self.mat_trns[1, 1] *= scale_vector[1]
        self.mat_trns[2, 2] *= scale_vector[2]
        self.mat_trns[3, 3] = 1
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def inverce_transform_model(self, mode):
        I = pyrr.matrix44.create_identity()
        T = pyrr.matrix44.create_from_translation((0, 0, 0))
        if mode == 'X': I[0, 0] *= -1
        if mode == 'Y': I[1, 1] *= -1
        if mode == 'Z': I[2, 2] *= -1
        M = pyrr.matrix44.multiply(I, T)

        self.mat_trns = pyrr.matrix44.multiply(M, self.mat_trns)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        # self.reset_view(skip_transform=True)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def transform_model(self, mode):

        T = pyrr.Matrix44.from_scale(self.scale)
        if mode == 'XY':
            M = pyrr.matrix44.create_from_z_rotation(-math.pi / 2)
        elif mode == 'XZ':
            M = pyrr.matrix44.create_from_y_rotation(-math.pi / 2)
        elif mode == 'YZ':
            M = pyrr.matrix44.create_from_x_rotation(-math.pi / 2)
        elif mode == 'xy':
            M = pyrr.matrix44.create_from_z_rotation(+math.pi / 2)
        elif mode == 'yz':
            M = pyrr.matrix44.create_from_y_rotation(+math.pi / 2)
        elif mode == 'xz':
            M = pyrr.matrix44.create_from_x_rotation(+math.pi / 2)
        self.mat_trns = numpy.dot(T, M)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "transform"), 1, GL_FALSE, self.mat_trns)
        self.reset_view(skip_transform=True)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def set_standardize_rvec(self, flag):
        self.is_enabled_standardize_rvec = flag
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def standardize_rvec(self, rvec):
        if self.is_enabled_standardize_rvec:
            rvec[0] = min(max(rvec[0], 0.01), math.pi - 0.01)
            rvec[2] = min(max(rvec[2], -math.pi / 2 + 0.02), math.pi / 2 - 0.02)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def rotate_model(self, delta_angle, silent=False):

        if self.on_rotate and self.mat_model_rotation_checkpoint is not None:
            S, Q, tvec = pyrr.matrix44.decompose(self.mat_model_rotation_checkpoint.copy())
        else:
            S, Q, tvec = pyrr.matrix44.decompose(self.mat_model.copy())

        rvec = tools_pr_geom.quaternion_to_euler(Q)
        rvec += numpy.array(delta_angle)
        # self.standardize_rvec(rvec)

        self.init_mat_model(rvec, tvec)

        # derive rotation, translation
        # if not silent:
        # r, t = tools_pr_geom.decompose_to_rvec_tvec(self.mat_model)
        # print('%1.2f %1.2f %1.2f'%(r[0]*180/numpy.pi,r[1]*180/numpy.pi,r[2]*180/numpy.pi))
        # r, t = tools_pr_geom.decompose_to_rvec_tvec(pyrr.matrix44.multiply(self.mat_model, pyrr.matrix44.inverse(self.mat_model_init)))
        # print(r, t)

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def translate_model_centric(self, delta_pos):

        self.acc_pos += delta_pos

        # if self.on_translate and self.mat_model_translation_checkpoint is not None:
        #    self.mat_model = self.mat_model_translation_checkpoint.copy()

        self.init_mat_model((0, 0, 0), (0, 0, 0))
        self.rotate_model((+2 * numpy.pi / 3, 0, 0), silent=True)
        T = pyrr.matrix44.create_from_translation(numpy.array(self.acc_pos))
        self.mat_model = pyrr.matrix44.multiply(T, self.mat_model)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.mat_model)

        # derive rotation, translation
        # r, t = tools_pr_geom.decompose_to_rvec_tvec(self.mat_model)
        # print('%1.2f %1.2f %1.2f' % (r[0] * 180 / numpy.pi, r[1] * 180 / numpy.pi, r[2] * 180 / numpy.pi))

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def translate_model(self, delta_pos):

        if self.on_translate and self.mat_model_translation_checkpoint is not None:
            rvec, tvec = tools_render_GL.decompose_to_rvec_tvec_GL(self.mat_model_translation_checkpoint.copy())
        else:
            rvec, tvec = tools_render_GL.decompose_to_rvec_tvec_GL(self.mat_model.copy())

        tvec += numpy.array(delta_pos)
        self.init_mat_model(rvec, tvec,do_rodriges=True, do_flip=True)


        return

    # ----------------------------------------------------------------------------------------------------------------------
    def resize_window(self, W, H):
        self.W = W
        self.H = H
        glViewport(0, 0, W, H)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def get_default_ETU(self):
        obj_min = self.object.coord_vert.min()
        obj_max = self.object.coord_vert.max()
        # eye = numpy.array((0, 0, +5 * (obj_max - obj_min)))
        # target = eye - numpy.array((0, 0, 1.0))
        # up = numpy.array((0, -1, 0.0))

        eye = numpy.array((0, 0, -5 * (obj_max - obj_min)))
        target = eye + numpy.array((0, 0, 1.0))
        up = numpy.array((0, -1, 0.0))

        return eye, target, up
    # ----------------------------------------------------------------------------------------------------------------------
    def get_best_collision(self, collisons):
        if collisons is None:
            return None


        collisons = numpy.array(collisons).reshape((-1, 3))
        X4D = numpy.hstack((collisons, numpy.full((collisons.shape[0], 1), 1)))

        X = tools_pr_geom.multiply([X4D,self.M_obj,self.mat_model,self.mat_trns])[:,:3]
        E, T, U = tools_pr_geom.mat_view_to_ETU(self.mat_view)
        n = (T-E)/numpy.linalg.norm(T-E)
        min_d = numpy.inf
        best_collision = None
        for collision,x in zip(collisons, X):
            if (numpy.dot((x-E),n)>0) and (numpy.linalg.norm(E-x)<min_d):
                min_d = numpy.linalg.norm(E-x)
                best_collision = collision


        return best_collision
# ----------------------------------------------------------------------------------------------------------------------
    def reset_view(self, skip_transform=False):

        self.stop_rotation()
        self.stop_translation()
        self.acc_pos = numpy.zeros(3)

        self.__init_mat_light(numpy.array((-math.pi / 2, -math.pi / 2, 0)))

        tg_half_fovx = numpy.tan(self.cam_fov_deg * numpy.pi / 360)
        self.init_mat_projection(tg_half_fovx, tg_half_fovx)
        self.init_mat_model((0, 0, 0), (0, 0, 0))
        if not skip_transform:
            self.__init_mat_transform(self.scale)


        self.init_mat_view_ETU(self.eye_default, self.target_default, self.up_default)

        #E, T, U = tools_pr_geom.mat_view_to_ETU(self.mat_view)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def BEV_view(self):
        self.reset_view()
        eye, target, up = self.get_default_ETU()
        self.init_mat_view_ETU(eye/5.0, target/5.0, up)
        return
# ----------------------------------------------------------------------------------------------------------------------
# self.__init_mat_view_RT((0,0,0),(0,0,+5 * (obj_max - obj_min)))
# self.init_mat_view_ETU(eye=(0, 0, 5), target=(0, 0, 0), up=(0, -1, 0))