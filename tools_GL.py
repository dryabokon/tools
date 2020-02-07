import cv2
from OpenGL.GL import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image,ImageDraw
import glfw
from OpenGL.GLUT import *
import numpy
import tools_aruco
# ----------------------------------------------------------------------------------------------------------------------
class render_GL(object):

    def __init__(self,image_texture):

        #glfw.init()
        #glfw.window_hint(glfw.VISIBLE, False)
        glutInit()
        glutCreateWindow("OpenGL")
        glutHideWindow()
        #self.window = glfw.create_window(200, 200, "hidden window", None, None)
        self.texture_is_setup = False
        self.image_texture = image_texture
        return
# ----------------------------------------------------------------------------------------------------------------------
    def update_texture(self, image_texture):
        self.image_texture = image_texture
        self.texture_is_setup = False
        return
# ----------------------------------------------------------------------------------------------------------------------
    def clean(self):
        glfw.destroy_window(self.window)
        glfw.terminate()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __refresh2d(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, width, 0.0, height, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __initTexture(self, inputimage):

        image = Image.fromarray(inputimage)
        width = image.size[0]
        height = image.size[1]

        image = image.tobytes("raw", "RGBX", 0, -1)
        texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, texture)  # 2d texture (x and y size)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        gluBuild2DMipmaps(GL_TEXTURE_2D, 3, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image)
        return texture
# ----------------------------------------------------------------------------------------------------------------------
    def __draw_image(self, image):

        glfw.make_context_current(self.window)
        glEnable(GL_TEXTURE_2D)

        self.__initTexture(image)
        height, width, _ = image.shape
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        self.__refresh2d(width, height)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1);glVertex2f(0, 0)
        glTexCoord2f(1, 1);glVertex2f(width, 0)
        glTexCoord2f(1, 0);glVertex2f(width, height)
        glTexCoord2f(0, 0);glVertex2f(0, height)

        glEnd()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __draw_2d_mesh(self, H,W,points_coord,points_text_coord, triangles):
        glfw.make_context_current(self.window)
        glEnable(GL_TEXTURE_2D)

        if not self.texture_is_setup:
            self.__initTexture(self.image_texture)
            self.texture_is_setup = True

        texture_height, texture_width, _ = self.image_texture.shape
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        self.__refresh2d(W, H)

        glBegin(GL_TRIANGLES)

        for triangle in triangles:
            x0 = points_coord[triangle[0],0]
            y0 = points_coord[triangle[0],1]
            x1 = points_coord[triangle[1],0]
            y1 = points_coord[triangle[1],1]
            x2 = points_coord[triangle[2],0]
            y2 = points_coord[triangle[2],1]

            tx0 = points_text_coord[triangle[0], 0]
            ty0 = points_text_coord[triangle[0], 1]
            tx1 = points_text_coord[triangle[1], 0]
            ty1 = points_text_coord[triangle[1], 1]
            tx2 = points_text_coord[triangle[2], 0]
            ty2 = points_text_coord[triangle[2], 1]

            glTexCoord2f(float(tx0/texture_width), texture_height-float(ty0/texture_height))
            glVertex2f(x0, y0)

            glTexCoord2f(float(tx1/texture_width), texture_height-float(ty1/texture_height))
            glVertex2f(x1, y1)

            glTexCoord2f(float(tx2/texture_width), texture_height-float(ty2/texture_height))
            glVertex2f(x2, y2)

        glEnd()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_image(self,image):
        width,height = image.shape[1], image.shape[0]
        glfw.set_window_size(self.window, width,height)
        self.__draw_image(image)
        image_buffer = glReadPixels(0, 0, width,height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(height, width, 3)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def morph_2D_mesh(self, H, W, points_coord, points_text_coord, triangles):
        height,width = H,W
        glfw.set_window_size(self.window, width, height)
        self.__draw_2d_mesh(H,W,points_coord,points_text_coord, triangles)
        image_buffer = glReadPixels(0, 0, width, height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(height, width, 3)
        return image[:height,:width]
# ----------------------------------------------------------------------------------------------------------------------
    def __draw_3d_mesh0(self,H,W,frame,rvec, tvec):

        glfw.make_context_current(self.window)
        glEnable(GL_TEXTURE_2D)


        height, width, _ = frame.shape
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        self.__refresh2d(width, height)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDrawPixels(frame.shape[1], frame.shape[0], GL_BGR, GL_UNSIGNED_BYTE, frame[:, :, [2, 1, 0]])
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        fx, fy = 1090, 1090
        image_shape = 400, 600
        principalX, principalY = fx / 2, fy / 2
        near = 1
        far = 1050
        marker_length = 0.1

        left = -principalX / fx
        right = (image_shape[1] - principalX) / fx
        bottom = (principalY - image_shape[0]) / fy
        top = principalY / fy
        glFrustum(left, right, bottom, top, near, far)

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(tools_aruco.compose_GL_MAT(rvec, tvec))
        tools_aruco.draw_native_axis(marker_length / 2)

        glPushMatrix()
        glRotatef(90, +1, 0, 0)
        glutSolidTeapot(marker_length / 4)
        #tools_aruco.draw_cube(size=marker_length/4)
        glPopMatrix()
        glFlush()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def __draw_3d_mesh1(self, H, W, frame, rvec, tvec):
        glfw.make_context_current(self.window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDrawPixels(frame.shape[1], frame.shape[0], GL_BGR, GL_UNSIGNED_BYTE, frame[:,:,[2,1,0]])
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        fx, fy = 1090, 1090
        image_shape = 400, 600
        principalX, principalY = fx / 2, fy / 2
        near = 1
        far = 1050
        marker_length = 0.1

        left = -principalX / fx
        right = (image_shape[1] - principalX) / fx
        bottom = (principalY - image_shape[0]) / fy
        top = principalY / fy
        glFrustum(left, right, bottom, top, near, far)

        # a = (GLfloat * 16)()
        # glGetFloatv(GL_PROJECTION_MATRIX, a)
        # a= list(a)

        if numpy.count_nonzero(rvec) > 0:
            glMatrixMode(GL_MODELVIEW)
            glLoadMatrixf(tools_aruco.compose_GL_MAT(rvec, tvec))
            tools_aruco.draw_native_axis(marker_length / 2)

            glPushMatrix()
            glRotatef(90, +1, 0, 0)
            #glutSolidTeapot(marker_length / 4)
            tools_aruco.draw_cube(size=marker_length/4,pos_x=0, pos_y=0, pos_z=+1)
            #tools_aruco.draw_point((10,10,10))
            glPopMatrix()

        #glutSwapBuffers()
        glFlush()
        #glutPostRedisplay()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __draw_3d_mesh(self, H, W, frame, rvec, tvec):
        xrot = 0
        yrot = 0
        lightpos = (1,1,1)
        greencolor = (0.2, 0.8, 0.0, 0.8)  # Зеленый цвет для иголок
        treecolor = (0.9, 0.6, 0.3, 0.8)  # Коричневый цвет для ствола

        #glfw.make_context_current(self.window)
        glClear(GL_COLOR_BUFFER_BIT)  # Очищаем экран и заливаем серым цветом
        glPushMatrix()  # Сохраняем текущее положение "камеры"
        glRotatef(xrot, 1.0, 0.0, 0.0)  # Вращаем по оси X на величину xrot
        glRotatef(yrot, 0.0, 1.0, 0.0)  # Вращаем по оси Y на величину yrot
        glLightfv(GL_LIGHT0, GL_POSITION, lightpos)  # Источник света вращаем вместе с елкой

        # Рисуем ствол елки
        # Устанавливаем материал: рисовать с 2 сторон, рассеянное освещение, коричневый цвет
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, treecolor)
        glTranslatef(0.0, 0.0, -0.7)  # Сдвинемся по оси Z на -0.7
        # Рисуем цилиндр с радиусом 0.1, высотой 0.2
        # Последние два числа определяют количество полигонов
        glutSolidCylinder(0.1, 0.2, 20, 20)
        # Рисуем ветки елки
        # Устанавливаем материал: рисовать с 2 сторон, рассеянное освещение, зеленый цвет
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, greencolor)
        glTranslatef(0.0, 0.0, 0.2)  # Сдвинемся по оси Z на 0.2
        # Рисуем нижние ветки (конус) с радиусом 0.5, высотой 0.5
        # Последние два числа определяют количество полигонов
        glutSolidCone(0.5, 0.5, 20, 20)
        glTranslatef(0.0, 0.0, 0.3)  # Сдвинемся по оси Z на -0.3
        glutSolidCone(0.4, 0.4, 20, 20)  # Конус с радиусом 0.4, высотой 0.4
        glTranslatef(0.0, 0.0, 0.3)  # Сдвинемся по оси Z на -0.3
        glutSolidCone(0.3, 0.3, 20, 20)  # Конус с радиусом 0.3, высотой 0.3

        glPopMatrix()  # Возвращаем сохраненное положение "камеры"
        glutSwapBuffers()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def morph_3D_mesh(self, H, W,image,r_vec, t_vec):
        height,width = H,W
        #glfw.set_window_size(self.window, width, height)
        self.__draw_3d_mesh(H,W,image,r_vec, t_vec)
        image_buffer = glReadPixels(0, 0, width, height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(height, width, 3)
        return image[:height,:width]
# ----------------------------------------------------------------------------------------------------------------------
