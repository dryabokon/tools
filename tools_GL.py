from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
import glfw
import OpenGL.GL.shaders
import numpy
# ----------------------------------------------------------------------------------------------------------------------
class render_GL(object):

    def __init__(self,image_texture):

        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, False)
        self.window = glfw.create_window(640, 480, "hidden window", None, None)
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
        image = image.tobytes("raw", "RGBX", 0, -1)
        texture = glGenTextures(1)
        width = inputimage.shape[1]
        height = inputimage.shape[0]

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
        glfw.set_window_size(self.window, W, H)
        self.__draw_2d_mesh(H,W,points_coord,points_text_coord, triangles)
        image_buffer = glReadPixels(0, 0, W, H, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(H, W, 3)
        return image[:H,:W]
# ----------------------------------------------------------------------------------------------------------------------
