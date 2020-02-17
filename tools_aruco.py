import cv2
import cv2.aruco as aruco
from OpenGL.GL import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------------
def draw_native_axis(length):

    glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT)

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glDisable(GL_LIGHTING)

    glBegin(GL_LINES)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)

    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)

    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, -length)
    glEnd()

    glPopAttrib()
    return
# ----------------------------------------------------------------------------------------------------------------------
def detect_marker_and_draw_axes(frame,marker_length,camera_matrix, dist):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), aruco.Dictionary_get(aruco.DICT_6X6_50))
    rvec, tvec = numpy.array([[[0,0,0]]]),numpy.array([[[0,0,0]]])
    res = None

    if len(corners) > 0:
        res = frame.copy()
        aruco.drawDetectedMarkers(res, corners)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], marker_length, camera_matrix, dist)
        #aruco.drawAxis(frame,camera_matrix,dist,rvec[0], tvec[0], marker_length / 2)
        res= tools_render_CV.draw_axis(frame, camera_matrix, dist, rvec[0], tvec[0], marker_length / 2)

    return res, rvec, tvec
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube(image_texture=None,pos_x=0, pos_y=0, pos_z=0,size=1):

    if image_texture != None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_texture.shape[1], image_texture.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, image_texture.tobytes())

    glBegin(GL_QUADS)


    glTexCoord2f(0.0, 0.0);    glVertex3f(pos_x-size/2, pos_y-size/2, pos_z+size/2)
    glTexCoord2f(1.0, 0.0);    glVertex3f(pos_x+size/2, pos_y-size/2, pos_z+size/2)
    glTexCoord2f(1.0, 1.0);    glVertex3f(pos_x+size/2, pos_y+size/2, pos_z+size/2)
    glTexCoord2f(0.0, 1.0);    glVertex3f(pos_x-size/2, pos_y+size/2, pos_z+size/2)

    glTexCoord2f(1.0, 0.0);    glVertex3f(pos_x-size/2, pos_y-size/2, pos_z-size/2)
    glTexCoord2f(1.0, 1.0);    glVertex3f(pos_x-size/2, pos_y+size/2, pos_z-size/2)
    glTexCoord2f(0.0, 1.0);    glVertex3f(pos_x+size/2, pos_y+size/2, pos_z-size/2)
    glTexCoord2f(0.0, 0.0);    glVertex3f(pos_x+size/2, pos_y-size/2, pos_z-size/2)

    glTexCoord2f(0.0, 1.0);    glVertex3f(pos_x-size/2, pos_y+size/2, pos_z-size/2)
    glTexCoord2f(0.0, 0.0);    glVertex3f(pos_x-size/2, pos_y+size/2, pos_z+size/2)
    glTexCoord2f(1.0, 0.0);    glVertex3f(pos_x+size/2, pos_y+size/2, pos_z+size/2)
    glTexCoord2f(1.0, 1.0);    glVertex3f(pos_x+size/2, pos_y+size/2, pos_z-size/2)

    glTexCoord2f(1.0, 1.0);    glVertex3f(pos_x-size/2, pos_y-size/2, pos_z-size/2)
    glTexCoord2f(0.0, 1.0);    glVertex3f(pos_x+size/2, pos_y-size/2, pos_z-size/2)
    glTexCoord2f(0.0, 0.0);    glVertex3f(pos_x+size/2, pos_y-size/2, pos_z+size/2)
    glTexCoord2f(1.0, 0.0);    glVertex3f(pos_x-size/2, pos_y-size/2, pos_z+size/2)

    glTexCoord2f(1.0, 0.0);    glVertex3f(pos_x+size/2, pos_y-size/2, pos_z-size/2)
    glTexCoord2f(1.0, 1.0);    glVertex3f(pos_x+size/2, pos_y+size/2, pos_z-size/2)
    glTexCoord2f(0.0, 1.0);    glVertex3f(pos_x+size/2, pos_y+size/2, pos_z+size/2)
    glTexCoord2f(0.0, 0.0);    glVertex3f(pos_x+size/2, pos_y-size/2, pos_z+size/2)

    glTexCoord2f(0.0, 0.0);    glVertex3f(pos_x-size/2, pos_y-size/2, pos_z-size/2)
    glTexCoord2f(1.0, 0.0);    glVertex3f(pos_x-size/2, pos_y-size/2, pos_z+size/2)
    glTexCoord2f(1.0, 1.0);    glVertex3f(pos_x-size/2, pos_y+size/2, pos_z+size/2)
    glTexCoord2f(0.0, 1.0);    glVertex3f(pos_x-size/2, pos_y+size/2, pos_z-size/2)

    glEnd()
    return
# ----------------------------------------------------------------------------------------------------------------------
