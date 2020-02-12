import cv2
import cv2.aruco as aruco
from OpenGL.GL import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy

# ----------------------------------------------------------------------------------------------------------------------
def compose_GL_MAT(rotv,tvecs):

    rotv = rotv.reshape(3, 1)
    tvecs = tvecs.reshape(3, 1)

    rotMat, jacobian = cv2.Rodrigues(rotv)

    matrix = numpy.identity(4)
    matrix[0:3, 0:3] = rotMat
    matrix[0:3, 3:4] = tvecs
    newMat = numpy.identity(4)
    if True:
        newMat[1][1] = -1
        newMat[2][2] = -1
    matrix = numpy.dot(newMat, matrix)

    return matrix.T
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
def project_points(points_3d, rvec, tvec, camera_matrix, dist):
    #https: // docs.opencv.org / 2.4 / modules / calib3d / doc / camera_calibration_and_3d_reconstruction.html
    R, _ = cv2.Rodrigues(rvec)
    v = numpy.c_[R, tvec.T]
    P = numpy.dot(camera_matrix,v)

    points_2d = []

    for each in points_3d:
        point_4d=numpy.array([each[0],each[1],each[2],1])
        x = numpy.dot(P, point_4d)
        points_2d.append(x/x[2])

    points_2d = numpy.array(points_2d)[:,:2].reshape(-1,1,2)

    return points_2d,0
# ----------------------------------------------------------------------------------------------------------------------
def draw_axis(img, camera_matrix, dist, rvec, tvec, axis_length):
    # equivalent to aruco.drawAxis(frame,camera_matrix,dist,rvec, tvec, marker_length)

    axis_3d_end   = numpy.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, +axis_length]],dtype = numpy.float32)
    axis_3d_start = numpy.array([[0, 0, 0]],dtype=numpy.float32)

    axis_2d_end, jac = cv2.projectPoints(axis_3d_end, rvec, tvec, camera_matrix, dist)
    axis_2d_start, jac = cv2.projectPoints(axis_3d_start, rvec, tvec, camera_matrix, dist)

    axis_2d_end = axis_2d_end.reshape((3,2))
    axis_2d_start = axis_2d_start.reshape((1,2))

    #axis_2d_end, jac = project_points(axis_3d_end, rvec, tvec, camera_matrix, dist)
    #axis_2d_start, jac = project_points(axis_3d_start, rvec, tvec, camera_matrix, dist)


    img = tools_draw_numpy.draw_line(img, axis_2d_start[0, 1], axis_2d_start[0, 0], axis_2d_end[0, 1],axis_2d_end[0, 0], (0, 0, 255))
    img = tools_draw_numpy.draw_line(img, axis_2d_start[0, 1], axis_2d_start[0, 0], axis_2d_end[1, 1],axis_2d_end[1, 0], (0, 255, 0))
    img = tools_draw_numpy.draw_line(img, axis_2d_start[0, 1], axis_2d_start[0, 0], axis_2d_end[2, 1],axis_2d_end[2, 0], (255, 0, 0))
    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube_numpy(img, camera_matrix, dist, rvec, tvec, scale=(1,1,1),color=(255,128,0),):

    pooints_3d = 0.5*numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],[-1, -1, +1], [-1, +1, +1], [+1, +1, +1], [+1, -1, +1]],dtype = numpy.float32)

    pooints_3d[:,0]*=scale[0]
    pooints_3d[:,1]*=scale[1]
    pooints_3d[:,2]*=scale[2]

    points_2d, jac = cv2.projectPoints(pooints_3d, rvec, tvec, camera_matrix, dist)
    points_2d = points_2d.reshape((-1,2))
    for i,j in zip((0,1,2,3,4,5,6,7,0,1,2,3),(1,2,3,0,5,6,7,4,4,5,6,7)):
        img = tools_draw_numpy.draw_line(img, points_2d[i, 1], points_2d[i, 0], points_2d[j, 1],points_2d[j, 0], color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_mesh_numpy(pooints_3d, img, camera_matrix, dist, rvec, tvec, scale=(1,1,1),color=(66,0,166),):

    pooints_3d[:,0]*=scale[0]
    pooints_3d[:,1]*=scale[1]
    pooints_3d[:,2]*=scale[2]

    points_2d, jac = cv2.projectPoints(pooints_3d, rvec, tvec, camera_matrix, dist)
    points_2d = points_2d.reshape((-1,2))
    for point in points_2d:
        img = tools_draw_numpy.draw_circle(img, point[1], point[0], 4, color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def detect_marker_and_draw_axes(frame,marker_length,camera_matrix, dist):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), aruco.Dictionary_get(aruco.DICT_6X6_50))
    rvec, tvec = numpy.array([[[0,0,0]]]),numpy.array([[[0,0,0]]])


    if len(corners) > 0:
        res = frame.copy()
        aruco.drawDetectedMarkers(res, corners)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], marker_length, camera_matrix, dist)
        res= draw_axis(frame, camera_matrix, dist, rvec[0], tvec[0], marker_length / 2)
        #aruco.drawAxis(frame,camera_matrix,dist,rvec[0], tvec[0], marker_length / 2)

    return res, rvec, tvec
# ----------------------------------------------------------------------------------------------------------------------
def draw_point(point_3d):
    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_POINTS)
    glVertex3f(point_3d[0],point_3d[1],point_3d[2])
    glEnd()
    return
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
