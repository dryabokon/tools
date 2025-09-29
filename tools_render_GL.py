import math
import cv2
import numpy
import pyrr
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
from CV import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=2)
# ----------------------------------------------------------------------------------------------------------------------
# GL_style
# [R 0]
# [t 1]
# ----------------------------------------------------------------------------------------------------------------------
def construct_cube_one(scale=(1,1,1)):
    points_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],[-1, -1, +1], [-1, +1, +1], [+1, +1, +1], [+1, -1, +1]],dtype = numpy.float32)
    points_3d[:,0]*=scale[0]
    points_3d[:,1]*=scale[1]
    points_3d[:,2]*=scale[2]
    return points_3d
# ----------------------------------------------------------------------------------------------------------------------
def project_points_MVP_ortho(points_3d, W,H, mat_projection, mat_view, mat_model, mat_trns):

    scale_factor = (2 / mat_projection[0, 0],2 / mat_projection[1, 1])

    M = pyrr.matrix44.multiply(mat_view.T, pyrr.matrix44.multiply(mat_model.T, mat_trns.T))
    X4D = numpy.full((points_3d.shape[0], 4), 1, dtype=numpy.float)
    L3D = pyrr.matrix44.multiply(M, X4D.T).T[:, :3]

    points_2d, jac = tools_pr_geom.project_points_ortho(L3D, (0.0, 0.0, 0.0), (0, 0.0, 0.0), numpy.zeros(4),W,H, scale_factor)
    points_2d = points_2d.reshape((-1, 2))
    points_2d[:, 0] = W - points_2d[:, 0]

    return points_2d
# ----------------------------------------------------------------------------------------------------------------------
def project_points_MVP_GL(points_3d,W,H,mat_projection,mat_view,mat_model,mat_trns):


    M = numpy.dot(mat_view.T, numpy.dot(mat_model.T, mat_trns.T))

    X4D = numpy.concatenate((points_3d,numpy.ones((points_3d.shape[0],1))),axis=1)
    X3D = numpy.dot(M, X4D.T).T[:, :3]

    camera_matrix_3x3 = tools_pr_geom.compose_projection_mat_3x3(W, H, 1 / mat_projection[0][0],1 / mat_projection[1][1])

    method = 0
    if method == 0:  # opencv equivalent
        points_2d, jac = cv2.projectPoints(X3D, numpy.zeros(3), numpy.zeros(3),camera_matrix_3x3, numpy.zeros(5))
    else:
        points_2d, jac = tools_pr_geom.project_points(X3D, numpy.zeros(3), numpy.zeros(3),camera_matrix_3x3)

    points_2d = points_2d.reshape((-1, 2))
    points_2d[:, 0] = W - points_2d[:, 0]

    return points_2d
# ----------------------------------------------------------------------------------------------------------------------
def project_points_RT_GL(points_3d,M,camera_matrix_3x3,mat_trns):
    H, W = 2*camera_matrix_3x3[1,2],2*camera_matrix_3x3[0,2]
    tg_half_fovx = camera_matrix_3x3[0, 2] / camera_matrix_3x3[0, 0]
    mat_projection = tools_pr_geom.compose_projection_mat_4x4_GL(2 * camera_matrix_3x3[0, 2],2 * camera_matrix_3x3[1, 2], tg_half_fovx,tg_half_fovx)
    points_2d = project_points_MVP_GL(points_3d, W, H, mat_projection, M       , numpy.eye(4), mat_trns)
    return points_2d
# ----------------------------------------------------------------------------------------------------------------------
def project_points_rvec_tvec_GL(points_3d, rvec,tvec, camera_matrix_3x3,mat_trns):

    method = -1
    H, W = camera_matrix_3x3[1,2]*2,camera_matrix_3x3[0,2]*2

    if method==0:
        points_2d, jac = cv2.projectPoints(points_3d, numpy.array(rvec,dtype=float).reshape((3,1)), numpy.array(tvec,dtype=float).reshape((3,1)), camera_matrix_3x3, numpy.zeros(5))
    elif method==1:
        mat_view,mat_model = numpy.eye(4),numpy.eye(4)
        mat_view = compose_RT_mat_GL(rvec, tvec, do_rodriges=True, do_flip=True)
        M = pyrr.matrix44.multiply(mat_view.T, pyrr.matrix44.multiply(mat_model.T, mat_trns.T))
        X4D = numpy.hstack((points_3d, numpy.full((len(points_3d), 1), 1)))
        L3D = pyrr.matrix44.multiply(M, X4D.T).T[:, :3]
        points_2d, jac = cv2.projectPoints(L3D, numpy.array((0.0, 0.0, 0.0)), numpy.array((0.0, 0.0, 0.0)),camera_matrix_3x3, numpy.zeros(4, dtype=float))
        points_2d = points_2d.reshape((-1, 2))
        points_2d[:, 0] = W - points_2d[:, 0]
    elif method == 2:
        tg_half_fovx = camera_matrix_3x3[0, 2] / camera_matrix_3x3[0, 0]
        M = compose_RT_mat_GL(rvec, tvec, do_rodriges=True, do_flip=True)
        mat_projection = tools_pr_geom.compose_projection_mat_4x4_GL(2 * camera_matrix_3x3[0, 2],2 * camera_matrix_3x3[1, 2], tg_half_fovx,tg_half_fovx)
        points_2d = project_points_MVP_GL(points_3d, W, H, mat_projection, M, numpy.eye(4), mat_trns)
    else:
        M = compose_RT_mat_GL(rvec, tvec, do_rodriges=True, do_flip=True)
        points_2d = project_points_RT_GL(points_3d, M, camera_matrix_3x3, mat_trns)

    return points_2d.reshape((-1,2)).astype(int)
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_MVP_GL(points_3d, img, mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166), w = 6,transperency=0.0,do_debug=False):

    H,W = img.shape[:2]
    points_2d = project_points_MVP_GL(points_3d,W,H,mat_projection,mat_view,mat_model,mat_trns)
    img = tools_draw_numpy.draw_points(img, points_2d,color=color,w=w,transperency=transperency)

    if do_debug:
        posx,posy = img.shape[1]-250,0
        img = tools_draw_numpy.draw_mat(mat_trns,  posx, posy+20, img,(0,128,255))
        img = tools_draw_numpy.draw_mat(mat_model, posx, posy+120, img,(0,128,255))
        img = tools_draw_numpy.draw_mat(mat_view,  posx, posy+220, img,(0,128,255))
        img = tools_draw_numpy.draw_mat(mat_projection, posx, posy+320, img,(0,128,255))
        img = tools_draw_numpy.draw_mat(mat_projection, posx, posy+320, img,(0,128,255))

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_rvec_tvec_GL(points_3d, img, rvec,tvec, camera_matrix_3x3,mat_trns,color=(66, 0, 166), w=6):
    points_2d = project_points_rvec_tvec_GL(points_3d, rvec,tvec, camera_matrix_3x3,mat_trns)
    img = tools_draw_numpy.draw_points(img, points_2d, color=color, w=w)
    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_RT_GL(points_3d, img, M, camera_matrix_3x3,mat_trns=numpy.eye(4),color=(66, 0, 166), w=6,transperency=0.0):
    tg_half_fovx = camera_matrix_3x3[0, 2] / camera_matrix_3x3[0, 0]
    mat_projection = tools_pr_geom.compose_projection_mat_4x4_GL(2*camera_matrix_3x3[0, 2], 2*camera_matrix_3x3[1, 2], tg_half_fovx, tg_half_fovx)
    image = draw_points_MVP_GL(points_3d, img, mat_projection, M, numpy.eye(4), mat_trns, color=color, w=w,transperency=transperency)
    return image
# ----------------------------------------------------------------------------------------------------------------------
def draw_lines_MVP_GL(lines_3d, img, mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166), w=6):
    points_2d_start = tools_pr_geom.project_points_MVP(lines_3d[:, :3], img, mat_projection, mat_view, mat_model, mat_trns)
    points_2d_end   = tools_pr_geom.project_points_MVP(lines_3d[:, 3:], img, mat_projection, mat_view, mat_model, mat_trns)
    lines_2d =[(int(point_start[0]), int(point_start[1]), int(point_end[0]), int(point_end[1])) for point_start, point_end in zip(points_2d_start, points_2d_end)]
    img = tools_draw_numpy.draw_lines(img, lines_2d, color=color, w=w)
    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube_MVP_GL(points_3d, img, mat_projection, mat_view, mat_model, mat_trns, color=(255, 128, 0), w=2,idx_mode=1):

    if points_3d is None:
        points_3d = construct_cube_one()
    lines_idx = tools_draw_numpy.cuboid_lines_idx1 if idx_mode == 1 else tools_draw_numpy.cuboid_lines_idx2
    lines_3d = numpy.array([(points_3d[i, 0], points_3d[i, 1],points_3d[i, 2], points_3d[j, 0], points_3d[j, 1],points_3d[j, 2]) for (i,j) in lines_idx])
    img = draw_points_MVP_GL    (points_3d, img, mat_projection, mat_view, mat_model, mat_trns,color)
    img = draw_lines_MVP_GL(lines_3d, img, mat_projection, mat_view, mat_model, mat_trns, color, w)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube_rvec_tvec_GL(points_3d, img, rvec, tvec, camera_matrix_3x3, mat_trns=numpy.eye(4), dist=numpy.zeros(5), scale=(1, 1, 1), color=(255, 128, 0), w=6):

    if points_3d is None:
        points_3d = construct_cube_one(scale)

    points_2d = project_points_rvec_tvec_GL(points_3d, rvec,tvec, camera_matrix_3x3,mat_trns)
    lines_2d = numpy.array([(points_2d[i, 0], points_2d[i, 1], points_2d[j, 0], points_2d[j, 1]) for (i,j) in tools_draw_numpy.cuboid_lines_idx2])
    img = tools_draw_numpy.draw_lines(img, lines_2d, color=color, w=1)
    img = tools_draw_numpy.draw_points(img, points_2d,color=color,w=w)
    return img
# ----------------------------------------------------------------------------------------------------------------------
def compose_RT_mat_GL(rvec, tvec, do_rodriges=False, do_flip=True):
    R = pyrr.matrix44.create_from_eulers(rvec).T

    if do_rodriges:
        R = pyrr.matrix44.create_from_matrix33(cv2.Rodrigues(numpy.array(rvec).astype(numpy.float32))[0])

    T = pyrr.matrix44.create_from_translation(numpy.array(tvec)).T
    M = pyrr.matrix44.multiply(T, R)
    M = M.T

    if do_flip:
        flip = numpy.identity(4)
        flip[1][1] = -1
        flip[2][2] = -1
        M = numpy.dot(M, flip)

    return M
# ----------------------------------------------------------------------------------------------------------------------
def decompose_to_rvec_tvec_GL(M):

    M_copy = M.copy()
    S, Q, tvec_check = pyrr.matrix44.decompose(M_copy)
    tvec_check[2]*=-1
    tvec_check[1]*=-1

    rvec_check = tools_pr_geom.quaternion_to_euler(Q)
    rvec_check = rvec_check[[0, 2, 1]]

    rvec_check[1]*= -1
    rvec_check[2]*= -1

    if rvec_check[0]>0:
        rvec_check -= (math.pi, 0, 0)
    else:
        rvec_check += (math.pi, 0, 0)

    rvec_check[0] *= -1

    return rvec_check,tvec_check
# ----------------------------------------------------------------------------------------------------------------------
def rotate_view(M,delta_angle):
    RR = pyrr.matrix44.multiply(pyrr.matrix44.multiply(M, pyrr.matrix44.create_from_eulers(delta_angle)), pyrr.matrix44.inverse(M))
    M_new = pyrr.matrix44.multiply(RR, M)
    # CV style
    # RR = pyrr.matrix44.multiply(pyrr.matrix44.inverse(M), pyrr.matrix44.multiply(R, M))
    # M_new = pyrr.matrix44.multiply(M, RR)
    return M_new
# ----------------------------------------------------------------------------------------------------------------------
def define_cam_position(W,H,cam_fov_deg = 11,cam_offset_dist = 64,cam_height = 6.5,cam_shift = 0.0):

    rvec, tvec  = [0,0.0,0], [cam_shift, cam_height, cam_offset_dist]
    tg_half_fovx = numpy.tan(cam_fov_deg * numpy.pi / 360)
    camera_matrix_3x3 = tools_pr_geom.compose_projection_mat_3x3(W, H, tg_half_fovx, tg_half_fovx)

    RT_GL = compose_RT_mat_GL(rvec, tvec, do_rodriges=True, do_flip=True)
    a_pitch = -numpy.arctan(cam_height / (cam_offset_dist + 1e-6))
    RT_GL = rotate_view(RT_GL,(a_pitch,0,0))
    rvec,tvec = decompose_to_rvec_tvec_GL(RT_GL)

    return camera_matrix_3x3,numpy.array(rvec).astype(numpy.float32), numpy.array(tvec).astype(numpy.float32), RT_GL
# ----------------------------------------------------------------------------------------------------------------------
