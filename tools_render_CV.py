import math
import cv2
import numpy
import pyrr
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_wavefront
import tools_calibrate
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=2)
# ----------------------------------------------------------------------------------------------------------------------
def check_decompose_RT(rotv, tvecs):
    if rotv is None or tvecs is None:
        return

    modelview  = compose_model_view_from_RT(rotv, tvecs)
    mat_model, mat_view = decompose_model_view(modelview)
    modelview_new = pyrr.matrix44.multiply(mat_model,mat_view)
    rotv_check, tvecs_check = decompose_model_view_to_RT(modelview_new)
    return
# ----------------------------------------------------------------------------------------------------------------------
def check_decompose_model_view(mat_model, mat_view):

    modelview = pyrr.matrix44.multiply(mat_model, mat_view)
    rvec, tvec = decompose_model_view_to_RT(modelview)
    modelview_new = compose_model_view_from_RT(rvec,tvec)
    mat_model_check, mat_view_check = decompose_model_view(modelview)
    return
# ----------------------------------------------------------------------------------------------------------------------
def compose_model_view_from_RT(rotv, tvecs):

    rotv = rotv.reshape(3, 1)
    tvecs = tvecs.reshape(3, 1)

    #rotMat, jacobian = cv2.Rodrigues(rotv)
    rotMat = pyrr.matrix44.create_from_eulers(rotv.flatten())

    M = numpy.identity(4)
    M[0:3, 0:3] = rotMat[0:3,0:3]
    M[0:3, 3:4] = tvecs
    return M.T
# ----------------------------------------------------------------------------------------------------------------------
def decompose_model_view(M):
    S, Q, tvec_view = pyrr.matrix44.decompose(M)
    rvec_model = tools_calibrate.quaternion_to_euler(Q)
    mat_model = pyrr.matrix44.create_from_eulers(rvec_model)
    mat_view = pyrr.matrix44.create_from_translation(tvec_view)
    R = pyrr.matrix44.create_from_eulers((0,math.pi,math.pi))
    mat_view = pyrr.matrix44.multiply(mat_view, R)
    return mat_model, mat_view
# ----------------------------------------------------------------------------------------------------------------------
def decompose_model_view_to_RRTT(mat_model, mat_view):
    S, Q, tvec_model = pyrr.matrix44.decompose(mat_model)
    rvec_model = tools_calibrate.quaternion_to_euler(Q)

    S, Q, tvec_view = pyrr.matrix44.decompose(mat_view)
    rvec_view = tools_calibrate.quaternion_to_euler(Q)

    return rvec_model,tvec_model,rvec_view,tvec_view

# ----------------------------------------------------------------------------------------------------------------------
def decompose_model_view_to_RT(M):
    newM = M.T
    newM[1,:]*=-1
    newM[2,:]*=-1

    tvec = newM[0:3, 3:4]
    R = numpy.eye(4)
    R[0:3, 0:3] = newM[0:3, 0:3]
    S, Q, tvec_model = pyrr.matrix44.decompose(R)
    rvec = tools_calibrate.quaternion_to_euler(Q)
    return rvec.flatten(), tvec.flatten()
# ----------------------------------------------------------------------------------------------------------------------
def project_points(points_3d, rvec, tvec, camera_matrix, dist):
    #https: // docs.opencv.org / 2.4 / modules / calib3d / doc / camera_calibration_and_3d_reconstruction.html

    #R, _ = cv2.Rodrigues(rvec)
    R = pyrr.matrix44.create_from_eulers(rvec)

    M=numpy.zeros((4,4))
    M[:3,:3] = R[:3,:3]
    M[:3,3] = numpy.array(tvec).T

    P = numpy.zeros((3,4))
    P[:3,:3] = camera_matrix

    points_2d = []

    for each in points_3d:
        X = pyrr.matrix44.apply_to_vector(M, numpy.array([each[0],each[1],each[2],1]))
        uv = numpy.dot(P, X)
        points_2d.append(uv/uv[2])


    points_2d = numpy.array(points_2d)[:,:2].reshape(-1,1,2)

    return points_2d,0
# ----------------------------------------------------------------------------------------------------------------------
def project_points_ortho(points_3d, rvec, tvec, camera_matrix, dist,scale_factor):
    # R, _ = cv2.Rodrigues(rvec)
    R = pyrr.matrix44.create_from_eulers(rvec)

    M = numpy.eye(4)
    M[:3, :3] = R[:3, :3]
    M[:3, 3] = numpy.array(tvec).T

    P = numpy.zeros((3, 4))
    P[:3, :3] = camera_matrix
    P[:, 3] = P[:, 2]*scale_factor
    P[:, 2] = 0

    points_2d = []

    for each in points_3d:
        X = pyrr.matrix44.apply_to_vector(M, numpy.array([each[0], each[1], each[2], 1]))
        uv = numpy.dot(P, X)
        points_2d.append(uv / uv[2])

    points_2d = numpy.array(points_2d)[:, :2].reshape(-1, 1, 2)

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

    points_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],[-1, -1, +1], [-1, +1, +1], [+1, +1, +1], [+1, -1, +1]],dtype = numpy.float32)

    points_3d[:,0]*=scale[0]
    points_3d[:,1]*=scale[1]
    points_3d[:,2]*=scale[2]

    #points_2d, jac = cv2.projectPoints(pooints_3d, rvec, tvec, camera_matrix, dist)
    points_2d, jac = project_points(points_3d, rvec, tvec, camera_matrix, dist)

    points_2d = points_2d.reshape((-1,2))
    for i,j in zip((0,1,2,3,4,5,6,7,0,1,2,3),(1,2,3,0,5,6,7,4,4,5,6,7)):
        img = tools_draw_numpy.draw_line(img, points_2d[i, 1], points_2d[i, 0], points_2d[j, 1],points_2d[j, 0], color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube_numpy_MVP(img,mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166)):

    fx, fy = float(img.shape[1]), float(img.shape[0])
    camera_matrix = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])

    points_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],[-1, -1, +1], [-1, +1, +1], [+1, +1, +1], [+1, -1, +1]],dtype = numpy.float32)

    points_3d_new = []
    for v in points_3d:
        vv = pyrr.matrix44.apply_to_vector(mat_trns, v)
        vv = pyrr.matrix44.apply_to_vector(mat_model, vv)
        vv = pyrr.matrix44.apply_to_vector(mat_view, vv)
        points_3d_new.append(vv)


    points_2d, jac = project_points(points_3d_new, (0,0,0), (0,0,0), camera_matrix, numpy.zeros(4))
    points_2d = points_2d.reshape((-1,2))
    for i,j in zip((0,1,2,3,4,5,6,7,0,1,2,3),(1,2,3,0,5,6,7,4,4,5,6,7)):
        img = tools_draw_numpy.draw_line(img, points_2d[i, 1], points_2d[i, 0], points_2d[j, 1],points_2d[j, 0], color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_numpy_RT(points_3d, img, camera_matrix, dist, rvec, tvec,mat_trns=None, color=(66, 0, 166)):

    if mat_trns is None:
        mat_trns = numpy.eye(4)

    M = compose_model_view_from_RT(numpy.array(rvec, dtype=numpy.float), numpy.array(tvec, dtype=numpy.float))

    L3D = []
    for v in points_3d:
        vv = pyrr.matrix44.apply_to_vector(mat_trns, v)
        vv = pyrr.matrix44.apply_to_vector(M, vv)
        L3D.append(vv)

    L3D = numpy.array((L3D))

    #points_2d, jac = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist)
    points_2d, jac = project_points(L3D, (0,0,0), (0,0,0), camera_matrix, dist)

    points_2d = points_2d.reshape((-1, 2))
    for point in points_2d:
        img = tools_draw_numpy.draw_circle(img, int(point[1]), int(point[0]), 4, color)

    return img
# ----------------------------------------------------------------------------------------------------------------------

def draw_points_numpy_MVP(points_3d, img, mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166)):

    fx, fy = float(img.shape[1]), float(img.shape[0])
    camera_matrix = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])

    L3D = []
    for v in points_3d:
        vv = pyrr.matrix44.apply_to_vector(mat_trns ,v)
        vv = pyrr.matrix44.apply_to_vector(mat_model,vv)
        vv = pyrr.matrix44.apply_to_vector(mat_view ,vv)
        L3D.append(vv)

    L3D = numpy.array(L3D)

    #points_2d, jac = cv2.projectPoints(points_3d_new, (0,0,0), (0,0,0), camera_matrix, numpy.zeros(4))
    points_2d, jac = project_points(L3D, (0,0,0), (0,0,0), camera_matrix, numpy.zeros(4))

    points_2d = points_2d.reshape((-1,2))
    points_2d[:,0]=img.shape[1]-points_2d[:,0]

    for point in points_2d:
        img = tools_draw_numpy.draw_circle(img, int(point[1]), int(point[0]), 4, color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_numpy_MVP_ortho(points_3d, img, mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166),do_debug=False):

    fx, fy = float(img.shape[1]), float(img.shape[0])
    camera_matrix = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])

    L3D = []
    for v in points_3d:
        vv = pyrr.matrix44.apply_to_vector(mat_trns, v)
        vv = pyrr.matrix44.apply_to_vector(mat_model, vv)
        vv = pyrr.matrix44.apply_to_vector(mat_view, vv)
        L3D.append(vv)

    L3D = numpy.array(L3D)

    scale_factor = 10*(0.2/mat_projection[0,0])
    points_2d, jac = project_points_ortho(L3D, (0, 0, 0), (0, 0, 0), camera_matrix, numpy.zeros(4),scale_factor)

    L3D_ortho =L3D.copy()
    L3D_ortho[:,2]=1
    camera_matrix_ortho = camera_matrix.copy()
    camera_matrix_ortho[:,2]*=scale_factor

    points_2d, jac = cv2.projectPoints(L3D_ortho, (0, 0, 0), (0, 0, 0), camera_matrix_ortho, numpy.zeros(4),scale_factor)
    points_2d = points_2d.reshape((-1, 2))/scale_factor

    points_2d[:, 0] = img.shape[1] - points_2d[:, 0]

    for point in points_2d:
        img = tools_draw_numpy.draw_circle(img, int(point[1]), int(point[0]), 4, color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def get_ray(point_2d, img, mat_projection, mat_view, mat_model, mat_trns):

    fx, fy = float(img.shape[1]), float(img.shape[0])
    camera_matrix = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])

    Z1 = -1
    X1 = (point_2d[0]*Z1 - Z1*fx/2)/fx
    Y1 = (point_2d[1]*Z1 - Z1*fy/2)/fy
    ray_begin = numpy.array((X1, Y1, Z1))

    Z2 = +1
    X2 = (point_2d[0]*Z2 - Z2*fx/2)/fx
    Y2 = (point_2d[1]*Z2 - Z2*fy/2)/fy
    ray_end = numpy.array((X2, Y2, Z2))

    #check
    points_2d_check, jac = project_points(numpy.array([(X1, Y1, Z1),(X2,Y2,Z2)]), (0, 0, 0), (0, 0, 0), camera_matrix, numpy.zeros(4))

    i_mat_view  = pyrr.matrix44.inverse(mat_view)
    i_mat_model = pyrr.matrix44.inverse(mat_model)
    i_mat_trans = pyrr.matrix44.inverse(mat_trns)

    #i_mat_view  = mat_view.T
    #i_mat_model = mat_model.T
    #i_mat_trans = mat_trns.T

    ray_begin_v = pyrr.matrix44.apply_to_vector(i_mat_view , ray_begin)
    ray_begin_check_v = pyrr.matrix44.apply_to_vector(mat_view, ray_begin_v)

    ray_begin_m = pyrr.matrix44.apply_to_vector(i_mat_model, ray_begin_v)
    ray_begin_check_m = pyrr.matrix44.apply_to_vector(mat_model, ray_begin_m)

    ray_begin_t = pyrr.matrix44.apply_to_vector(i_mat_trans , ray_begin_m)
    ray_begin_check_t = pyrr.matrix44.apply_to_vector(mat_trns, ray_begin_t)

    #check
    vv = pyrr.matrix44.apply_to_vector(mat_trns, ray_begin_t)
    vv = pyrr.matrix44.apply_to_vector(mat_model, vv)
    vv = pyrr.matrix44.apply_to_vector(mat_view, vv)
    x = (vv == ray_begin)


    ray_end = pyrr.matrix44.apply_to_vector(i_mat_view , ray_end)
    ray_end = pyrr.matrix44.apply_to_vector(i_mat_model, ray_end)
    ray_end = pyrr.matrix44.apply_to_vector(i_mat_trans, ray_end)

    return ray_begin_t,ray_end
# ----------------------------------------------------------------------------------------------------------------------
def line_plane_intersection(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = numpy.array(planeNormal[:3]).dot(numpy.array(rayDirection[:3]))
    if numpy.isnan(ndotu) or abs(ndotu) < epsilon :
        return None

    w = numpy.array(rayPoint[:3]) - numpy.array(planePoint[:3])
    si = -numpy.array(planeNormal[:3]).dot(w) / ndotu
    Psi = w + si * numpy.array(rayDirection[:3]) + numpy.array(planePoint[:3])
    return Psi
# ----------------------------------------------------------------------------------------------------------------------
def normalize(x):
    n = numpy.sqrt((x ** 2).sum())
    if n>0:
        y = x/n
    else:
        y = x
    return y
# ----------------------------------------------------------------------------------------------------------------------
def get_normal(triangles_3d):
    A = triangles_3d[1] - triangles_3d[0]
    B = triangles_3d[2] - triangles_3d[0]
    Nx = A[1] * B[2] - A[2] * B[1]
    Ny = A[2] * B[0] - A[0] * B[2]
    Nz = A[0] * B[1] - A[1] * B[0]
    n = -numpy.array((Nx, Ny, Nz), dtype=numpy.float)
    n = normalize(n)
    return n
# ----------------------------------------------------------------------------------------------------------------------
def get_interception_ray_triangle(pos, direction, triangle):
    n = get_normal(triangle)
    collision = line_plane_intersection(n, triangle[0,:3], direction[:3], pos[:3], epsilon=1e-6)

    if collision is not None:
        if is_point_inside_triangle(collision,triangle):
            return collision

    return None
# ----------------------------------------------------------------------------------------------------------------------
def get_interception_ray_triangles(pos, direction, coord_vert, coord_norm, idxv, idxn):
    collisions = []

    for iv,inr in zip(idxv,idxn):
        triangle = coord_vert[iv]
        n = coord_norm[inr[0]]#n0 = get_normal(triangle)

        collision = line_plane_intersection(n, triangle[0,:3], direction[:3], pos[:3], epsilon=1e-6)

        if collision is not None:
            if is_point_inside_triangle(collision,triangle):
                collisions.append(collision)

    if len(collisions)==0:return None
    if len(collisions)==1:return collisions[0]

    X = numpy.array([collision-pos for collision in collisions])
    X = numpy.mean(X**2,axis=1)
    i = numpy.argmin(X)
    return collisions[i]
# ----------------------------------------------------------------------------------------------------------------------
def is_point_inside_triangle(contact_point, P):

    V1=P[0]
    V2=P[1]
    V3=P[2]

    line1 = normalize(V1-contact_point)
    line2 = normalize(V2-contact_point)
    dot1=numpy.dot(line1,line2)

    line1 = normalize(V2-contact_point)
    line2 = normalize(V3-contact_point)
    dot2=numpy.dot(line1,line2)

    line1 = normalize(V3-contact_point)
    line2 = normalize(V1-contact_point)
    dot3=numpy.dot(line1,line2)

    if numpy.isnan(dot1) or numpy.isnan(dot2) or numpy.isnan(dot3):
        return  False

    dot1 = min(+1,max(dot1,-1))
    dot2 = min(+1,max(dot2,-1))
    dot3 = min(+1,max(dot3,-1))

    accumilator = math.acos(dot1) + math.acos (dot2) + math.acos(dot3)
    if accumilator < (2*math.pi - 0.01):
        return False

    return True
# ----------------------------------------------------------------------------------------------------------------------
def align_two_model(filename_obj1,filename_markers1,filename_obj2,filename_markers2,filename_obj_res,filename_markers_res):
    object1 = tools_wavefront.ObjLoader()
    object1.load_mesh(filename_obj1, do_autoscale=False)

    object2 = tools_wavefront.ObjLoader()
    object2.load_mesh(filename_obj2, do_autoscale=False)

    markers1 = tools_IO.load_mat(filename_markers1, dtype=numpy.float, delim=',')
    markers2 = tools_IO.load_mat(filename_markers2, dtype=numpy.float, delim=',')

    result_markers = markers1.copy()
    result_vertex  = object1.coord_vert.copy()

    for dim in range(0,3):
        min_value_s = markers1[:, dim].min()
        min_value_t = markers2[:, dim].min()

        max_value_s = markers1[:, dim].max()
        max_value_t = markers2[:, dim].max()
        scale = (max_value_t - min_value_t) / (max_value_s - min_value_s)

        result_markers[:, dim]=(result_markers[:,dim]-min_value_s)*scale + min_value_t
        result_vertex[:,dim]  =(result_vertex[:,dim] -min_value_s)*scale + min_value_t


    tools_IO.save_mat(result_markers, filename_markers_res,delim=',')
    object1.export_mesh(filename_obj_res,X=result_vertex, idx_vertex=object1.idx_vertex)

    return
# ----------------------------------------------------------------------------------------------------------------------
def my_solve_PnP(L3D, L2D, K, dist):
    #return cv2.solvePnP(L3D, L2D, K, dist)

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    n = L3D.shape[0]


    #Step 1. Construct matrix A, whose size is 2n x12.
    A = numpy.zeros((2 * n, 12))
    for i in range(n):
        pt3d = L3D[i]
        pt2d = L2D[i]

        x = pt3d[0]
        y = pt3d[1]
        z = pt3d[2]
        u = pt2d[0]
        v = pt2d[1]

        A[2 * i, 0] = x * fx
        A[2 * i, 1] = y * fx
        A[2 * i, 2] = z * fx
        A[2 * i, 3] = fx
        A[2 * i, 4] = 0.0
        A[2 * i, 5] = 0.0
        A[2 * i, 6] = 0.0
        A[2 * i, 7] = 0.0
        A[2 * i, 8] = x * cx - u * x
        A[2 * i, 9] = y * cx - u * y
        A[2 * i, 10] = z * cx - u * z
        A[2 * i, 11] = cx - u
        A[2 * i + 1, 0] = 0.0
        A[2 * i + 1, 1] = 0.0
        A[2 * i + 1, 2] = 0.0
        A[2 * i + 1, 3] = 0.0
        A[2 * i + 1, 4] = x * fy
        A[2 * i + 1, 5] = y * fy
        A[2 * i + 1, 6] = z * fy
        A[2 * i + 1, 7] = fy
        A[2 * i + 1, 8] = x * cy - v * x
        A[2 * i + 1, 9] = y * cy - v * y
        A[2 * i + 1, 10] = z * cy - v * z
        A[2 * i + 1, 11] = cy - v

    #Step 2. Solve Ax = 0 by SVD
    u, s, vh = numpy.linalg.svd(A)

    a1 = vh[0, 11]
    a2 = vh[1, 11]
    a3 = vh[2, 11]
    a4 = vh[3, 11]
    a5 = vh[4, 11]
    a6 = vh[5, 11]
    a7 = vh[6, 11]
    a8 = vh[7, 11]
    a9 = vh[8, 11]
    a10= vh[9, 11]
    a11= vh[10,11]
    a12= vh[11,11]

    R_bar = numpy.array([[a1, a2, a3], [a5, a6, a7], [a9, a10, a11]])
    U_R, V_Sigma , V_R = numpy.linalg.svd(R_bar)
    R = numpy.dot(U_R , V_R.T)


    beta = 1.0 / ((V_Sigma[0] + V_Sigma[1] + V_Sigma[2]) / 3.0)

    t_bar = numpy.array((a4, a8, a12))
    t = beta * t_bar

    R = pyrr.matrix44.create_from_matrix33(R)

    S, Q, tvec_view = pyrr.matrix44.decompose(R)
    rvec = tools_calibrate.quaternion_to_euler(Q)


    return (0, rvec, t)
# ----------------------------------------------------------------------------------------------------------------------