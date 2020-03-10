import math
import cv2
import numpy
import pyrr

# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_wavefront
import tools_IO
import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=2)
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

    #points_2d, jac = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist)
    points_2d, jac = tools_pr_geom.project_points(points_3d, rvec, tvec, camera_matrix, dist)

    points_2d = points_2d.reshape((-1,2))
    for i,j in zip((0,1,2,3,4,5,6,7,0,1,2,3),(1,2,3,0,5,6,7,4,4,5,6,7)):
        img = tools_draw_numpy.draw_line(img, points_2d[i, 1], points_2d[i, 0], points_2d[j, 1],points_2d[j, 0], color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube_numpy_MVP(img,mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166)):

    fx, fy = float(img.shape[1]), float(img.shape[0])
    camera_matrix = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])

    points_3d = 0.01*numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],[-1, -1, +1], [-1, +1, +1], [+1, +1, +1], [+1, -1, +1]],dtype = numpy.float32)

    points_3d_new = []
    for v in points_3d:
        vv = pyrr.matrix44.apply_to_vector(mat_trns, v)
        vv = pyrr.matrix44.apply_to_vector(mat_model, vv)
        vv = pyrr.matrix44.apply_to_vector(mat_view, vv)
        points_3d_new.append(vv)


    points_2d, jac = tools_pr_geom.project_points(points_3d_new, (0,0,0), (0,0,0), camera_matrix, numpy.zeros(4))
    points_2d = points_2d.reshape((-1,2))
    for i,j in zip((0,1,2,3,4,5,6,7,0,1,2,3),(1,2,3,0,5,6,7,4,4,5,6,7)):
        img = tools_draw_numpy.draw_line(img, points_2d[i, 1], points_2d[i, 0], points_2d[j, 1],points_2d[j, 0], color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_numpy_MVP(points_3d, img, mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166)):

    fx, fy = float(img.shape[1]), float(img.shape[0])
    camera_matrix = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])
    M = pyrr.matrix44.multiply(mat_view.T,pyrr.matrix44.multiply(mat_model.T,mat_trns.T))
    X4D = numpy.full((points_3d.shape[0], 4), 1,dtype=numpy.float)
    X4D[:, :3] = points_3d[:,:]
    L3D = pyrr.matrix44.multiply(M, X4D.T).T[:,:3]

    #points_2d, jac = cv2.projectPoints(points_3d_new, (0,0,0), (0,0,0), camera_matrix, numpy.zeros(4))
    points_2d, jac = tools_pr_geom.project_points(L3D, (0,0,0), (0,0,0), camera_matrix, numpy.zeros(4))

    points_2d = points_2d.reshape((-1,2))
    points_2d[:,0]=img.shape[1]-points_2d[:,0]

    for point in points_2d:
        img = tools_draw_numpy.draw_circle(img, int(point[1]), int(point[0]), 4, color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_numpy_MVP_ortho(points_3d, img, mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166),do_debug=False):

    fx, fy = float(img.shape[1]), float(img.shape[0])
    scale_factor = (2 / mat_projection[0, 0],2 / mat_projection[1, 1])

    M = pyrr.matrix44.multiply(mat_view.T, pyrr.matrix44.multiply(mat_model.T, mat_trns.T))
    X4D = numpy.full((points_3d.shape[0], 4), 1, dtype=numpy.float)
    X4D[:, :3] = points_3d
    X4D = pyrr.matrix44.multiply(M,X4D.T).T

    points_2d, jac = tools_pr_geom.project_points_ortho(X4D[:,:3], (0, 0, 0), (0, 0, 0), numpy.zeros(4),fx,fy, scale_factor)
    points_2d = points_2d.reshape((-1, 2))
    points_2d[:, 1] = img.shape[0] - points_2d[:, 1]
    for point in points_2d:
        img = tools_draw_numpy.draw_circle(img, int(point[1]), int(point[0]), 4, color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def check_projection_ortho(L3D, points_2d, rvec, tvec, fx,fy,scale_factor,do_debug=False):

    L3D_ortho = L3D.copy()

    points_2d_check, _ = tools_pr_geom.project_points_ortho(L3D_ortho, rvec.flatten(), tvec.flatten(), numpy.zeros(4),fx,fy, scale_factor)
    points_2d_check = points_2d_check.reshape((-1, 2))
    loss = math.sqrt(((points_2d_check - points_2d) ** 2).mean())
    if do_debug:
        print('ortho loss = ', loss)
        image = numpy.full((fy, fx, 3), 255)
        for x in points_2d: cv2.circle(image, (int(x[0]), int(x[1])), 4, (0, 128, 255), -1)
        for x in points_2d_check:cv2.circle(image, (int(x[0]), int(x[1])), 4, (0, 32, 190), -1)
        cv2.imwrite('./images/output/fit_check_RT.png', image)
    return loss
# ----------------------------------------------------------------------------------------------------------------------
def check_projection_ortho_modelview(L3D, points_2d, modelview, fx,fy,scale_factor, do_debug=False):

    points_2d_check, _ = tools_pr_geom.project_points_ortho_modelview(L3D, modelview, numpy.zeros(4), fx,fy,scale_factor)
    points_2d_check = points_2d_check.reshape((-1, 2))
    loss = math.sqrt(((points_2d_check - points_2d) ** 2).mean())
    if do_debug:
        print('ortho loss = ', loss)
        image = numpy.full(( fy, fx, 3), 255)
        for x in points_2d: cv2.circle(image, (int(x[0]), int(x[1])), 4, (0, 128, 255), -1)
        for x in points_2d_check:cv2.circle(image, (int(x[0]), int(x[1])), 4, (0, 32, 190), -1)
        cv2.imwrite('./images/output/fit_check_modelview.png', image)
    return loss
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
    points_2d_check, jac = tools_pr_geom.project_points(numpy.array([(X1, Y1, Z1),(X2,Y2,Z2)]), (0, 0, 0), (0, 0, 0), camera_matrix, numpy.zeros(4))

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
def fit_rvec_tvec(modelview):
    MT, MR, MK = tools_pr_geom.decompose_into_TRK(modelview)
    #tvec = MT[3,:3]
    tvec = MT[:3 ,3]
    rvec = tools_pr_geom.rotationMatrixToEulerAngles(MR[:3, :3])



    #check
    R = pyrr.matrix44.create_from_eulers(rvec).T
    T = pyrr.matrix44.create_from_translation(tvec).T
    modelview_check = pyrr.matrix44.multiply(T, R)

    return rvec, tvec
# ----------------------------------------------------------------------------------------------------------------------
def fit_scale_factor(X4D, X_target, a0, a1, a2, fx, fy,do_debug=False):



    R = pyrr.matrix44.create_from_eulers((a0,a2,a1)).T
    P = tools_pr_geom.compose_projection_mat(fx,fy,(7,7))
    PR = pyrr.matrix44.multiply(P, R)

    X_projection = pyrr.matrix44.multiply(PR, X4D.T).T[:, :2]
    E, result_E = tools_pr_geom.fit_affine(X_projection, X_target)
    M = numpy.eye(4)
    M[:2, :2] = E[:, :2]
    M[:2,  3] = E[:,  2]

    MPR = pyrr.matrix44.multiply(M, PR)

    # check
    X_target_check = pyrr.matrix44.multiply(MPR, X4D.T).T
    if do_debug:
        image = numpy.full((int(fy), int(fx), 3), 20)
        for x in X_target       : cv2.circle(image, (int(x[0]), int(x[1])), 16, (0, 128, 255), 4)
        for x in result_E       : cv2.circle(image, (int(x[0]), int(x[1])), 4, (0, 32, 190), -1)
        #for x in X_projection  : cv2.circle(image, (int(x[0]), int(x[1])), 3, (128, 128, 128), -1)

        cv2.imwrite('./data/output/fit_%03d_%03d.png'%(a0*180/math.pi,a1*180/math.pi), image)


    MT, MR, MK = tools_pr_geom.decompose_into_TRK(MPR)
    scale_factor = (fx/MK[0, 0], fy/MK[1, 1])


    newP = tools_pr_geom.compose_projection_mat(fx, fy, scale_factor)
    newPR = pyrr.matrix44.multiply(newP, MR.T)

    X_target_4D = numpy.full((X_target.shape[0],4),1,dtype=numpy.float)
    X_target_4D[:,:2] = X_target[:,:2]
    X_S = pyrr.matrix44.multiply(newPR, X4D.T).T

    T, X_target_check2 = tools_pr_geom.fit_translation(X_S,X_target_4D)
    T[ 2,3]=0
    T[:3,3]/=((MK[0, 0] + MK[1, 1])/2)

    TMR = pyrr.matrix44.multiply(T, MR.T)

    #check
    X_target_check3 = pyrr.matrix44.multiply(pyrr.matrix44.multiply(newP,TMR), X4D.T).T

    return scale_factor, TMR, X_target_check2
# ----------------------------------------------------------------------------------------------------------------------
def find_ortho_fit(X3D, X_target, fx, fy,do_debug = False):

    n_steps = 20
    min_loss = -1
    best_rvec, best_tvec, best_scale,best_modelview = None,None, None, None
    best_result = []
    a2 = 0

    X4D = numpy.full((X3D.shape[0],4),1,dtype=numpy.float)
    X4D[:,:3] = X3D[:,:3]

    rotation_range = numpy.arange(-math.pi / 4, math.pi / 4, math.pi / n_steps)
    rotation_range[0]=0

    #iterate thru 2 angles
    for a0 in rotation_range:
        for a1 in rotation_range:

            scale_factor, TR, result = fit_scale_factor(X4D, X_target, a0, a1, a2, fx, fy,do_debug)
            loss = ((result[:,:2]-X_target)**2).mean()

            if min_loss<0 or loss<min_loss:
                min_loss = loss
                best_result = result
                best_scale = scale_factor
                best_modelview = TR
                best_rvec, best_tvec = fit_rvec_tvec(TR)

    if do_debug:
        image = numpy.full((int(fy), int(fx), 3), 255)
        for x in X_target   : cv2.circle(image, (int(x[0]), int(x[1])), 4, (0, 128, 255), -1)
        for x in best_result: cv2.circle(image, (int(x[0]), int(x[1])), 4, (0, 32, 190), -1)
        cv2.imwrite('./images/output/fit.png', image)

    return best_rvec, best_tvec, best_scale, best_modelview
# ----------------------------------------------------------------------------------------------------------------------