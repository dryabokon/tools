import math
import cv2
import numpy
import pyrr
from sympy.geometry import intersection, Point, Line
from scipy.spatial import distance as dist
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
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
def draw_mat(M, posx, posy, image,color=(128, 128, 0)):
    for row in range(M.shape[0]):
        if M.shape[1]==4:
            string1 = '%+1.2f %+1.2f %+1.2f %+1.2f' % (M[row, 0], M[row, 1], M[row, 2], M[row, 3])
        else:
            string1 = '%+1.2f %+1.2f %+1.2f' % (M[row, 0], M[row, 1], M[row, 2])
        image = cv2.putText(image, '{0}'.format(string1), (posx, posy + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color, 1, cv2.LINE_AA)
    return image
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube_numpy(img, camera_matrix, dist, rvec, tvec, scale=(1,1,1),color=(255,128,0)):

    points_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],
                             [-1, -1, +0], [-1, +1, +0], [+1, +1, +0], [+1, -1, +0]],dtype = numpy.float32)
    colors = tools_IO.get_colors(8)

    points_3d[:,0]*=scale[0]
    points_3d[:,1]*=scale[1]
    points_3d[:,2]*=scale[2]

    if len(rvec.shape)==2 and rvec.shape[0]==3 and rvec.shape[1]==3:
        method = 'xx'
    else:
        method = 'cv'

    if method=='cv':
        points_2d, jac = cv2.projectPoints(points_3d, numpy.array(rvec,dtype=float).reshape((3,1)), numpy.array(tvec,dtype=float).reshape((3,1)), camera_matrix, dist)
    else:
        points_2d, jac = tools_pr_geom.project_points(points_3d, rvec, tvec, camera_matrix, dist)

    result = img.copy()
    points_2d = points_2d.reshape((-1,2)).astype(int)
    for i,j in zip((0,1,2,3,4,5,6,7,0,1,2,3),(1,2,3,0,5,6,7,4,4,5,6,7)):
        cv2.line(result,(points_2d[i, 0], points_2d[i, 1]),(points_2d[j, 0],points_2d[j, 1]),color,thickness=2)

    for i in (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3):
        cv2.circle(result,(points_2d[i, 0], points_2d[i, 1]),5,colors[i].tolist(),thickness=-1)

    clr = (255, 255, 255)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3):
        cv2.putText(result, '{0}   {1} {2} {3}'.format(i, points_3d[i,0],points_3d[i,1],points_3d[i,2]), (points_2d[i, 0], points_2d[i, 1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)

    return result
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


    points_2d, jac = tools_pr_geom.project_points(numpy.array(points_3d_new), numpy.array((0,0,0)), numpy.array((0,0,0)), camera_matrix, numpy.zeros(4))
    points_2d = points_2d.reshape((-1,2))
    for i,j in zip((0,1,2,3,4,5,6,7,0,1,2,3),(1,2,3,0,5,6,7,4,4,5,6,7)):
        img = tools_draw_numpy.draw_line(img, points_2d[i, 1], points_2d[i, 0], points_2d[j, 1],points_2d[j, 0], color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_numpy_MVP(points_3d, img, mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166),do_debug=False):

    aperture = 0.5 * (1 - mat_projection[2][0])
    camera_matrix_3x3 = tools_pr_geom.compose_projection_mat_3x3(img.shape[1], img.shape[0], aperture,aperture)

    M = pyrr.matrix44.multiply(mat_view.T,pyrr.matrix44.multiply(mat_model.T,mat_trns.T))
    X4D = numpy.full((points_3d.shape[0], 4), 1,dtype=numpy.float)
    X4D[:, :3] = points_3d[:,:]
    L3D = pyrr.matrix44.multiply(M, X4D.T).T[:,:3]

    method=1
    if method==0:
        img = draw_points_numpy_RT(L3D,img,numpy.eye(4),camera_matrix_3x3,color,flipX=False)

    elif method==1:
        #opencv equivalent
        #points_2d, jac = cv2.projectPoints(           L3D, numpy.array((0, 0, 0)), numpy.array((0, 0, 0)), camera_matrix_3x3, numpy.zeros(4,dtype=float))
        points_2d, jac = tools_pr_geom.project_points(L3D, numpy.array((0, 0, 0)), numpy.array((0, 0, 0)), camera_matrix_3x3, numpy.zeros(4))
        points_2d = points_2d.reshape((-1,2))
        points_2d[:, 0] = img.shape[1] - points_2d[:, 0]
        for point in points_2d:img = tools_draw_numpy.draw_circle(img, int(point[1]), int(point[0]), 2, color)

    if do_debug:
        posx,posy = img.shape[1]-250,0
        draw_mat(mat_trns,  posx, posy+20, img,(0,128,255))
        draw_mat(mat_model, posx, posy+120, img,(0,128,255))
        draw_mat(mat_view,  posx, posy+220, img,(0,128,255))
        draw_mat(mat_projection, posx, posy+320, img,(0,128,255))

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_numpy_RT(points_3d,img,RT,camera_matrix_3x3,color=(66, 0, 166),flipX=False):
    points_2d = tools_pr_geom.project_points_M(points_3d, RT, camera_matrix_3x3, numpy.zeros(5)).reshape((-1, 2))
    points_2d[:,0] = img.shape[1]-points_2d[:,0]
    if flipX:points_2d[:,0] = img.shape[1]-points_2d[:,0]

    for point in points_2d:
        if numpy.any(numpy.isnan(point)): continue
        img = tools_draw_numpy.draw_circle(img, int(point[1]), int(point[0]), 2, color)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_lines_numpy_RT(lines_3d,img,RT,camera_matrix_3x3,color=(66, 0, 166),w=6,flipX=False):
    points_2d_start = tools_pr_geom.project_points_M(lines_3d[:,:3], RT, camera_matrix_3x3, numpy.zeros(5)).reshape((-1, 2))
    if flipX:points_2d_start[:,0] = img.shape[1]-points_2d_start[:,0]

    points_2d_end = tools_pr_geom.project_points_M(lines_3d[:, 3:], RT, camera_matrix_3x3, numpy.zeros(5)).reshape((-1, 2))
    if flipX:points_2d_end[:, 0] = img.shape[1] - points_2d_end[:, 0]

    for point_start, point_end in zip(points_2d_start, points_2d_end):
        cv2.line(img, (int(point_start[0]), int(point_start[1])), (int(point_end[0]), int(point_end[1])), color,thickness=w)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_ellipse_numpy_RT(points_3d,img,RT,camera_matrix_3x3,color=(66, 0, 166),w=6,flipX=False):

    points_2d = tools_pr_geom.project_points_M(points_3d, RT, camera_matrix_3x3, numpy.zeros(5)).reshape((-1, 2))
    if flipX:points_2d[:,0] = img.shape[1]-points_2d[:,0]
    ellipse = cv2.fitEllipse(numpy.array(points_2d,dtype=numpy.float32))
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
    rotation_angle = ellipse[2]
    cv2.ellipse(img, center, axes, rotation_angle, startAngle=0, endAngle=360, color=color, thickness=w)

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
        img = tools_draw_numpy.draw_circle(img, int(point[1]), int(point[0]), 2, color)

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
def line_intersection_sympi_slow(line1, line2):
    res = intersection(Line(Point(line1[0], line1[1]), Point(line1[2], line1[3])), Line(Point(line2[0], line2[1]), Point(line2[2], line2[3])))
    return int(res[0].x), int(res[0].y)
# ----------------------------------------------------------------------------------------------------------------------
def line_intersection_unstable(line1, line2):

    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    L1 = line([line1[0], line1[1]], [line1[2], line1[3]])
    L2 = line([line2[0], line2[1]], [line2[2], line2[3]])

    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x), int(y)
    else:
        return numpy.nan, numpy.nan
# ----------------------------------------------------------------------------------------------------------------------
def line_intersection_very_unstable(line1, line2):
    Ax1, Ay1, Ax2, Ay2 = line1
    Bx1, By1, Bx2, By2 = line2

    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return numpy.nan, numpy.nan
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return numpy.nan, numpy.nan
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)

    return int(x), int(y)
# ----------------------------------------------------------------------------------------------------------------------
def line_intersection(l1, l2):

    p0, p1, d = distance_between_lines(l1,l2,clampAll=False)
    if p0 is not None:
        return numpy.array((p0[0],p0[1]))
    else:
        return numpy.array((numpy.nan, numpy.nan))
# ---------------------------------------------------------------------------------------------------------------------
def do_lines_intersect(line1, line2):
    ax1, ay1, ax2, ay2 = line1
    bx1, by1, bx2, by2 = line2

    v1 = (bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)
    v2 = (bx2 - bx1) * (ay2 - by1) - (by2 - by1) * (ax2 - bx1)
    v3 = (ax2 - ax1) * (by1 - ay1) - (ay2 - ay1) * (bx1 - ax1)
    v4 = (ax2 - ax1) * (by2 - ay1) - (ay2 - ay1) * (bx2 - ax1)
    if (numpy.sign(v1)!=numpy.sign(v2)) and(numpy.sign(v3)!=numpy.sign(v4)):
        return True
    else:
        return False
#----------------------------------------------------------------------------------------------------------------------
def is_point_above_line(point,line,tol=0.01):
    res = numpy.cross(point - line[:2], point - line[2:]) < tol
    return res
# ----------------------------------------------------------------------------------------------------------------------
def reflect_point_line(point,line):
    #https://math.stackexchange.com/questions/1013230/how-to-find-coordinates-of-reflected-point

    p,q=point[0],point[1]

    # 1*y = Ax + B
    A =(line[1] - line[3]) / (line[0] - line[2])
    B = line[1] - A * line[0]

    # y = -b/a * x - c/a
    a=-1
    c=B
    b=A

    x= (p*(a**2 -b**2)-2*b*(a*q+c))/(a**2+b**2)
    y= (q*(b**2 -a**2)-2*a*(b*p+c))/(a**2+b**2)

    return x, y
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
    P = tools_pr_geom.compose_projection_ortho_mat(fx, fy, (7, 7))
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


    newP = tools_pr_geom.compose_projection_ortho_mat(fx, fy, scale_factor)
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
def distance_between_lines(line1, line2, clampAll=False, clampA0=False, clampA1=False, clampB0=False, clampB1=False):

    if line1 is None or numpy.any(numpy.isnan(line1)) or line2 is None or numpy.any(numpy.isnan(line2)):
        return (numpy.nan,numpy.nan),(numpy.nan,numpy.nan),numpy.nan

    if numpy.linalg.norm(numpy.array(line1)-numpy.array(line2))<=0.001:
        return line1[:2],line2[:2],0

    #Return the closest points on each segment and their distance

    if len(line1)==4:
        a0, a1 = numpy.hstack((line1[:2],0)), numpy.hstack((line1[2:],0))
        b0, b1 = numpy.hstack((line2[:2],0)), numpy.hstack((line2[2:],0))
    else:
        a0, a1 = line1[:3], line1[3:]
        b0, b1 = line2[:3], line2[3:]

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = numpy.linalg.norm(A)
    magB = numpy.linalg.norm(B)

    if magA <=0.0001:
        _A = A.copy()
    else:
        _A = A / magA

    if magB <= 0.0001:
        _B = B.copy()
    else:
        _B = B / magB

    cross = numpy.cross(_A, _B)
    denom = numpy.linalg.norm(cross) ** 2


    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = numpy.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = numpy.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if numpy.absolute(d0) < numpy.absolute(d1):
                        return a0, b0, numpy.linalg.norm(a0 - b0)
                    return a0, b1, numpy.linalg.norm(a0 - b1)


            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if numpy.absolute(d0) < numpy.absolute(d1):
                        return a1, b0, numpy.linalg.norm(a1 - b0)
                    return a1, b1, numpy.linalg.norm(a1 - b1)


        # Segments overlap, return distance between parallel segments
        return None, None, numpy.linalg.norm(((d0 * _A) + a0) - b0)



    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = numpy.linalg.det([t, _B, cross])
    detB = numpy.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = numpy.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = numpy.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)


    return pA, pB, numpy.linalg.norm(pA - pB)
# ----------------------------------------------------------------------------------------------------------------------
def distance_point_to_line(line, point):
    p1 = numpy.array(line)[:2]
    p2 = numpy.array(line)[2:]

    norm = numpy.linalg.norm(p2 - p1)
    if norm<=0.0001:
        return 0
    else:
        d = numpy.cross(p2 - p1, point - p1) / norm
    return numpy.abs(d)
# ----------------------------------------------------------------------------------------------------------------------
def perp_point_line(line,point):

    x1,y1,x2,y2 = line
    x3,y3 = point[0],point[1]

    px = x2 - x1
    py = y2 - y1
    dAB = px * px + py * py

    u = ((x3 - x1) * px + (y3 - y1) * py) / dAB
    x = x1 + u * px
    y = y1 + u * py

    return x,y
# ----------------------------------------------------------------------------------------------------------------------
def distance_extract_to_line(extract, line, do_debug=False):
    point1 = extract[:2]
    point2 = extract[2:]
    h1 = distance_point_to_line(line, point1)
    h2 = distance_point_to_line(line, point2)

    p1,p2,d = distance_between_lines(line, extract, clampB0=True, clampB1=True)

    tol = 0.01

    if d<tol is not None:
        if h1+h2<=0.001:
            result =0
        else:
            result = (h1**2 + h2**2)/(h1+h2)/2
    else:
        result = (h1+h2)/2

    if do_debug:
        color_blue = (255, 128, 0)
        color_red = (20, 0, 255)
        image = numpy.full((1000,1000,3),64,dtype=numpy.uint8)
        cv2.line(image,(int(line[0]),int(line[1])),(int(line[2]),int(line[3])),color=color_red)
        cv2.line(image, (int(extract[0]), int(extract[1])), (int(extract[2]), int(extract[3])), color=color_blue)
        cv2.imwrite('./images/output/inter.png',image)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def line_box_intersection(line, box):
    segms = [(box[0], box[1], box[2], box[1]), (box[2], box[1], box[2], box[3]), (box[2], box[3], box[0], box[3]),(box[0], box[3], box[0], box[1])]

    result = numpy.array(line, dtype=int)
    tol = 0.01
    i=0
    for segm in segms:
        if numpy.all(line[:2]==segm[:2]):
            result[i], result[i + 1] = line[0], line[1]
            i += 2
        elif numpy.all(line[:2]==segm[2:]):
            result[i], result[i + 1] = line[0], line[1]
            i += 2
        elif numpy.all(line[2:]==segm[:2]):
            result[i], result[i + 1] = line[2], line[3]
            i += 2
        elif numpy.all(line[2:]==segm[2:]):
            result[i], result[i + 1] = line[2], line[3]
            i += 2
        else:
            p1, p2, d = distance_between_lines(segm, line, clampA0=True, clampA1=True, clampB0=False, clampB1=False)
            if numpy.abs(d) < tol:
                result[i], result[i + 1] = p1[0], p1[1]
                i+=2

        if i==4:break

    return result
# ----------------------------------------------------------------------------------------------------------------------
def line_segm_intersection(line, segms):

    result = numpy.array(line, dtype=int)
    tol = 0.01
    i = 0
    for segm in segms:
        p1, p2, d = distance_between_lines(segm, line, clampA0=True, clampA1=True, clampB0=False, clampB1=False)
        if numpy.abs(d) < tol:
            result[i], result[i + 1] = p1[0], p1[1]
            i += 2

        if i == 4: break
    return
# ----------------------------------------------------------------------------------------------------------------------
def circle_line_intersection(center, R, pt1, pt2, full_line=True, tangent_tol=1e-9,do_debug=False):

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = R ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        result = []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            result = [intersections[0]]
        else:
            result = intersections

        if do_debug:
            W, H = int(1.2 * (center[0] + R)), int(1.2 * (center[1] + R))
            image_debug = numpy.full((H, W, 3), 32, dtype=numpy.uint8)
            cv2.ellipse(image_debug, center, (R,R), 0, startAngle=0, endAngle=360, color=(0, 0, 190),thickness=4)
            cv2.line(image_debug,pt1,pt2,color=(255,128,0),thickness=4)
            for point in intersections:
                cv2.circle(image_debug,(int(point[0]),int(point[1])),4,(255,255,255),thickness=-1)

            cv2.imwrite('./images/output/image_debug.png', image_debug)

    return result
# ----------------------------------------------------------------------------------------------------------------
def ellipse_line_intersection(ellipse, line, full_line=True,do_debug=False):

    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    Rxy = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
    rotation_angle_rad = ellipse[2]*numpy.pi/180.0


    M = numpy.array([[numpy.cos(-rotation_angle_rad), -numpy.sin(-rotation_angle_rad), 0],[numpy.sin(-rotation_angle_rad), numpy.cos(-rotation_angle_rad), 0], [0, 0, 1]])

    pt1 = numpy.array(line[:2],dtype=numpy.float32).reshape((-1,1,2))
    pt1[:,:,0]-=center[0]
    pt1[:,:,1]-=center[1]
    pt1 = cv2.transform(pt1, M)[:,:,:2]
    pt1[:,:,1]*=Rxy[0]/Rxy[1]

    pt2 = numpy.array(line[2:],dtype=numpy.float32).reshape((-1,1,2))
    pt2[:,:,0]-=center[0]
    pt2[:,:,1]-=center[1]
    pt2 = cv2.transform(pt2, M)[:,:,:2]
    pt2[:,:,1] *= Rxy[0] / Rxy[1]

    intersections_trans = circle_line_intersection((0,0), Rxy[0], pt1.flatten(), pt2.flatten(),full_line=full_line)

    intersections = numpy.array(intersections_trans).copy()
    intersections[:,1]/=Rxy[0] / Rxy[1]
    intersections = cv2.transform(numpy.array(intersections,dtype=numpy.float32).reshape((-1,1,2)), numpy.linalg.inv(M))[:, 0, :2]
    intersections[:,0]+=center[0]
    intersections[:,1]+=center[1]


    if do_debug:
        W, H = int(1.4 * (center[0] + Rxy[0])), int(1.4 * (center[1] + Rxy[1]))
        image_original = numpy.full((H, W, 3), 32, dtype=numpy.uint8)
        cv2.ellipse(image_original, center, Rxy, ellipse[2], startAngle=0, endAngle=360, color=(0, 0, 190), thickness=4)
        cv2.line(image_original, tuple(line[:2]), tuple(line[2:]), color=(255, 128, 0), thickness=4)
        for point in intersections:
            cv2.circle(image_original, (int(point[0]), int(point[1])), 4, (255, 255, 255), thickness=-1)
        cv2.imwrite('./images/output/original.png', image_original)

        image_trans = numpy.full((int(Rxy[0] * 4), int(Rxy[0] * 4), 3), 32, dtype=numpy.uint8)
        shift = numpy.array((200, 200))
        cv2.ellipse(image_trans, (shift[0], shift[1]), (Rxy[0],Rxy[0]), 0, startAngle=0, endAngle=360, color=(0, 0, 190), thickness=4)
        cv2.line(image_trans,(int(pt1.flatten()[0])+shift[0], int(pt1.flatten()[1])+shift[1]), (int(pt2.flatten()[0])+shift[0], int(pt2.flatten()[1])+shift[1]), color = (255, 128, 0), thickness=4)
        for point in intersections_trans:
            cv2.circle(image_trans, (shift[0]+int(point[0]), shift[1]+int(point[1])), 4, (255, 255, 255), thickness=-1)
        cv2.imwrite('./images/output/transformed.png', image_trans)


    return intersections
# ----------------------------------------------------------------------------------------------------------------
def distance_point_centredEllipse(Rx, Ry, points):

    res = []
    for p in numpy.array(points).reshape((-1,2)):
        if Rx*Ry==0:
            res.append(numpy.nan)
            continue

        px,py = abs(p[0]),abs(p[1])
        a,b = Rx,Ry
        t = math.pi / 4

        for x in range(0, 3):
            x = a * math.cos(t)
            y = b * math.sin(t)
            ex = (a*a - b*b) * math.cos(t)**3 / a
            ey = (b*b - a*a) * math.sin(t)**3 / b
            rx = x - ex
            ry = y - ey
            qx = px - ex
            qy = py - ey
            r = math.hypot(ry, rx)
            q = math.hypot(qy, qx)
            delta_c = r * math.asin((rx*qy - ry*qx)/(r*q))
            delta_t = delta_c / math.sqrt(a*a + b*b - x*x - y*y)
            t += delta_t
            t = min(math.pi/2, max(0, t))
        x = a * math.cos(t)
        y = b * math.sin(t)
        p_ellipse = numpy.array((math.copysign(x, p[0]), math.copysign(y, p[1])))

        res.append(numpy.linalg.norm(p_ellipse - p))

    return numpy.array(res)
# ----------------------------------------------------------------------------------------------------------------
def distance_point_ellipse(point,ellipse,do_debug=False):
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    Rxy = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
    rotation_angle_deg = ellipse[2]
    rotation_angle_rad = rotation_angle_deg*numpy.pi/180.0

    new_point = numpy.array(point).reshape((-1,1,2))
    new_point[:,:,0]-=center[0]
    new_point[:,:,1]-=center[1]

    M = numpy.array([[numpy.cos(-rotation_angle_rad), -numpy.sin(-rotation_angle_rad), 0],[numpy.sin(-rotation_angle_rad), numpy.cos(-rotation_angle_rad), 0], [0, 0, 1]])
    new_point = cv2.transform(new_point, M)[:,:,:2]

    if do_debug:

        W, H = int(1.4 * (center[0]+Rxy[0])), int(1.4 * (center[1]+Rxy[1]))
        image_original = numpy.full((H, W, 3), 32, dtype=numpy.uint8)
        cv2.ellipse(image_original, center, Rxy, rotation_angle_deg, startAngle=0, endAngle=360, color=(0, 0, 190),thickness=4)
        cv2.circle(image_original,point,4,(255,255,255),thickness=-1)
        cv2.imwrite('./images/output/original.png',image_original)

        image_trans = numpy.full((int(Rxy[1]*4), int(Rxy[0]*4), 3), 32, dtype=numpy.uint8)
        shift = numpy.array((200, 200))
        cv2.ellipse(image_trans, (shift[0],shift[1]), Rxy, 0, startAngle=0, endAngle=360, color=(0, 0, 190), thickness=4)
        cv2.circle(image_trans, (new_point.flatten()[0]+shift[0],new_point.flatten()[1]+shift[1]), 4, (255, 255, 255), thickness=-1)
        cv2.imwrite('./images/output/transformed.png', image_trans)

    return distance_point_centredEllipse(Rxy[0], Rxy[1], new_point)
# ----------------------------------------------------------------------------------------------------------------
def trim_line_by_box(line,box):

    x1,y1,x2,y2 = line
    result = numpy.array(line,dtype=int)
    tol = 0.01

    is_1_in = (x1 >= box[0] and x1 < box[2] and y1 >= box[1] and y1 < box[3])
    is_2_in = (x2>=box[0] and x2<box[2] and y2>=box[1] and y2<box[3])

    if (is_2_in) and (is_1_in):
        return line

    segments = [(box[0], box[1], box[2], box[1]), (box[2], box[1], box[2], box[3]), (box[2], box[3], box[0], box[3]),(box[0], box[3], box[0], box[1])]

    if (not is_1_in) and (is_2_in):
        for segm in segments:
            p1, p2, d = distance_between_lines(segm, line,clampA0=True, clampA1=True,clampB0=False,clampB1=True)
            if numpy.abs(d) < tol:
                result[0], result[1] = p2[0], p2[1]
                return result

    if (not is_2_in) and (is_1_in):
        for segm in segments:
            p1,p2,d = distance_between_lines(segm, line, clampA0=True, clampA1=True,clampB0=True,clampB1=False)
            if numpy.abs(d)<tol:
                result[2],result[3] = p2[0],p2[1]
                return result


    d12 = numpy.linalg.norm(numpy.array((line[0], line[1])) - numpy.array((line[2], line[3])))

    i = 0
    for segm in segments:
        p1,p2,d = distance_between_lines(segm, line, clampA0=True, clampA1=True,clampB0=False,clampB1=False)

        if numpy.abs(d) < tol and (p1 is not None) and (p2 is not None):
            #check if result inside candidate
            d1 = numpy.linalg.norm( numpy.array((p1[0], p1[1])) - numpy.array((line[0], line[1])) )
            d2 = numpy.linalg.norm( numpy.array((p1[0], p1[1])) - numpy.array((line[2], line[3])) )
            if numpy.abs(d1+d2-d12)<tol:
                result[i], result[i+1] = p1[0], p1[1]
                i+=2
                if i == 4:
                    return result
    if i==4:
        return result

    return (numpy.nan,numpy.nan,numpy.nan,numpy.nan)
# ----------------------------------------------------------------------------------------------------------------------
def get_ratio_4_lines(vert1, horz1,vert2, horz2,do_debug=False):
    p11, p12, dist = distance_between_lines(vert1, horz1, clampAll=False)
    p21, p22, dist = distance_between_lines(vert1, horz2, clampAll=False)
    p31, p32, dist = distance_between_lines(vert2, horz1, clampAll=False)
    p41, p42, dist = distance_between_lines(vert2, horz2, clampAll=False)
    rect = (p11[:2], p21[:2], p31[:2], p41[:2])

    lenx1 = numpy.linalg.norm((rect[0] - rect[1]))
    lenx2 = numpy.linalg.norm((rect[2] - rect[3]))

    leny1 = numpy.linalg.norm((rect[1] - rect[3]))
    leny2 = numpy.linalg.norm((rect[2] - rect[0]))

    ratio_xy = (lenx1 + lenx2)/(leny1 + leny2)

    if do_debug:
        color_blue = (255, 128, 0)
        color_red = (20, 0, 255)
        color_gray = (180, 180, 180)
        image = numpy.full((1080, 1920, 3), 64, dtype=numpy.uint8)
        cv2.line(image, (int(vert1[0]), int(vert1[1])), (int(vert1[2]), int(vert1[3])), color_red, thickness=4)
        cv2.line(image, (int(vert2[0]), int(vert2[1])), (int(vert2[2]), int(vert2[3])), color_red, thickness=4)
        cv2.line(image, (int(horz1[0]), int(horz1[1])), (int(horz1[2]), int(horz1[3])), color_blue, thickness=4)
        cv2.line(image, (int(horz2[0]), int(horz2[1])), (int(horz2[2]), int(horz2[3])), color_blue, thickness=4)

        image = tools_draw_numpy.draw_convex_hull(image, rect, color=color_gray, transperency=0.25)
        cv2.imwrite('./images/output/06_ratio.png', image)


    return ratio_xy
# ----------------------------------------------------------------------------------------------------------------------
def get_sizes_4_lines(vert1, horz1,vert2, horz2):
    p11, p12, dist = distance_between_lines(vert1, horz1, clampAll=False)
    p21, p22, dist = distance_between_lines(vert1, horz2, clampAll=False)
    p31, p32, dist = distance_between_lines(vert2, horz1, clampAll=False)
    p41, p42, dist = distance_between_lines(vert2, horz2, clampAll=False)
    rect = (p11[:2], p21[:2], p31[:2], p41[:2])

    lenx1 = numpy.linalg.norm((rect[0] - rect[1]))
    lenx2 = numpy.linalg.norm((rect[2] - rect[3]))

    leny1 = numpy.linalg.norm((rect[1] - rect[3]))
    leny2 = numpy.linalg.norm((rect[2] - rect[0]))

    return (lenx1 + lenx2)/2,(leny1 + leny2)/2
# ----------------------------------------------------------------------------------------------------------------------
def encode_boxed_position(x,y, W, H):

    if   y==0   :code = x
    elif x==W-1: code = W+y
    elif y==H-1: code = W+H+(W-x)
    elif x==0  : code = W+H+W+(H-y)
    else: code = -1

    return int(code)
# ----------------------------------------------------------------------------------------------------------------------
def decode_boxed_position(code, W, H):

    if   0     <= code < W       : x,y = code      , 0
    elif W     <= code < W+H     : x,y = W-1       , code - W
    elif W+H   <= code < W+H+W   : x,y = W+H+W-code, H-1
    elif W+H+W <= code < W+H+W+H : x,y = 0         , W+H+W+H-code
    else:x,y = -1,-1

    return int(x),int(y)
# ----------------------------------------------------------------------------------------------------------------------
def encode_boxed_line(boxed_line, W, H):
    code1 = encode_boxed_position(boxed_line[0], boxed_line[1], W, H)
    code2 = encode_boxed_position(boxed_line[2], boxed_line[3], W, H)
    res = (min(code1, code2), max(code1, code2))

    #test
    #test = decode_boxed_line(res,W,H)


    return res
# ----------------------------------------------------------------------------------------------------------------------
def decode_boxed_line(code,W,H):
    x1,y1 = decode_boxed_position(code[0], W, H)
    x2,y2 = decode_boxed_position(code[1], W, H)

    return (x1,y1,x2,y2)
# ----------------------------------------------------------------------------------------------------------------------
def get_inverce_perspective_mat(image,point_vanishing,line_up,line_bottom,target_width,target_height):
    if line_bottom is None:
        line_bottom = numpy.array((0,image.shape[0],image.shape[1],image.shape[0]),dtype=numpy.float32)

    p1a = (line_up[0], line_up[1])
    p2a = (line_up[2], line_up[3])
    if p1a[0]>p2a[0]:p1a,p2a = p2a,p1a

    p3a = (line_bottom[0], line_bottom[1])
    p4a = (line_bottom[2], line_bottom[3])
    if p3a[0]>p4a[0]:p3a,p4a = p4a,p3a

    p1b = line_intersection(line_up    ,(point_vanishing[0],point_vanishing[1],p3a[0],p3a[1]))
    if p1a[0]<p1b[0]:p1 = p1a
    else:p1 = p1b

    p2b = line_intersection(line_up    ,(point_vanishing[0],point_vanishing[1],p4a[0],p4a[1]))
    if p2a[0]>p2b[0]:p2 = p2a
    else:p2 = p2b


    line1 = (point_vanishing[0],point_vanishing[1],p1[0],p1[1])
    line2 = (point_vanishing[0],point_vanishing[1],p2[0],p2[1])

    p3 = line_intersection(line_bottom,line1)
    p4 = line_intersection(line_bottom,line2)

    M = get_four_point_transform_mat(numpy.array((p1, p2, p3, p4), dtype=numpy.float32), target_width, target_height)

    #res = tools_draw_numpy.draw_points(image,[p1,p2,p3,p4],w=16)

    return M
# ---------------------------------------------------------------------------------------------------------------------
def get_four_point_transform_mat(p_src, target_width, target_height):
    def order_points(pts):
        xSorted = pts[numpy.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[numpy.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        D = dist.cdist(tl[numpy.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[numpy.argsort(D)[::-1], :]
        return numpy.array([tl, tr, br, bl], dtype=numpy.float32)

    p_src_ordered = order_points(p_src)

    #tl, tr, br, bl = p_src_ordered
    #widthA = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    #widthB = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    #maxWidth = max(int(widthA), int(widthB))
    #heightA = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    #heightB = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    #maxHeight = max(int(heightA), int(heightB))

    (maxWidth, maxHeight) = (target_width,target_height)

    p_dst = numpy.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype=numpy.float32)
    M = cv2.getPerspectiveTransform(p_src_ordered, p_dst)

    return M
# ---------------------------------------------------------------------------------------------------------------------
