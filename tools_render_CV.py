import math
import cv2
import numpy
import pyrr
from sympy.geometry import intersection, Point, Line
from scipy.spatial import distance as dist
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_IO
from CV import tools_pr_geom
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
def draw_grid(image, camera_matrix, dist, rvec, tvec,color=(0,0,190),w=1):

    z = 0
    step  =1
    x_min, x_max = 0.0, +100
    y_min, y_max = -1, +1
    image_result = image.copy()

    for x in numpy.arange(x_min,x_max,step):
        lines_3d = numpy.array([[x, y_min, z], [x, y_max, z] ],dtype = numpy.float32)
        lines_2d, jac = cv2.projectPoints(lines_3d, rvec, tvec, camera_matrix, dist)
        image_result = tools_draw_numpy.draw_lines(image_result,lines_2d.reshape((-1, 4)),color,w)


    for y in numpy.arange(y_min, y_max, step):
        lines_3d = numpy.array([[x_min, y, z], [x_max, y, z]], dtype=numpy.float32)
        lines_2d, jac = cv2.projectPoints(lines_3d, rvec, tvec, camera_matrix, dist)
        image_result = tools_draw_numpy.draw_lines(image_result, lines_2d.reshape((-1, 4)),color,w)

    return image_result
# ----------------------------------------------------------------------------------------------------------------------
def draw_compass_p3x4(image, p3x4, R, Z=0, step=1, color=(0,128,255),draw_labels=False):
    points_3d = numpy.array([(R * numpy.sin(angle * numpy.pi / 180.0), R * numpy.cos(angle * numpy.pi / 180.0), Z) for angle in range(0, 360, step)])
    points_2d = tools_pr_geom.project_points_p3x4(points_3d, p3x4)
    idx = (points_2d[:, 0] > 0) * (points_2d[:, 1] > 0) * (points_2d[:, 0] < image.shape[1]) * (points_2d[:, 1] < image.shape[0])

    image_result = image.copy()
    if numpy.array(1 * idx).sum() > 1:
        points_2d_visible = points_2d[idx]
        lines_2d = [
            (points_2d_visible[p, 0], points_2d_visible[p, 1], points_2d_visible[p + 1, 0], points_2d_visible[p + 1, 1])
            for p in range(len(points_2d_visible) - 1)]
        image_result = tools_draw_numpy.draw_lines(image, lines_2d, color=color, w=2)

        xy = (int(points_2d_visible[:, 0].mean()), int(points_2d_visible[:, 1].mean()))
        cv2.putText(image_result, '{0}'.format(R), xy, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 1, cv2.LINE_AA)

        if draw_labels:
            colors = tools_draw_numpy.get_colors(len(points_2d), colormap='rainbow')
            labels = numpy.array([('%d' % d) for d in range(0, 360, step)])
            image_result = tools_draw_numpy.draw_points(image_result, points_2d[idx], colors[idx], labels=labels[idx])

    return image_result
# ----------------------------------------------------------------------------------------------------------------------
def draw_compass(image, camera_matrix, M, R, Z=0, step=1, color=(0,128,255),draw_labels=False):

    points_3d = numpy.array([(R * numpy.sin(angle * numpy.pi / 180.0), R * numpy.cos(angle * numpy.pi / 180.0), Z) for angle in range(0, 360, step)])

    if camera_matrix.shape[0]==3 and camera_matrix.shape[1]==3:
        points_2d = tools_pr_geom.project_points_M(points_3d, M, camera_matrix)
        points_2d=points_2d.reshape((-1,2))
        points_2dx = tools_pr_geom.apply_matrix_GL(M, points_3d)
        is_front = points_2dx[:, 2] > 0
    else:
        points_2d = tools_pr_geom.project_points_p3x4(points_3d, camera_matrix)
        points_2dx = tools_pr_geom.project_points_p3x4(points_3d, camera_matrix, check_reverce=True)
        is_front = numpy.array([~numpy.any(numpy.isnan(p)) for p in points_2dx])


    idx = (points_2d[:,0]>0) * (points_2d[:,1]>0) * (points_2d[:,0]<image.shape[1]) * (points_2d[:,1]<image.shape[0]) * (is_front[:])
    image_result = image.copy()
    if numpy.array(1*idx).sum()>1:
        points_2d_visible = points_2d[idx]
        idx = numpy.argsort(points_2d_visible[:,0])
        points_2d_visible = points_2d_visible[idx]
        lines_2d =[(points_2d_visible[p,0],points_2d_visible[p,1],points_2d_visible[p+1,0],points_2d_visible[p+1,1]) for p in range(len(points_2d_visible)-1)]
        image_result = tools_draw_numpy.draw_lines(image,lines_2d,color=color,w=2)

        xy = (int(points_2d_visible[:, 0].mean()), int(points_2d_visible[:, 1].mean()))
        cv2.putText(image_result, '{0}'.format(R), xy, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 1, cv2.LINE_AA)

        if draw_labels:
            colors = tools_draw_numpy.get_colors(len(points_2d), colormap='rainbow')
            labels = numpy.array([('%d' % d) for d in range(0, 360, step)])
            image_result = tools_draw_numpy.draw_points(image_result, points_2d[idx], colors[idx],labels=labels[idx])

    return image_result
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
def draw_image(im_target, im_source_one, camera_matrix, dist, rvec, tvec,scale=(1,1,1)):

    H, W = im_source_one.shape[:2]
    points_2d_source = numpy.array([[0, 0], [0, H], [W, H], [W, 0]], dtype=numpy.float32)

    points_3d = numpy.array([[-1, +1, 0], [-1, -1, 0],[+1, -1, 0], [+1, +1, 0]], dtype=numpy.float32)
    points_3d[:, 0] *= scale[0]
    points_3d[:, 1] *= scale[1]
    points_3d[:, 2] *= scale[2]

    points_2d_target, jac = tools_pr_geom.project_points(points_3d, rvec, tvec, camera_matrix, dist)

    homography, result  = tools_pr_geom.fit_homography(points_2d_source,points_2d_target)

    image_trans = cv2.warpPerspective(im_source_one, homography, (im_target.shape[1], im_target.shape[0]))
    mask =  cv2.warpPerspective(numpy.ones((H,W)), homography, (im_target.shape[1], im_target.shape[0]))
    mask = ~(mask>0)
    image_trans[mask]=im_target[mask]


    return  image_trans
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube_numpy(img, camera_matrix, dist, rvec, tvec, scale=(1,1,1),color=(255,128,0),points_3d=None):

    if points_3d is None:
        points_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],[-1, -1, +1], [-1, +1, +1], [+1, +1, +1], [+1, -1, +1]],dtype = numpy.float32)
        points_3d[:,0]*=scale[0]
        points_3d[:,1]*=scale[1]
        points_3d[:,2]*=scale[2]

    colors = tools_draw_numpy.get_colors(8)
    if len(rvec.shape)==2 and rvec.shape[0]==3 and rvec.shape[1]==3:
        method = 'xx'
    else:
        method = 'cv'

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

    clr = (0, 0, 255)
    #for i in (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3):
        #cv2.putText(result, '{0}   {1} {2} {3}'.format(i, points_3d[i,0],points_3d[i,1],points_3d[i,2]), (points_2d[i, 0], points_2d[i, 1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube_numpy_MVP_GL(img, mat_projection, mat_view, mat_model, mat_trns, color=(255, 128, 0), w=2, points_3d=None):

    lines_idx = [(1, 0), (0, 2), (2, 3), (3, 1), (7, 6), (6, 4), (4, 5), (5, 7), (1, 0), (0, 6), (6, 7), (7, 1), (3, 2),(2, 4), (4, 5), (5, 3)]
    if points_3d is None:
        points_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],[-1, -1, +1], [-1, +1, +1], [+1, +1, +1], [+1, -1, +1]],dtype = numpy.float32)
        lines_idx = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    img = draw_points_numpy_MVP_GL(points_3d, img, mat_projection, mat_view, mat_model, mat_trns, color, do_debug=False)
    lines_3d = []

    for (i,j) in lines_idx:
        lines_3d.append((points_3d[i, 0], points_3d[i, 1],points_3d[i, 2], points_3d[j, 0], points_3d[j, 1],points_3d[j, 2]))

    img = draw_lines_numpy_MVP(numpy.array(lines_3d), img, mat_projection, mat_view, mat_model, mat_trns,color,w)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_numpy_MVP_GL(points_3d, img, mat_projection, mat_view, mat_model, mat_trns, color=(66, 0, 166), w = 2, do_debug=False):
    # all matrices should be in GL style
    camera_matrix_3x3 = tools_pr_geom.compose_projection_mat_3x3(img.shape[1], img.shape[0], 1 / mat_projection[0][0],1 / mat_projection[1][1])

    #M = pyrr.matrix44.multiply(mat_view, pyrr.matrix44.multiply(mat_model, mat_trns))
    #L3D = tools_pr_geom.apply_matrix(M.T, points_3d)

    M0 = pyrr.matrix44.multiply(mat_view.T,pyrr.matrix44.multiply(mat_model.T,mat_trns.T))
    X4D = numpy.hstack((points_3d,numpy.full((len(points_3d),1),1)))
    L3D = pyrr.matrix44.multiply(M0, X4D.T).T[:,:3]



    method=1
    if method==0:
        img = draw_points_numpy_RT(L3D,img,numpy.eye(4),camera_matrix_3x3,color,w)

    elif method==1:
        #opencv equivalent
        #points_2d, jac = cv2.projectPoints(           L3D, numpy.array((0, 0, 0)), numpy.array((0, 0, 0)), camera_matrix_3x3, numpy.zeros(4,dtype=float))
        points_2d, jac = tools_pr_geom.project_points(L3D, numpy.array((0, 0, 0)), numpy.array((0, 0, 0)), camera_matrix_3x3, numpy.zeros(4))
        points_2d = points_2d.reshape((-1,2))
        points_2d[:, 0] = img.shape[1] - points_2d[:, 0]
        for point in points_2d:
            if numpy.any(numpy.isnan(point)):continue
            if numpy.any(numpy.isinf(point)): continue
            img = tools_draw_numpy.draw_circle(img, int(point[1]), int(point[0]), w, color)

    if do_debug:
        posx,posy = img.shape[1]-250,0
        draw_mat(mat_trns,  posx, posy+20, img,(0,128,255))
        draw_mat(mat_model, posx, posy+120, img,(0,128,255))
        draw_mat(mat_view,  posx, posy+220, img,(0,128,255))
        draw_mat(mat_projection, posx, posy+320, img,(0,128,255))
        draw_mat(mat_projection, posx, posy+320, img,(0,128,255))

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_points_numpy_RT(points_3d,img,RT,camera_matrix_3x3,color=(66, 0, 166),w=6,transperency=0):

    points_2d = tools_pr_geom.project_points_M(points_3d, RT, camera_matrix_3x3, numpy.zeros(5)).reshape((-1, 2))
    img = tools_draw_numpy.draw_points(img, points_2d, color=color, w=w,transperency=transperency)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_lines_numpy_RT(lines_3d,img,RT,camera_matrix_3x3,color=(66, 0, 166),w=6):
    points_2d_start = tools_pr_geom.project_points_M(lines_3d[:,:3], RT, camera_matrix_3x3, numpy.zeros(5)).reshape((-1, 2))
    points_2d_end = tools_pr_geom.project_points_M(lines_3d[:, 3:], RT, camera_matrix_3x3, numpy.zeros(5)).reshape((-1, 2))

    for point_start, point_end in zip(points_2d_start, points_2d_end):
        cv2.line(img, (int(point_start[0]), int(point_start[1])), (int(point_end[0]), int(point_end[1])), color,thickness=w)

    return img
# ----------------------------------------------------------------------------------------------------------------------
def draw_lines_numpy_MVP(lines_3d, img, mat_projection, mat_view, mat_model, mat_trns,color=(66, 0, 166),w=6):
    if lines_3d.shape[0]==0:
        return img
    points_2d_start = tools_pr_geom.project_points_MVP(lines_3d[:, :3],img, mat_projection, mat_view, mat_model, mat_trns)
    points_2d_end = tools_pr_geom.project_points_MVP(lines_3d[:, 3:], img, mat_projection, mat_view, mat_model, mat_trns)

    for point_start, point_end in zip(points_2d_start, points_2d_end):
        if numpy.any(numpy.isnan(point_start)) or numpy.any(numpy.isnan(point_end)):continue
        if numpy.any(numpy.isinf(point_start)) or numpy.any(numpy.isinf(point_end)): continue
        line = (int(point_start[0]), int(point_start[1]),int(point_end[0]), int(point_end[1]))
        img = tools_draw_numpy.draw_lines(img, [line],color=color,w=w)

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
def inverce_project(point_2d,image,mat_projection, mat_view, mat_model, mat_trns,distance=None):

    H, W = image.shape[:2]
    tg_half_fovx = 1.0 / mat_projection[0][0]

    far_plane_distance = 0.5 * W / tg_half_fovx if distance is None else distance
    far_plane_W = 2 * far_plane_distance * tg_half_fovx
    far_plane_H = far_plane_W * H / W
    x = (-point_2d[0] + W / 2) * far_plane_distance / (0.5 * W / tg_half_fovx)
    y = (+point_2d[1] - H / 2) * far_plane_distance / (0.5 * W / tg_half_fovx)
    far_plane = numpy.array(((-far_plane_W / 2, 0, +far_plane_H / 2, 1), (-far_plane_W / 2, 0, -far_plane_H / 2, 1),(+far_plane_W / 2, 0, +far_plane_H / 2, 1), (+far_plane_W / 2, 0, -far_plane_H / 2, 1),(x, 0, y, 1)))
    t0 = (0, far_plane_distance, 0)

    mat_view0 = tools_pr_geom.ETU_to_mat_view((0, 0, 0), (0, 1, 0), (0, 0, -1))
    mat_model0 = tools_pr_geom.compose_RT_mat((0, 0, 0), (0, 0, 0), do_rodriges=False, do_flip=False,GL_style=True)
    M_obj0 = tools_pr_geom.compose_RT_mat((0, 0, 0), t0, do_rodriges=False, do_flip=False, GL_style=True)

    M = tools_pr_geom.multiply([M_obj0,
                                tools_pr_geom.multiply([mat_trns, mat_model0]),
                                tools_pr_geom.multiply([mat_view0, numpy.linalg.pinv(mat_view)]),
                                tools_pr_geom.multiply([numpy.linalg.pinv(mat_model), numpy.linalg.pinv(mat_trns)]),
                                ])

    far_plane_t = tools_pr_geom.apply_matrix_GL(M.T, far_plane)[:,:3]
    point_3d = far_plane_t[-1]
    #check
    # point_2d_check = tools_pr_geom.project_points_MVP_GL(far_plane_t, image, mat_projection, mat_view,mat_model, mat_trns)
    # far_plane_t1 = tools_pr_geom.apply_matrix_GL(M.T, far_plane)
    # far_plane_t2 = far_plane.dot(M)[:, :3]
    # empty = numpy.full((H, W, 3), 32, dtype=numpy.uint8)
    # ray_end = far_plane_t2[-1]
    # points_2d = tools_pr_geom.project_points_MVP_GL(far_plane_t2, empty, mat_projection, mat_view,mat_model, mat_trns)
    # X0 = tools_pr_geom.multiply([far_plane, M_obj0, mat_view0, mat_model0, mat_trns])
    # X0p = tools_pr_geom.project_points_MVP_GL(far_plane.dot(M_obj0)[:, :3], empty, mat_projection, mat_view0,mat_model0, mat_trns)
    # X1 = tools_pr_geom.multiply([far_plane_t1, mat_view, mat_model, mat_trns])
    # X1p = tools_pr_geom.project_points_MVP_GL(far_plane_t1[:, :3], empty, mat_projection, mat_view,mat_model, mat_trns)

    return point_3d
# ----------------------------------------------------------------------------------------------------------------------
def get_ray(point_2d,image,mat_projection, mat_view, mat_model, mat_trns,d = None):
    H, W = image.shape[:2]
    tg_half_fovx = 1.0 / mat_projection[0][0]
    far_plane_distance = 0.5 * W / tg_half_fovx if d is None else d
    ray_start = inverce_project(point_2d, image, mat_projection, mat_view, mat_model, mat_trns,far_plane_distance+1)
    ray_end   = inverce_project(point_2d, image, mat_projection, mat_view, mat_model, mat_trns,far_plane_distance)

    return ray_start,ray_end
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
def angle_between_lines(lineA,lineB):
    # def slope(x1, y1, x2, y2):return (y2-y1)/(x2-x1)
    # def angle(s1, s2): return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))
    # slope1 = slope(lineA[0], lineA[1], lineA[2], lineA[3])
    # slope2 = slope(lineB[0], lineB[1], lineB[2], lineB[3])
    # ang0 = angle(slope1, slope2)

    a = (lineA[0]-lineA[2],lineA[1]-lineA[3])
    b = (lineB[0]-lineB[2],lineB[1]-lineB[3])
    d = numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))
    ang = math.degrees(math.acos(d))
    return ang
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
def get_interception_ray_triangle(pos, direction, triangle,allow_outside=False):
    n = get_normal(triangle)
    collision = line_plane_intersection(n, triangle[0,:3], direction[:3], pos[:3], epsilon=1e-6)

    if collision is not None:
        if (allow_outside==True) or is_point_inside_triangle(collision,triangle):
            return collision

    return None
# ----------------------------------------------------------------------------------------------------------------------
def get_interceptions_ray_triangles(pos, direction,coord_vert, idxv):

    collisions = [get_interception_ray_triangle(pos, direction, coord_vert[iv])for iv in idxv]
    collisions = [c for c in collisions if c is not None]

    return collisions
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
    t = (b0 - a0)
    detA = numpy.linalg.det([t, _B, cross])
    detB = numpy.linalg.det([t, _A, cross])

    t0 = detA/denom
    t1 = detB/denom

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
        intersections = []
        for sign in ((1, -1) if dy < 0 else (-1, 1)):
            intersections.append((cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2, cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2))

        # intersections = [(cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2, cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
        #     for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= abs(frac) <= 1]
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

    if Rxy[1] == 0: return  []

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
    if len(intersections_trans)>0 and Rxy[1]>0:
        intersections = numpy.array(intersections_trans).copy()
        intersections[:,1]/=Rxy[0] / Rxy[1]
        intersections = cv2.transform(numpy.array(intersections,dtype=numpy.float32).reshape((-1,1,2)), numpy.linalg.inv(M))[:, 0, :2]
        intersections[:,0]+=center[0]
        intersections[:,1]+=center[1]
    else:
        intersections = []

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

def regularize_rect(point_2d, inlined = True, do_debug=False):
    point_2d_centred = point_2d - numpy.mean(point_2d, axis=0)
    empty = numpy.full((1000, 1000, 3),190,dtype=numpy.uint8)
    a_range = numpy.arange(0, 180, 5)
    losses,results = [],[]

    if do_debug:
        tools_IO.remove_files('./images/output/','*.png')

    for a in a_range:
        R = numpy.array([[math.cos(a * numpy.pi / 180), -math.sin(a * numpy.pi / 180)],[math.sin(a * numpy.pi / 180), math.cos(a * numpy.pi / 180)]])
        p1 = R.dot(numpy.array((0,10)))
        p2 = R.dot(numpy.array((10,0)))
        line1 = numpy.array((0,0,p1[0],p1[1]))
        line2 = numpy.array((0,0,p2[0],p2[1]))
        p_perp1 = numpy.array([perp_point_line(line1, p) for p in point_2d_centred])
        p_perp2 = numpy.array([perp_point_line(line2, p) for p in point_2d_centred])
        idx1 = numpy.array([is_point_above_line(p, line2) for p in p_perp1])
        idx2 = numpy.array([is_point_above_line(p, line1) for p in p_perp2])

        if inlined:
            d11 =  numpy.array([numpy.linalg.norm(p) for p in p_perp1[ idx1]]).mean()
            d12 = -numpy.array([numpy.linalg.norm(p) for p in p_perp1[~idx1]]).mean()
            d21 =  numpy.array([numpy.linalg.norm(p) for p in p_perp2[ idx2]]).mean()
            d22 = -numpy.array([numpy.linalg.norm(p) for p in p_perp2[~idx2]]).mean()
        else:
            d11 =  numpy.array([numpy.linalg.norm(p) for p in p_perp1[ idx1]]).min()
            d12 = -numpy.array([numpy.linalg.norm(p) for p in p_perp1[~idx1]]).min()
            d21 =  numpy.array([numpy.linalg.norm(p) for p in p_perp2[ idx2]]).min()
            d22 = -numpy.array([numpy.linalg.norm(p) for p in p_perp2[~idx2]]).min()

        pp = numpy.array([R.dot(x) for x in numpy.array(((d21,d11),(d21,d12),(d22,d11),(d22,d12)))])

        res = []
        loss = 0
        for p in point_2d_centred:
            v = numpy.array([numpy.linalg.norm(x - p) for x in pp])
            loss+=v.min()
            res.append(pp[numpy.argmin(v)])

        res_test = set(map(sum, res))
        if len(res_test) == 4:
            results.append(numpy.array(res))
            losses.append(loss)

        if do_debug:
            image = tools_draw_numpy.draw_convex_hull(empty, 150 * point_2d_centred + 500, color=(0,0,255),transperency=0.5)
            image = tools_draw_numpy.draw_convex_hull(image, 150 * pp + 500              , color=(255,200,0),transperency=0.5)

            image = tools_draw_numpy.draw_lines(image, [50*line1+500], (64,64,64),w=1)
            image = tools_draw_numpy.draw_lines(image, [50*line2+500], (64,64,64),w=1)
            cv2.imwrite('./images/output/xxx_%05d.png'%int(10*loss),image)


    result = numpy.mean(point_2d, axis=0) + numpy.array(results)[numpy.argmin(numpy.array(losses))]

    return result
# ----------------------------------------------------------------------------------------------------------------------
def plane_plane_intersection(a,b):
    a_vec, b_vec = numpy.array(a[:3]), numpy.array(b[:3])
    aXb_vec = numpy.cross(a_vec, b_vec)
    A = numpy.array([a_vec, b_vec, aXb_vec])
    d = numpy.array([-a[3], -b[3], 0.]).reshape(3, 1)
    p_inter = numpy.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]
# ----------------------------------------------------------------------------------------------------------------------
def distance_point_to_line_3d(line, point):
    p_start, p_end = line[:3],line[3:]
    d = (p_end - p_start) / (numpy.linalg.norm(p_end-p_start))
    v = point - p_start
    t = v.dot(d)
    P = p_start + t * d
    return numpy.linalg.norm(P - point)
# ----------------------------------------------------------------------------------------------------------------------