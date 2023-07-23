import math
import cv2
import pyrr
import numpy
from sklearn.linear_model import LinearRegression
from scipy.linalg import polar
# ---------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_render_CV
# ---------------------------------------------------------------------------------------------------------------------
def debug_projection(X_source, X_target,result):
    colors_s = tools_draw_numpy.get_colors(len(X_source))
    colors_t = colors_s.copy()
    colors_s = numpy.array([((255,255,255)+c)/2 for c in colors_s])


    padding = 10

    xmin = int(min(X_source[:,0].min(), X_target[:,0].min(),result[:,0].min()))-padding
    xmax = int(max(X_source[:,0].max(), X_target[:,0].max(),result[:,0].max()))+padding
    ymin = int(min(X_source[:,1].min(), X_target[:,1].min(),result[:,1].min()))-padding
    ymax = int(max(X_source[:,1].max(), X_target[:,1].max(),result[:,1].max()))+padding

    image = numpy.full((ymax-ymin+1,xmax-xmin+1,3),32,dtype=numpy.uint8)

    for i,(x,y) in enumerate(X_target):cv2.circle(image, (int(x-xmin),int(y-ymin)),18,(int(colors_t[i][0]),int(colors_t[i][1]),int(colors_t[i][2])),  2)
    for i,(x,y) in enumerate(result)  :cv2.circle(image, (int(x-xmin),int(y-ymin)),12,(int(colors_s[i][0]),int(colors_s[i][1]),int(colors_s[i][2])),  2)
    for i,(x,y) in enumerate(X_source):
        cv2.circle(image, (int(x-xmin),int(y-ymin)), 4, colors_s[i].tolist(), -1)
        cv2.putText(image, '{0}'.format(i), (int(x-xmin),int(y-ymin)), cv2.FONT_HERSHEY_SIMPLEX,0.6, colors_s[i].tolist(), 1, cv2.LINE_AA)

    return image
# ----------------------------------------------------------------------------------------------------------------------
def debug_projection_2d(X_target,X_result):
    colors_s = tools_draw_numpy.get_colors(len(X_target))
    colors_s = numpy.array([numpy.array((255,255,255.0))*0.25+c*0.75 for c in colors_s],dtype=numpy.uint8)


    padding = 10

    xmin = int(min(X_target[:,0].min(), X_result[:,0].min()))-padding
    xmax = int(max(X_target[:,0].max(), X_result[:,0].max()))+padding
    ymin = int(min(X_target[:,1].min(), X_result[:,1].min()))-padding
    ymax = int(max(X_target[:,1].max(), X_result[:,1].max()))+padding

    image = numpy.full((ymax-ymin+1,xmax-xmin+1,3),32,dtype=numpy.uint8)


    for i,(x,y) in enumerate(X_result)  :cv2.circle(image, (int(x-xmin),int(y-ymin)),12,(int(colors_s[i][0]),int(colors_s[i][1]),int(colors_s[i][2])),  2)
    for i,(x,y) in enumerate(X_target):
        cv2.circle(image, (int(x-xmin),int(y-ymin)), 4, colors_s[i].tolist(), -1)
        cv2.putText(image, '{0}'.format(i), (int(x-xmin),int(y-ymin)), cv2.FONT_HERSHEY_SIMPLEX,0.6, colors_s[i].tolist(), 1, cv2.LINE_AA)

    return image
# ----------------------------------------------------------------------------------------------------------------------


# translation -> euclide (=rotation + translation) -> affine (keep parallelism) -> homography
# ----------------------------------------------------------------------------------------------------------------------
def fit_translation(X_source,X_target):
    t = numpy.mean((X_target  - X_source ),axis=0)
    E = pyrr.matrix44.create_from_translation(t).T
    #X4D = numpy.full((X_source.shape[0],4),1,dtype=numpy.float)
    #X4D[:,:2]=X_source
    result = pyrr.matrix44.multiply(E,X_source.T).T
    loss = ((result - X_target) ** 2).mean()
    return E, result
# ----------------------------------------------------------------------------------------------------------------------
def fit_euclid(X_source,X_target,do_debug=False):
    E, _ = cv2.estimateAffinePartial2D(numpy.array(X_source), numpy.array(X_target))
    result = cv2.transform(X_source.reshape((-1, 1, 2)), E).reshape((-1,2))
    loss = ((result - X_target) ** 2).mean()
    if do_debug:debug_projection(X_source, X_target,result)

    return E, result
# ----------------------------------------------------------------------------------------------------------------------
def fit_affine(X_source,X_target,do_debug=False):
    A, _ = cv2.estimateAffine2D(numpy.array(X_source), numpy.array(X_target), confidence=0.95)
    result = cv2.transform(X_source.reshape((-1, 1, 2)), A).reshape((-1,2))
    loss = ((result - X_target) ** 2).mean()
    if do_debug:
        image = debug_projection(X_source, X_target, result)
    return A, result
# ----------------------------------------------------------------------------------------------------------------------
def fit_homography(X_source,X_target,method = cv2.RANSAC,do_debug=False):

    #method = cv2.LMEDS
    #method = cv2.RHO
    if X_source.shape[0]<5 or X_target.shape[0]<5:
        return None,None

    H, mask = cv2.findHomography(X_source, X_target, method, 3.0)
    xxx = X_source.reshape((-1, 1, 2)).astype(numpy.float32)
    result  = cv2.perspectiveTransform(xxx,H).reshape((-1,2))

    loss =  ((result-X_target)**2).mean()
    if do_debug:
        image_debug = debug_projection(X_source, X_target, result)
        #cv2.imwrite('./images/output/image_dbug.png',image_debug)

    return H, result
# ----------------------------------------------------------------------------------------------------------------------
def fit_regression(X_source,X_target,do_debug=None):

    tol = 0.01

    H = numpy.zeros((2, 3),dtype=numpy.float32)
    reg = LinearRegression()
    S=numpy.array([X_source[:,0]],dtype=numpy.float32).T
    T=numpy.array([X_target[:,0]],dtype=numpy.float32).T
    reg.fit(S, T)
    result_X = reg.predict(S)
    check_X = S*reg.coef_[0] + reg.intercept_[0]
    H[0, 0] = float(reg.coef_[0])
    H[0, 2] = float(reg.intercept_[0])
    if H[0, 0]<tol:
        H[0, 0] = 1
        H[0, 2] = 0

    S=numpy.array([X_source[:,1]],dtype=numpy.float32).T
    T=numpy.array([X_target[:,1]],dtype=numpy.float32).T
    reg.fit(S, T)
    result_Y = reg.predict(S)
    result = numpy.hstack((result_X, result_Y))
    H[1, 1] = float(reg.coef_[0])
    H[1, 2] = float(reg.intercept_[0])

    if H[1, 1]<tol:
        H[1, 1] = 1
        H[1, 2] = 0

    if do_debug:
        debug_projection(X_source, X_target, result)

    return H,result
# ----------------------------------------------------------------------------------------------------------------------
def fit_pnp(landmarks_3d,landmarks_2d,mat_camera_3x3,dist=numpy.zeros(5)):

    if len(landmarks_3d)<=3:
        return None,None,None

    #(_, r_vec, t_vec) = cv2.solvePnP(landmarks_3d, landmarks_2d,mat_camera_3x3, dist)
    _, r_vec, t_vec ,_ = cv2.solvePnPRansac(landmarks_3d, landmarks_2d, mat_camera_3x3, dist)

    landmarks_2d_check, jac = cv2.projectPoints(landmarks_3d, r_vec, t_vec, mat_camera_3x3, dist)
    landmarks_2d_check = numpy.reshape(landmarks_2d_check, (-1, 2))

    return r_vec, t_vec, landmarks_2d_check
# ----------------------------------------------------------------------------------------------------------------------
#
def fit_p3l(lines_3d,lines_2d,mat_camera_3x3):
    from zhou_accv_2018 import p3l
    points3d_start = lines_3d[:, :3]
    points3d_end   = lines_3d[:, 3:]

    points3d = (points3d_start + points3d_end ) / 2
    directions3d = (points3d_end  - points3d_start)
    norms = numpy.linalg.norm(directions3d, axis=1)[:, None]
    directions3d = directions3d / norms

    directions3d*=numpy.array([1,-1,1])[:,None]

    poses = p3l(numpy.array(lines_2d).reshape((3,2,2)), (points3d, directions3d), mat_camera_3x3)
    return poses

# ----------------------------------------------------------------------------------------------------------------------
def fit_manual(X3D,target_2d,fx, fy,xref=None,lref=None,do_debug=True):

    P = compose_projection_ortho_mat(fx, fy, (200, 200))

    n_steps = 90
    rotation_range = numpy.arange(0, 2*math.pi, 2*math.pi / n_steps)
    rotation_range[0] = 0

    X4D = numpy.full((X3D.shape[0], 4), 1)
    X4D[:, :3] = X3D

    X4Dref = numpy.full((xref.shape[0], 4), 1)
    X4Dref[:, :3] = xref

    X4Dlines = numpy.full((2*lref.shape[0], 4), 1)
    X4Dlines[:, :2] = lref.reshape(-1,2)

    M = None
    a2=0,0
    for a0 in rotation_range:
        R = pyrr.matrix44.create_from_eulers((0,a0,0))
        T = pyrr.matrix44.create_from_translation((0,0,0)).T
        M = pyrr.matrix44.multiply(T, R)
        PRT = pyrr.matrix44.multiply(P, M)
        projection_2d     = (pyrr.matrix44.multiply(PRT, X4D.T).T)[:, :2]
        projection_2d_ref = (pyrr.matrix44.multiply(PRT, X4Dref.T).T)[:, :2]
        projection_2d_lin = (pyrr.matrix44.multiply(PRT, X4Dlines.T).T)[:, :2]

        E, result_E  = fit_euclid(projection_2d, target_2d)
        result_E     = cv2.transform(projection_2d.reshape(-1, 1, 2), E).reshape((-1, 2))
        result_E_ref = cv2.transform(projection_2d_ref.reshape(-1, 1, 2), E).reshape((-1, 2))
        result_E_lin = cv2.transform(projection_2d_lin.reshape(-1, 1, 2), E).reshape((-1, 2))

        if do_debug:
            image = numpy.full((int(fy), int(fx), 3), 20)
            for x in target_2d   : cv2.circle(image, (int(x[0]), int(x[1])), 16, (0, 128, 255), 4)
            for x in result_E    : cv2.circle(image, (int(x[0]), int(x[1])), 4, (0, 32, 190), -1)
            for x in result_E_ref: cv2.circle(image, (int(x[0]), int(x[1])), 4, (128, 128, 128), -1)
            for i in numpy.arange(0,result_E_lin.shape[0]-1,2):cv2.line(image, (int(result_E_lin[i,0]), int(result_E_lin[i,1])),(int(result_E_lin[i+1,0]), int(result_E_lin[i+1,1])), (128, 128, 128), 2)
            cv2.imwrite('./data/output/fit_%03d.png' % int(a0*180/math.pi), image)

    return M
# ----------------------------------------------------------------------------------------------------------------------
def fit_R(points_3d,points_2d,camera_matrix,rvec):

    a_roll_deg = 0
    range_pitch = numpy.arange(-10,10,1)
    range_yaw = numpy.arange(-10,10,1)
    g_loss = None
    g_rvec = None
    g_tvec = None

    for height in numpy.arange(0,20,0.19):
        for a_pitch_deg in range_pitch:
            for a_yaw_deg in range_yaw:
                tvec_new = numpy.array((0, height, 0 ))

                r = numpy.array(rvec).flatten() + numpy.array((a_pitch_deg*numpy.pi/180,a_yaw_deg*numpy.pi/180,a_roll_deg*numpy.pi/180))
                points_2d_cand,err = project_points(points_3d, r, tvec_new, camera_matrix,numpy.zeros(5))
                loss = ((points_2d_cand-points_2d)**2).sum()
                if g_loss is None or loss<g_loss:
                    g_loss = loss
                    g_rvec = r
                    g_tvec = tvec_new

                # image_result = tools_draw_numpy.draw_points(empty, points_2d_cand, color=(0, 128, 255), w=16)
                # image_result = tools_draw_numpy.draw_points(image_result, points_2d, color=(0, 0, 255), w=8)
                # cv2.imwrite('./output/xxx_%1.3f.png' % (180 + a_yaw_deg), image_result)

    points_2d_cand, err = project_points(points_3d, g_rvec, g_tvec, camera_matrix, numpy.zeros(5))
    # image_result = tools_draw_numpy.draw_points(empty, points_2d_cand, color=(0, 128, 255), w=16)
    # image_result = tools_draw_numpy.draw_points(image_result, points_2d, color=(0, 0, 255), w=8)
    # cv2.imwrite('D:/Soccer/output/yyy_%04d.png'%camera_matrix[0,0], image_result)

    return g_rvec,g_tvec,points_2d_cand
# ----------------------------------------------------------------------------------------------------------------------
def compose_projection_mat_3x3(W, H, tg_half_fovx=0.5, tg_half_fovy=None):
    fx = 0.5*W/(tg_half_fovx)
    fy = 0.5*H/(tg_half_fovx)
    mat_camera_3x3 = numpy.array([[fx, 0., W/2], [0., fx, H/2], [0., 0., 1.]])

    return mat_camera_3x3
# ----------------------------------------------------------------------------------------------------------------------
def compose_projection_mat_4x4(fx, fy, fov_x=0.5, fov_y=0.5):
    mat_camera_4x4 = numpy.array([[fx, 0, fx * fov_x, 0], [0, fy, fy * fov_y, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return mat_camera_4x4
# ----------------------------------------------------------------------------------------------------------------------
def convert_projection_mat_3x3(camera_matrix_3x3):
    fx=camera_matrix_3x3[0, 0]
    fy=camera_matrix_3x3[1, 1]
    fov_x = camera_matrix_3x3[0, 2] / camera_matrix_3x3[0, 0]
    fov_y = camera_matrix_3x3[1, 2] / camera_matrix_3x3[1, 1]
    #mat_camera_4x4 = compose_projection_mat_4x4(fx, fy,fov_x,fov_y)
    mat_camera_4x4 = numpy.array([[fx, 0, fx * fov_x, 0], [0, fy, fy * fov_y, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return mat_camera_4x4
# ----------------------------------------------------------------------------------------------------------------------
def compose_projection_mat_4x4_GL(W, H, fov_x, fov_y):
    near, far = 0.1, 10000.0

    mat_projection = numpy.zeros((4, 4), dtype=float)

    mat_projection[0][0] = 1 / fov_x
    mat_projection[0][1] = 0.0
    mat_projection[0][2] = 0.0
    mat_projection[0][3] = 0.0

    mat_projection[1][0] = 0.0
    mat_projection[1][1] = 1 / (fov_y * H / W)
    mat_projection[1][2] = 0.0
    mat_projection[1][3] = 0.0

    mat_projection[2][0] = 0
    mat_projection[2][1] = 0
    mat_projection[2][2] = (far + near) / (near - far)
    mat_projection[2][3] = -1.0

    mat_projection[3][0] = 0.0
    mat_projection[3][1] = 0.0
    mat_projection[3][2] = 2.0 * far * near / (near - far)
    mat_projection[3][3] = 0.0
    return mat_projection
# ----------------------------------------------------------------------------------------------------------------------

def compose_projection_ortho_mat(fx, fy, scale_factor):
    P = numpy.array([[fx, 0, 0, fx / 2], [0, fy, 0, fy / 2], [0, 0, 1, 0], [0, 0, 0, 1]])
    P[:, 3] *= scale_factor[0]
    P /= scale_factor[0]
    #P = numpy.array([[fx/scale_factor[0], 0, 0, fx / 2], [0, fy/scale_factor[1], 0, fy / 2], [0, 0, 1, 0], [0, 0, 0, 1]])
    return P
# ----------------------------------------------------------------------------------------------------------------------
def decompose_into_TRK(M):
    tvec = M[:3, 3]
    L = M.copy()
    L[:3, 3] = 0
    R, K = polar(L)
    R = numpy.array(R)
    #f, X = numpy.linalg.eig(K)

    if numpy.linalg.det(R) < 0:
        R[:3, :3] = -R[:3, :3]
        K[:3, :3] = -K[:3, :3]

    rvec = rotationMatrixToEulerAngles(R[:3,:3])
    T = pyrr.matrix44.create_from_translation(tvec).T
    R = pyrr.matrix44.create_from_eulers(rvec).T

    M_check = pyrr.matrix44.multiply(T,pyrr.matrix44.multiply(R,K))

    return T,R,K
# ----------------------------------------------------------------------------------------------------------------------
def decompose_to_rvec_tvec(mat,do_flip=False):
    if mat.shape[0]==3:
        M = pyrr.matrix44.create_from_matrix33(mat.copy())
    else:
        M = mat.copy()
    S, Q, tvec = pyrr.matrix44.decompose(M)
    rvec = quaternion_to_euler(Q)
    if do_flip:
        rvec -= (math.pi, 0, 0)
        tvec *= -1
    return rvec.astype(numpy.float32),tvec.astype(numpy.float32)
# ----------------------------------------------------------------------------------------------------------------------
def decompose_model_view(M):
    S, Q, tvec_view = pyrr.matrix44.decompose(M)
    rvec_model = quaternion_to_euler(Q)
    mat_model = pyrr.matrix44.create_from_eulers(rvec_model)
    mat_view = pyrr.matrix44.create_from_translation(tvec_view)
    R = pyrr.matrix44.create_from_eulers((0,math.pi,math.pi))
    mat_view = pyrr.matrix44.multiply(mat_view, R)
    return mat_model, mat_view
# ----------------------------------------------------------------------------------------------------------------------
def decompose_model_view_to_RRTT(mat_model, mat_view):
    S, Q, tvec_model = pyrr.matrix44.decompose(mat_model)
    rvec_model = quaternion_to_euler(Q)
    S, Q, tvec_view = pyrr.matrix44.decompose(mat_view)
    rvec_view = quaternion_to_euler(Q)
    return rvec_model,tvec_model,rvec_view,tvec_view
# ----------------------------------------------------------------------------------------------------------------------
def rotationMatrixToEulerAngles(R,do_flip=False):

    Rt = numpy.transpose(R)
    shouldBeIdentity = numpy.dot(Rt, R)
    I = numpy.identity(3, dtype=R.dtype)
    n = numpy.linalg.norm(I - shouldBeIdentity)

    if True or (n < 1e-6):

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        if do_flip:
            x*=-1
            y*=-1

    return numpy.array([x, z, y])
#----------------------------------------------------------------------------------------------------------------------
def normalize(vec):
    res = (vec.T / numpy.sqrt(numpy.sum(vec ** 2, axis=-1))).T
    return res
#----------------------------------------------------------------------------------------------------------------------
def mat_view_to_ETU(mat):

    side    =  mat[0, :3]
    up      =  mat[1, :3]
    forward =  mat[2, :3]
    dot     =  mat[3, :3]

    side = normalize(side)
    up = normalize(up)
    forward = normalize(forward)

    eye = numpy.dot((numpy.array([side, up, -forward])),dot)
    eye *= (-1, -1, +1)
    target = -forward+eye

    return eye,target,up
#----------------------------------------------------------------------------------------------------------------------
def mat_view_to_YPR(mat):
    #yaw = math.atan(mat[2,1]/mat[11])
    #pitch = numpy.atan(-mat[3,1]/)
    rvec = rotationMatrixToEulerAngles(mat[:3, :3])
    return rvec[0],rvec[1],rvec[2]
#----------------------------------------------------------------------------------------------------------------------
def ETU_to_mat_view(eye2, target2, up2):

    forward2 = numpy.array(eye2) - numpy.array(target2)
    forward2 = normalize(forward2)
    side2 = -normalize(numpy.cross(forward2, up2))
    dot2 = -numpy.array((numpy.dot(side2, eye2), numpy.dot(up2, eye2), numpy.dot(forward2, eye2)))



    mat2 = numpy.eye(4)
    mat2[0, :3] = normalize(side2)
    mat2[1, :3] = normalize(numpy.array(up2))
    mat2[2, :3] = normalize(forward2)
    mat2[3, :3] = dot2

    return mat2
# ----------------------------------------------------------------------------------------------------------------------
def quaternion_to_euler(Q):
    x, y, z, w = Q[0],Q[1],Q[2],Q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return numpy.array((X, Z, Y))*math.pi/180
#----------------------------------------------------------------------------------------------------------------------
def euler_to_quaternion(rvec):
    yaw, pitch, roll = rvec[0],rvec[1], rvec[2]
    qx = numpy.sin(roll / 2) * numpy.cos(pitch / 2) * numpy.cos(yaw / 2) - numpy.cos(roll / 2) * numpy.sin(pitch / 2) * numpy.sin(yaw / 2)
    qy = numpy.cos(roll / 2) * numpy.sin(pitch / 2) * numpy.cos(yaw / 2) + numpy.sin(roll / 2) * numpy.cos(pitch / 2) * numpy.sin(yaw / 2)
    qz = numpy.cos(roll / 2) * numpy.cos(pitch / 2) * numpy.sin(yaw / 2) - numpy.sin(roll / 2) * numpy.sin(pitch / 2) * numpy.cos(yaw / 2)
    qw = numpy.cos(roll / 2) * numpy.cos(pitch / 2) * numpy.cos(yaw / 2) + numpy.sin(roll / 2) * numpy.sin(pitch / 2) * numpy.sin(yaw / 2)
    return numpy.array((qx, qy, qz, qw))
#----------------------------------------------------------------------------------------------------------------------
def eulerAnglesToRotationMatrix(theta):

    R_x = numpy.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = numpy.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = numpy.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = numpy.dot(R_z, numpy.dot(R_y, R_x))

    return R
# ----------------------------------------------------------------------------------------------------------------------
def apply_matrix_GL(M, X):

    if M.shape[0]==3 and M.shape[1]==3:
        M = pyrr.matrix44.create_from_matrix33(M)



    X = numpy.array(X)

    if len(X.shape)==1:
        X = numpy.array([X])

    if X.shape[1]==3:
        X4D = numpy.hstack((X,numpy.full((X.shape[0],1),1)))
    else:
        X4D = X

    #Y1 = M.dot(X4D.T).T
    Y1 = X4D.dot(M.T)

    if X.shape[1] == 3:
        Y1 = Y1[:, :3]

    return Y1
# ----------------------------------------------------------------------------------------------------------------------
def apply_rotation(rvec,X):
    R = pyrr.matrix44.create_from_eulers(rvec)
    #R = pyrr.matrix44.create_from_matrix33(cv2.Rodrigues(rvec)[0])
    Y = apply_matrix_GL(R, X)
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def apply_translation(tvec,X):
    T = pyrr.matrix44.create_from_translation(tvec).T
    Y = apply_matrix_GL(T, X)
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def from_RT_GL(RT_GL):
    flip = numpy.eye(4)
    flip[1,1]=-1
    flip[2,2] = -1

    RT = numpy.matmul(flip,RT_GL.T)
    return RT
# ----------------------------------------------------------------------------------------------------------------------
def to_RT_GL(RT):
    flip = numpy.eye(4)
    flip[1,1]=-1
    flip[2,2]=-1

    RT_GL = numpy.matmul(numpy.linalg.pinv(flip),RT).T
    return RT_GL
# ----------------------------------------------------------------------------------------------------------------------
def apply_RT(rvec,tvec,X):
    #R = pyrr.matrix44.create_from_eulers(rvec) - this does not work correctily
    R = pyrr.matrix44.create_from_matrix33(cv2.Rodrigues(numpy.array(rvec,dtype=numpy.float))[0])
    T = pyrr.matrix44.create_from_translation(tvec).T
    M = pyrr.matrix44.multiply(T,R)
    Y = apply_matrix_GL(M, X)
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def compose_RT_mat(rvec,tvec,do_rodriges=False,do_flip=True,GL_style=True):
    R = pyrr.matrix44.create_from_eulers(rvec).T

    if do_rodriges:
        R = pyrr.matrix44.create_from_matrix33(cv2.Rodrigues(rvec)[0])

    T = pyrr.matrix44.create_from_translation(numpy.array(tvec)).T
    M = pyrr.matrix44.multiply(T, R)

    if GL_style:
        M = M.T

    if do_flip:
        flip = numpy.identity(4)
        flip[1][1] = -1
        flip[2][2] = -1
        M = numpy.dot(M, flip)

    return M
# ----------------------------------------------------------------------------------------------------------------------
def perspective_transform(points_2d,homography):
    method = 0

    if method==0:
        res = cv2.perspectiveTransform(points_2d, homography)

    else:
        res0 = cv2.perspectiveTransform(points_2d, homography)

        X = points_2d.reshape((-1,2))
        Y = numpy.full((X.shape[0],4),1,dtype=numpy.float)
        Y[:,:2]=X
        M = pyrr.matrix44.create_from_matrix33(homography)


        Z = apply_matrix_GL(M, Y)
        Z[:, 0] = Z[:, 0] / Z[:, 2]
        Z[:, 1] = Z[:, 1] / Z[:, 2]
        res = Z[:,:2].reshape((-1,1,2))




    return res
# ----------------------------------------------------------------------------------------------------------------------
def RT_to_H(RT, camera_matrix_3x3, Z):

    P = compose_projection_mat_4x4(camera_matrix_3x3[0,0],camera_matrix_3x3[1,1],camera_matrix_3x3[0,2]/camera_matrix_3x3[0,0],camera_matrix_3x3[1, 2] / camera_matrix_3x3[1,1])
    PM = pyrr.matrix44.multiply(P, RT)
    h_3x3 = PM[:3, :3]
    h_3x3[:,2]*=Z

    #check
    #points_3d = numpy.array((0,0+50.0,Z))
    #points_2d_v1, _ = project_points(points_3d, rvec, tvec, camera_matrix_3x3, numpy.zeros(5))
    #points_2d_v2    = project_points_M(points_3d, M0, camera_matrix_3x3, numpy.zeros(5))


    #source_2d = points_3d[:2].reshape((-1,1,2))
    #points_2d_check = perspective_transform(source_2d, h_3x3)

    return h_3x3
# ----------------------------------------------------------------------------------------------------------------------
def H_to_RT(homography,camera_matrix_3x3):

    ps = [[459. ,  390.  ],[641. ,  390.  ],[550. ,    0.  ], [550.  , 299.  ],[550.   ,481.  ], [550. ,  790.  ], [370.27 ,  0.  ], [729.72 ,  0.  ], [365.67 ,790.  ], [734.33, 790.  ]]
    pt = [[ 435.,  379.  ],[ 818.,  380.  ],[630.1 ,  277. ],[ 630.52, 340.01],[631.06,  420.  ],[ 632.88,  689.99],[ 370.1 ,  277.  ],[ 879.51,  277.  ],[-157.48,  688.75],[1385.65,  691.17]]

    ps = numpy.array(ps,dtype=numpy.float32)
    pt = numpy.array(pt)
    pt2 = perspective_transform(ps.reshape((-1, 1, 2)),homography)




    P = compose_projection_mat_4x4(camera_matrix_3x3[0, 0], camera_matrix_3x3[1, 1],
                                   camera_matrix_3x3[0, 2] / camera_matrix_3x3[0, 0],
                                   camera_matrix_3x3[1, 2] / camera_matrix_3x3[1, 1])

    XXX = numpy.eye(4)
    XXX[:3, [0, 1, 3]] = homography
    RT = numpy.dot(numpy.linalg.inv(P),XXX)


    ps_3d = numpy.insert(ps, 2, numpy.zeros((len(ps))), axis=1)
    #pt3 = project_points_M(ps_3d, RT, camera_matrix_3x3, numpy.zeros(5))

    #r_vec, t_vec, pt4 = fit_pnp(ps_3d,pt,camera_matrix_3x3)


    #n, R, T, normal = cv2.decomposeHomographyMat(homography, camera_matrix_3x3)
    #R = numpy.array(R[0])
    #T = numpy.array(T[0])
    #normal = numpy.array(normal[0])
    #homography3 = tools_calibrate.compose_homography(R, T, normal, camera_matrix_3x3)

    #HH = R + numpy.dot(T, normal.T)
    #M = numpy.dot(HH, numpy.linalg.inv(K))
    #rvec = rotationMatrixToEulerAngles(R)
    #M = compose_RT_mat(rvec,T.flatten(),do_flip=False,do_rodriges=True).T


    return RT
# ----------------------------------------------------------------------------------------------------------------------
def project_points_MVP(points_3d, img, mat_projection, mat_view, mat_model, mat_trns):

    # M = pyrr.matrix44.multiply(mat_projection, pyrr.matrix44.multiply(mat_view, pyrr.matrix44.multiply(mat_model, mat_trns)))
    # points_2d = apply_matrix(M, points_3d)
    # points_2d[:, 0] = points_2d[:, 0] / points_2d[:, 2]
    # points_2d[:, 1] = points_2d[:, 1] / points_2d[:, 2]
    # points_2d = numpy.array(points_2d)[:, :2].reshape((-1, 2))

    camera_matrix_3x3 = compose_projection_mat_3x3(img.shape[1], img.shape[0], 1 / mat_projection[0][0],1 / mat_projection[1][1])

    M = pyrr.matrix44.multiply(mat_view.T,pyrr.matrix44.multiply(mat_model.T,mat_trns.T))
    X4D = numpy.full((points_3d.shape[0], 4), 1,dtype=float)
    X4D[:, :3] = points_3d[:,:]
    L3D = pyrr.matrix44.multiply(M, X4D.T).T[:,:3]

    points_2d, jac = project_points(L3D, numpy.array((0, 0, 0)), numpy.array((0, 0, 0)),camera_matrix_3x3, numpy.zeros(4))
    points_2d = points_2d.reshape((-1, 2))
    points_2d[:, 0] = img.shape[1] - points_2d[:, 0]

    return points_2d
# ----------------------------------------------------------------------------------------------------------------------
def project_points_MVP_GL(points_3d, img, mat_projection, mat_view, mat_model, mat_trns):

    camera_matrix_3x3 = compose_projection_mat_3x3(img.shape[1], img.shape[0], 1 / mat_projection[0][0],1 / mat_projection[1][1])

    M0 = pyrr.matrix44.multiply(mat_view.T,pyrr.matrix44.multiply(mat_model.T,mat_trns.T))
    X4D = numpy.hstack((points_3d,numpy.full((len(points_3d),1),1)))
    L3D = pyrr.matrix44.multiply(M0, X4D.T).T[:,:3]

    #opencv equivalent
    #points_2d, jac = cv2.projectPoints(           L3D, numpy.array((0, 0, 0)), numpy.array((0, 0, 0)), camera_matrix_3x3, numpy.zeros(4,dtype=float))

    points_2d, jac = project_points(L3D, numpy.array((0, 0, 0)), numpy.array((0, 0, 0)), camera_matrix_3x3, numpy.zeros(4))
    points_2d = points_2d.reshape((-1,2))
    points_2d[:, 0] = img.shape[1] - points_2d[:, 0]

    return points_2d
# ----------------------------------------------------------------------------------------------------------------------

def project_points(points_3d, rvec, tvec, camera_matrix_3x3, dist=numpy.zeros(5)):
    #https: // docs.opencv.org / 2.4 / modules / calib3d / doc / camera_calibration_and_3d_reconstruction.html
    P = compose_projection_mat_4x4(camera_matrix_3x3[0,0],camera_matrix_3x3[1,1],camera_matrix_3x3[0,2]/camera_matrix_3x3[0,0],camera_matrix_3x3[1, 2] / camera_matrix_3x3[1,1])

    if len(rvec.shape)==2 and rvec.shape[0]==3 and rvec.shape[1]==3:
        R = pyrr.matrix44.create_from_matrix33(rvec)
    else:
        R = pyrr.matrix44.create_from_matrix33(cv2.Rodrigues(numpy.array(rvec, dtype=float).flatten())[0])

    T = pyrr.matrix44.create_from_translation(numpy.array(tvec,dtype=float).flatten()).T
    M = pyrr.matrix44.multiply(T, R)

    #!!!
    #M = compose_RT_mat(rvec, tvec, do_rodriges=False, do_flip=False, GL_style=False)

    PM = pyrr.matrix44.multiply(P, M)
    points_2d = project_points_p3x4(points_3d, PM, check_reverce=False)

    return points_2d,0
# ----------------------------------------------------------------------------------------------------------------------
def project_points_p3x4(points_3d, p4x4,check_reverce=False):

    points_2d = apply_matrix_GL(p4x4, points_3d)

    Z = points_2d[:, 2]

    points_2d[:, 0] = points_2d[:, 0] / points_2d[:, 2]
    points_2d[:, 1] = points_2d[:, 1] / points_2d[:, 2]
    points_2d = numpy.array(points_2d)[:, :2].reshape((-1, 2))

    if check_reverce:
        points_2d[:,0][Z<0] = numpy.nan
        points_2d[:,1][Z<0] = numpy.nan

    return points_2d
# ----------------------------------------------------------------------------------------------------------------------
def reverce_project_points_Z0(points_2d, rvec, tvec, camera_matrix_3x3, dist=None):
    P = compose_projection_mat_4x4(camera_matrix_3x3[0,0],camera_matrix_3x3[1,1],
                                   camera_matrix_3x3[0,2]/camera_matrix_3x3[0,0],camera_matrix_3x3[1, 2] / camera_matrix_3x3[1,1])

    M = compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=False, GL_style=False)
    PM = pyrr.matrix44.multiply(P, M)

    points_3d = reverce_project_points_Z0_M(points_2d, PM)

    return points_3d
# ----------------------------------------------------------------------------------------------------------------------
def reverce_project_points_Z0_M(points_2d, PM):


    points_3d = []
    for (i, j) in points_2d:
        A = numpy.array((((PM[0, 0] - i * PM[2, 0]), (PM[0, 1] - i * PM[2, 1])),((PM[1, 0] - j * PM[2, 0]), (PM[1, 1] - j * PM[2, 1]))))
        iA = numpy.linalg.inv(A)
        b = numpy.array(((i * PM[2, 3] - PM[0, 3]), (j * PM[2, 3] - PM[1, 3])))
        xx = iA.dot(b)
        points_3d.append((xx[0], xx[1], 0))

    points_3d = numpy.array(points_3d)

    return points_3d
# ----------------------------------------------------------------------------------------------------------------------
def reverce_project_points(points_2d, rvec, tvec, camera_matrix_3x3,planeNormal, planePoint):
    p0 = reverce_project_points_Z0(points_2d, rvec, tvec, camera_matrix_3x3, dist=None)
    points_3d = [tools_render_CV.line_plane_intersection(planeNormal, planePoint, rvec, p) for p in p0]

    return points_3d
# ----------------------------------------------------------------------------------------------------------------------
def project_points_M(points_3d, RT, camera_matrix_3x3, dist=numpy.zeros(5),do_flip=False):
    # RT should be Transformed !!!
    if camera_matrix_3x3.shape[0]==3 and camera_matrix_3x3.shape[1]==3:
        P = compose_projection_mat_4x4(camera_matrix_3x3[0,0],camera_matrix_3x3[1,1],camera_matrix_3x3[0,2]/camera_matrix_3x3[0,0],camera_matrix_3x3[1, 2] / camera_matrix_3x3[1,1])
    else:
        P = camera_matrix_3x3.copy()
    #
    #
    # PM = pyrr.matrix44.multiply(P, RT)
    #
    # #points_2d = apply_matrix_GL(PM, points_3d)
    #
    # points_3d_t = apply_matrix_GL(RT, points_3d)
    # points_2d = apply_matrix_GL(P, points_3d_t)
    #
    #
    # points_2d[:, 0] = points_2d[:, 0] / points_2d[:, 2]
    # points_2d[:, 1] = points_2d[:, 1] / points_2d[:, 2]
    # points_2d = numpy.array(points_2d,dtype=numpy.float32)[:, :2].reshape((-1,2))
    #
    # if do_flip:
    #     points_2d[:, 0] = camera_matrix_3x3[0,0] - points_2d[:, 0]


    PM = pyrr.matrix44.multiply(P, RT)

    # points_2d = apply_matrix_GL(PM, points_3d)
    # points_2d[:, 0] = points_2d[:, 0] / points_2d[:, 2]
    # points_2d[:, 1] = points_2d[:, 1] / points_2d[:, 2]
    # points_2d = numpy.array(points_2d)[:, :2].reshape((-1, 2))
    points_2d = project_points_p3x4(points_3d, PM, check_reverce=False)

    return points_2d
# ----------------------------------------------------------------------------------------------------------------------
def project_points_ortho(points_3d, rvec, tvec, dist,fx,fy,scale_factor):

    R = pyrr.matrix44.create_from_eulers(rvec).T
    T = pyrr.matrix44.create_from_translation(tvec).T
    RT = pyrr.matrix44.multiply(T, R)
    P = compose_projection_ortho_mat(fx, fy, scale_factor)

    PRTS = pyrr.matrix44.multiply(P,RT)
    X4D = numpy.full((points_3d.shape[0],4),1,dtype=numpy.float)
    X4D[:,:3]=points_3d
    points_2d = pyrr.matrix44.multiply(PRTS,X4D.T).T
    points_2d = numpy.array(points_2d)[:, :2].reshape(-1, 1, 2)

    return points_2d ,0
# ----------------------------------------------------------------------------------------------------------------------
def project_points_ortho_modelview(points_3d, modelview, dist,fx,fy,scale_factor):

    P = compose_projection_ortho_mat(fx, fy, scale_factor)

    PRT = pyrr.matrix44.multiply(P,modelview)
    X4D = numpy.full((points_3d.shape[0],4),1,dtype=numpy.float)
    X4D[:,:3]=points_3d
    points_2d = pyrr.matrix44.multiply(PRT,X4D.T).T

    points_2d = numpy.array(points_2d)[:, :2].reshape(-1, 1, 2)

    return points_2d ,0
# ----------------------------------------------------------------------------------------------------------------------
def derive_dims(cuboid_3d):
    idx_h = 1
    idx_lw = [0,2]

    h = numpy.abs(cuboid_3d[[1,3,5,7], idx_h].mean()-cuboid_3d[[0,2,4,6], idx_h].mean())
    ps = cuboid_3d[[1,3,5,7],:]
    ps = ps[:,idx_lw]

    d = numpy.array([numpy.linalg.norm(ps[i]-ps[j]) for i,j in zip([0,1,2,3],[1,2,3,0])])
    w = min(d[[0,2]].mean(), d[[1,3]].mean())
    l = max(d[[0,2]].mean(), d[[1,3]].mean())

    return  h,w,l
# ----------------------------------------------------------------------------------------------------------------------
def fit_1M(cuboid):

    h, w, l = 1, 1, 1
    #h, w, l = derive_dims(cuboid)

    cube_1M = numpy.array([
        [i - l/2 for i in [l,l,l,l,0,0,0,0]],
        [i - h/1 for i in [0,h,0,h,0,h,0,h]],
        [i - w/2 for i in [0,0,w,w,w,w,0,0]]],dtype=numpy.float32).T

    hh = cuboid[:,1]

    #cube_1M+=numpy.mean(cuboid[[1,3,5,7]],axis=0)
    cube_1M+=numpy.mean(cuboid[:],axis=0)

    cube_1M[:,1] = cuboid[:,1]

    return cube_1M
# ----------------------------------------------------------------------------------------------------------------------
def fit_3D_3D(cuboid,idx_h = 1,idx_wh = [0,2],do_debug=False):

    idx_h0 = numpy.where(cuboid[:,idx_h]> cuboid[:,idx_h].mean())[0]
    idx_h1 = numpy.where(cuboid[:,idx_h]<=cuboid[:,idx_h].mean())[0]

    rect = cuboid[:, idx_wh]
    rect_h0 = rect[[idx_h0]]
    rect_h1 = rect[[idx_h1]]


    rect_h0_reg = tools_render_CV.regularize_rect(rect_h0,inlined=False, do_debug=do_debug)
    rect_h1_reg = tools_render_CV.regularize_rect(rect_h1,inlined=False, do_debug=do_debug)
    cuboid2 = cuboid.copy()

    for i,idx in enumerate(idx_h0):
        cuboid2[idx,idx_wh] = rect_h0_reg[i]
        cuboid2[idx,idx_h ] = cuboid[idx_h0,idx_h].mean()

    for i,idx in enumerate(idx_h1):
        cuboid2[idx,idx_wh] = rect_h1_reg[i]
        cuboid2[idx,idx_h ] = cuboid[idx_h1,idx_h].mean()


    return cuboid2
# ----------------------------------------------------------------------------------------------------------------------
def multiply(list_of_M):

    res = list_of_M[-1].copy()
    for M in reversed(list_of_M[:-1]):
        res = numpy.dot(M,res.copy())
    return res
# ----------------------------------------------------------------------------------------------------------------------
def perpendicular_vector3d(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            return numpy.array((0,0,0))
        else:
            return numpy.cross(v, [0, 1, 0])
    return numpy.cross(v, [1, 0, 0])
# ----------------------------------------------------------------------------------------------------------------------