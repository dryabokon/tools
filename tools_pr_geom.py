import math
import cv2
import pyrr
import numpy
from scipy.linalg import polar
from zhou_accv_2018 import p3l,p3p
from skimage.transform import ProjectiveTransform
# ---------------------------------------------------------------------------------------------------------------------
def debug_projection(X_source, X_target,result,colors=None,prefix='_'):
    if colors is None:colors = [(int(128),int(128),int(128))]*len(X_source)

    padding = 10

    xmin = int(min(X_source[:,0].min(), X_target[:,0].min(),result[:,0].min()))-padding
    xmax = int(max(X_source[:,0].max(), X_target[:,0].max(),result[:,0].max()))+padding
    ymin = int(min(X_source[:,1].min(), X_target[:,1].min(),result[:,1].min()))-padding
    ymax = int(max(X_source[:,1].max(), X_target[:,1].max(),result[:,1].max()))+padding

    image = numpy.full((ymax-ymin+1,xmax-xmin+1,3),32,dtype=numpy.uint8)
    for i,(x,y) in enumerate(X_source):
        cv2.circle(image, (int(x-xmin),int(y-ymin)), 6, colors[i].tolist(), -1)
        cv2.putText(image, '{0}'.format(i), (int(x-xmin),int(y-ymin)), cv2.FONT_HERSHEY_SIMPLEX,0.6, colors[i].tolist(), 1, cv2.LINE_AA)

    for i,(x,y) in enumerate(X_target):cv2.circle(image, (int(x-xmin),int(y-ymin)),16,(int(colors[i][0]),int(colors[i][1]),int(colors[i][2])),  3)
    for i,(x,y) in enumerate(result)  :cv2.circle(image, (int(x-xmin),int(y-ymin)), 6,(int(colors[i][0]),int(colors[i][1]),int(colors[i][2])), -1)

    cv2.imwrite('./images/output/fit_%02d.png'%prefix,image)
    return
# ----------------------------------------------------------------------------------------------------------------------
def fit_homography(X_source,X_target,colors=None,do_debug=None):
    method = cv2.RANSAC
    #method = cv2.LMEDS
    #method = cv2.RHO
    H, mask = cv2.findHomography(X_source, X_target, method, 3.0)

    result  = cv2.perspectiveTransform(X_source.reshape(-1, 1, 2),H).reshape((-1,2))
    loss =  ((result-X_target)**2).mean()
    if do_debug: debug_projection(X_source, X_target, result, colors,prefix=do_debug)

    return H, result
# ----------------------------------------------------------------------------------------------------------------------
def fit_affine(X_source,X_target,colors=None,do_debug=False):
    A, _ = cv2.estimateAffine2D(numpy.array(X_source), numpy.array(X_target), confidence=0.95)
    result = cv2.transform(X_source.reshape(-1, 1, 2), A).reshape((-1,2))
    loss = ((result - X_target) ** 2).mean()
    if do_debug: debug_projection(X_source, X_target, result, colors)
    return A, result
# ----------------------------------------------------------------------------------------------------------------------
def fit_euclid(X_source,X_target,colors=None,do_debug=False):
    E, _ = cv2.estimateAffinePartial2D(numpy.array(X_source), numpy.array(X_target))
    result = cv2.transform(X_source.reshape(-1, 1, 2), E).reshape((-1,2))
    loss = ((result - X_target) ** 2).mean()
    if do_debug:debug_projection(X_source, X_target,result,colors)

    return E, result
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
def fit_pnp(landmarks_3d,landmarks_2d,mat_camera_3x3):
    (_, r_vec, t_vec) = cv2.solvePnP(landmarks_3d, landmarks_2d,mat_camera_3x3, numpy.zeros(4))
    landmarks_2d_check, jac = cv2.projectPoints(landmarks_3d, r_vec, t_vec, mat_camera_3x3, numpy.zeros(4))
    landmarks_2d_check = numpy.reshape(landmarks_2d_check, (-1, 2))

    return r_vec, t_vec, landmarks_2d_check
# ----------------------------------------------------------------------------------------------------------------------
def fit_p3l(lines_3d,lines_2d,mat_camera_3x3):

    points3d_start = lines_3d[:, :3]
    points3d_end   = lines_3d[:, 3:]

    points3d = (points3d_start + points3d_end ) / 2
    directions3d = (points3d_end  - points3d_start)
    norms = numpy.linalg.norm(directions3d, axis=1)[:, None]
    directions3d = directions3d / norms

    directions3d*=numpy.array([1,-1,1])[:,None]

    poses = p3l(lines_2d.reshape((3,2,2)), (points3d, directions3d), mat_camera_3x3)
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
def compose_projection_mat_3x3(fx, fy, aperture_x=0.5,aperture_y=0.5):
    mat_camera = numpy.array([[fx, 0, 0.5*fx*(aperture_x/0.5)], [0, fy, 0.5*fy*(aperture_y/0.5)], [0, 0, (0.5*(aperture_x+aperture_y)/0.5)]],dtype=numpy.float)
    return mat_camera
# ----------------------------------------------------------------------------------------------------------------------
def compose_projection_mat_4x4(fx, fy, aperture_x=0.5,aperture_y=0.5):
    mat_camera = numpy.array([[fx, 0, 0.5*fx*(aperture_x/0.5),0], [0, fy, 0.5*fy*(aperture_y/0.5),0],[0,0,1,0],[0,0,0,1]])
    return mat_camera
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
    return rvec,tvec
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
def mat_view_to_ETU(mat):

    side    =  mat[0, :3]
    up      =  mat[1, :3]
    forward =  mat[2, :3]
    dot     = mat[3, :3]

    eye_new = numpy.dot(numpy.linalg.inv(numpy.array([side, up, -forward])),dot)
    eye_new *= (-1, -1, -1)

    yyy=0

    return eye_new,eye_new + forward,up
#----------------------------------------------------------------------------------------------------------------------
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
def apply_matrix(M,X):

    if len(X.shape)==1:
        X = numpy.array([X])

    X = numpy.array(X)
    if len(X.shape)==1:
        X=numpy.array([X])


    if X.shape[1]==3:
        X4D = numpy.full((X.shape[0], 4), 1,dtype=numpy.float)
        X4D[:, :3] = X
    else:
        X4D = X

    Y1 = pyrr.matrix44.multiply(M, X4D.T).T
    Y2 = numpy.array([pyrr.matrix44.apply_to_vector(M.T, x) for x in X4D])

    #if X.shape[0]==1:
    #    Y1=Y1[0]

    return Y1
# ----------------------------------------------------------------------------------------------------------------------
def apply_rotation(rvec,X):
    R = pyrr.matrix44.create_from_eulers(rvec)
    #R = pyrr.matrix44.create_from_matrix33(cv2.Rodrigues(rvec)[0])
    Y = apply_matrix(R, X)
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def apply_translation(tvec,X):
    T = pyrr.matrix44.create_from_translation(tvec).T
    Y = apply_matrix(T,X)
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def apply_RT(rvec,tvec,X):
    #R = pyrr.matrix44.create_from_eulers(rvec) - this does not work correctily
    R = pyrr.matrix44.create_from_matrix33(cv2.Rodrigues(numpy.array(rvec,dtype=numpy.float))[0])
    T = pyrr.matrix44.create_from_translation(tvec).T
    M = pyrr.matrix44.multiply(T,R)
    Y = apply_matrix(M,X)
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def compose_RT_mat(rvec,tvec,do_flip=True,do_rodriges=False):
    R = pyrr.matrix44.create_from_eulers(rvec).T
    if do_rodriges: R = pyrr.matrix44.create_from_matrix33(cv2.Rodrigues(rvec)[0])
    T = pyrr.matrix44.create_from_translation(numpy.array(tvec)).T
    M = pyrr.matrix44.multiply(T, R).T
    flip = numpy.identity(4)
    if do_flip:
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

        Z0 = apply_matrix(M, Y)
        Z = 100*apply_matrix(M/100, Y)
        yy=0
        Z[:, 0] = Z[:, 0] / Z[:, 2]
        Z[:, 1] = Z[:, 1] / Z[:, 2]
        res = Z[:,:2].reshape((-1,1,2))




    return res
# ----------------------------------------------------------------------------------------------------------------------
def project_points(points_3d, rvec, tvec, camera_matrix_3x3, dist):
    #https: // docs.opencv.org / 2.4 / modules / calib3d / doc / camera_calibration_and_3d_reconstruction.html

    P = compose_projection_mat_4x4(camera_matrix_3x3[0,0],camera_matrix_3x3[1,1],camera_matrix_3x3[0,2]/camera_matrix_3x3[0,0],camera_matrix_3x3[1, 2] / camera_matrix_3x3[1,1])

    if len(rvec.shape)==2 and rvec.shape[0]==3 and rvec.shape[1]==3:
        R = pyrr.matrix44.create_from_matrix33(rvec)
    else:
        R = pyrr.matrix44.create_from_matrix33(cv2.Rodrigues(numpy.array(rvec, dtype=float).flatten())[0])

    T = pyrr.matrix44.create_from_translation(numpy.array(tvec,dtype=float).flatten()).T
    M = pyrr.matrix44.multiply(T, R)
    PM = pyrr.matrix44.multiply(P, M)

    points_2d = apply_matrix(PM, points_3d)
    points_2d[:, 0] = points_2d[:, 0] / points_2d[:, 2]
    points_2d[:, 1] = points_2d[:, 1] / points_2d[:, 2]
    points_2d = numpy.array(points_2d)[:, :2].reshape(-1, 1, 2)

    return points_2d,0
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
