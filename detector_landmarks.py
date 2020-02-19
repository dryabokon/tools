import numpy
import cv2
import tools_image
import dlib
import tools_draw_numpy
from scipy.spatial import Delaunay
from scipy import ndimage
import math
import tools_GL
import tools_IO
import pyrr
# --------------------------------------------------------------------------------------------------------------------
class detector_landmarks(object):
    def __init__(self,filename_config,filename_3dmarkers=None):
        self.name = "landmark_detector"
        self.idx_head = numpy.arange(0,27,1).tolist()
        self.idx_nose = numpy.arange(27, 36, 1).tolist()
        self.idx_eyes = numpy.arange(36, 48, 1).tolist()
        self.idx_mouth = numpy.arange(48, 68, 1).tolist()
        self.idx_removed_eyes = numpy.arange(0,68,1).tolist()
        for each in [37,38,40,41,43,44,46,47]:
            self.idx_removed_eyes.remove(each)

        self.model_68_points = self.__get_full_model_points()
        if filename_3dmarkers is not None:
            self.model_68_points = tools_IO.load_mat(filename_3dmarkers,dtype=numpy.float, delim=',')

        self.r_vec = None
        self.t_vec = None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(filename_config)
        return
# --------------------------------------------------------------------------------------------------------------------
    def detect_face(self,image):
        gray = tools_image.desaturate(image)
        objects = self.detector(gray)
        if len(objects) == 1:
            for face in objects:
                xmin = face.left()
                ymin = face.top()
                xmax = face.right()
                ymax = face.bottom()
                return image[ymin:ymax, xmin:xmax, :]

        return None
# ----------------------------------------------------------------------------------------------------------------------
    def draw_face(self,image):
        gray = tools_image.desaturate(image)
        objects = self.detector(gray)
        if len(objects) == 1:
            for face in objects:
                xmin = face.left()
                ymin = face.top()
                xmax = face.right()
                ymax = face.bottom()
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_landmarks(self,image):

        gray = tools_image.desaturate(image)
        landmarks = self.get_landmarks(image)
        if len(landmarks)!=68 or numpy.sum(landmarks)==0:
            return gray

        del_triangles = Delaunay(landmarks).vertices
        for landmark in landmarks:
            cv2.circle(gray,(int(landmark[0]),int(landmark[1])),2,(0,128,255),-1)
            #gray=tools_draw_numpy.draw_circle(gray, landmark[1],landmark[0], 3, (0,128,255),alpha_transp=0.7)


        for t in del_triangles:
            p0 = (int(landmarks[t[0],0]), int(landmarks[t[0],1]))
            p1 = (int(landmarks[t[1],0]), int(landmarks[t[1],1]))
            p2 = (int(landmarks[t[2],0]), int(landmarks[t[2],1]))
            cv2.line(gray,p0,p1,(0,0,255))
            cv2.line(gray,p0,p2,(0, 0, 255))
            cv2.line(gray,p2,p1,(0, 0, 255))
            #gray = tools_draw_numpy.draw_line(gray, p0[1], p0[0], p1[1], p1[0], (0, 0, 255), alpha_transp=0.7)
            #gray = tools_draw_numpy.draw_line(gray, p0[1], p0[0], p2[1], p2[0], (0, 0, 255), alpha_transp=0.7)
            #gray = tools_draw_numpy.draw_line(gray, p2[1], p2[0], p1[1], p1[0], (0, 0, 255), alpha_transp=0.7)


        return gray
# ----------------------------------------------------------------------------------------------------------------------
    def draw_landmarks_v2(self, image,landmarks,del_triangles=None,color=(0, 128, 255)):

        gray = tools_image.desaturate(image,level=0)

        for landmark in landmarks:
            cv2.circle(gray, (int(landmark[0]), int(landmark[1])), 2, color, -1)

        if del_triangles is not None:
            for t in del_triangles:
                p0 = (int(landmarks[t[0], 0]), int(landmarks[t[0], 1]))
                p1 = (int(landmarks[t[1], 0]), int(landmarks[t[1], 1]))
                p2 = (int(landmarks[t[2], 0]), int(landmarks[t[2], 1]))
                cv2.line(gray, p0, p1, (0, 0, 255))
                cv2.line(gray, p0, p2, (0, 0, 255))
                cv2.line(gray, p2, p1, (0, 0, 255))

        return gray
# ----------------------------------------------------------------------------------------------------------------------
    def get_landmarks(self, image):
        res = numpy.zeros( (68,2), dtype=numpy.float)
        if image is None:
            return res
        gray = tools_image.desaturate(image)
        objects = self.detector(gray)

        if len(objects) == 1:
            landmarks = self.predictor(gray, objects[0])
            res=[]
            for n in range(0, 68):
                res.append([landmarks.part(n).x, landmarks.part(n).y])
            res = numpy.array(res)

        return res.astype(numpy.float)
# ----------------------------------------------------------------------------------------------------------------------
    def are_landmarks_valid(self,landmarks):
        if (landmarks.min() == landmarks.max() == 0):
            return False

        return True
# ----------------------------------------------------------------------------------------------------------------------
    def extract_face_by_landmarks(self,image,landmarks):
        if not (landmarks.min()==landmarks.max()==0):
            rect = self.get_rect_by_landmarks(landmarks)
            top, left, bottom, right = rect[0],rect[1],rect[2],rect[3]
            res = image[top:bottom,left:right,:]
        else:
            return image
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def get_rect_by_landmarks(self,landmarks):
        top = (landmarks[:, 1].min())
        left = (landmarks[:, 0].min())
        bottom = landmarks[:, 1].max()
        right = landmarks[:, 0].max()
        return numpy.array([top, left, bottom,right])
# ----------------------------------------------------------------------------------------------------------------------
    def cut_face(self, image, L, desiredLeftEye=(0.35, 0.35), target_W=256, target_H=256):

        (lStart, lEnd) = (42,47)
        (rStart, rEnd) = (36,41)
        leftEyePts = L[lStart:lEnd]
        rightEyePts = L[rStart:rEnd]


        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = numpy.degrees(numpy.arctan2(dY, dX)) - 180

        desiredRightEyeX = 1.0 - desiredLeftEye[0]


        dist = numpy.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= target_W
        scale = desiredDist / dist

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,(leftEyeCenter[1] + rightEyeCenter[1]) // 2)


        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = target_W * 0.5
        tY = target_H * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        (w, h) = (target_W, target_H)
        output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)

        return output
# --------------------------------------------------------------------------------------------------------------------
    def __get_full_model_points(self):

        raw_value = [-73.393523,
                -72.775014, -70.533638, -66.850058, -59.790187, -48.368973, -34.121101,-17.875411, 0.098749,
                17.477031,32.648966,46.372358,57.343480,64.388482,68.212038, 70.486405,71.375822,
                -61.119406,-51.287588,-37.804800,-24.022754,-11.635713,12.056636,25.106256, 38.338588,
                51.191007, 60.053851, 0.653940, 0.804809,0.992204, 1.226783, -14.772472,-7.180239,
                0.555920, 8.272499,15.214351,-46.047290,-37.674688,-27.883856, -19.648268,-28.272965, -38.082418,
                19.265868, 27.894191,37.437529,45.170805,38.196454, 28.764989, -28.916267,-17.533194,
                -6.684590,0.381001,8.375443,18.876618,28.794412,19.057574,8.956375,0.381549,
                -7.428895, -18.160634,-24.377490, -6.897633,0.340663,8.444722, 24.474473,8.449166,0.205322,
                -7.198266,-29.801432,-10.949766,7.929818,26.074280,42.564390,56.481080, 67.246992,
                75.056892,77.061286, 74.758448, 66.929021,56.311389,42.419126,25.455880, 6.990805,
                -11.666193, -30.365191,-49.361602,-58.769795,-61.996155,-61.033399,-56.686759, -57.391033,
                -61.902186,-62.777713, -59.302347, -50.190255,-42.193790,-30.993721,-19.944596, -8.414541,
                2.598255,4.751589,6.562900, 4.661005,2.643046,-37.471411,-42.730510,-42.711517,
                -36.754742, -35.134493,-34.919043,-37.032306,-43.342445, -43.110822, -38.086515,-35.532024,
                -35.484289, 28.612716,22.172187,19.029051,20.721118, 19.035460,22.394109,28.079924,
                36.298248, 39.634575, 40.395647,39.836405, 36.677899, 28.677771,25.475976,26.014269,
                25.326198, 28.323008, 30.596216,31.408738,30.844876, 47.667532,45.909403,44.842580,
                43.141114, 38.635298, 30.750622,18.456453, 3.609035,-0.881698,5.181201, 19.176563,
                30.770570, 37.628629,40.886309,42.281449,44.142567, 47.140426, 14.254422,7.268147,
                0.442051,-6.606501,-11.967398,-12.051204, -7.315098,-1.022953,5.349435,11.615746,
                -13.380835,-21.150853,-29.284036, -36.948060,-20.132003,-23.536684, -25.944448, -23.695741,
                -20.858157,7.037989,3.021217,1.353629,-0.111088,-0.147273, 1.476612,-0.665746,
                0.247660, 1.696435,4.894163,0.282961, -1.172675,-2.240310,-15.934335,-22.611355,
                -23.748437, -22.721995,-15.610679,-3.217393,-14.987997, -22.554245, -23.591626,-22.406106,
                -15.121907, -4.785684,-20.893742,-22.220479, -21.025520, -5.712776, -20.671489,-21.903670,
                -20.328022]
        model_points = numpy.array(raw_value, dtype=numpy.float32)
        model_points = numpy.reshape(model_points, (3, -1)).T
        model_points[:, -1] *= -1


        return model_points

# --------------------------------------------------------------------------------------------------------------------
    def get_pose(self, image,landmarks_2d, landmarks_3d,mat_trns=None):

        if mat_trns is None: mat_trns = numpy.eye(4)

        fx, fy = float(image.shape[1]), float(image.shape[0])
        self.mat_camera = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])
        dist_coefs = numpy.zeros((4, 1))

        if landmarks_3d is None:L3D = self.model_68_points
        else:L3D = landmarks_3d.copy()
        L3D = numpy.array([pyrr.matrix44.apply_to_vector(mat_trns,v) for v in L3D])

        if True:#self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(L3D, landmarks_2d, self.mat_camera, dist_coefs)
        else:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(L3D, landmarks_2d, self.mat_camera, numpy.zeros(4), rvec=self.r_vec, tvec=self.t_vec, useExtrinsicGuess=True)

        self.r_vec = rotation_vector.flatten()
        self.t_vec = translation_vector.flatten()

        return self.r_vec, self.t_vec
# --------------------------------------------------------------------------------------------------------------------
    def get_mat_model(self, image, landmarks_2d, landmarks_3d, mat_trns=None, mat_view=None):

        if mat_trns is None: mat_trns = numpy.eye(4)
        if mat_view is None: mat_view = numpy.eye(4)

        fx, fy = float(image.shape[1]), float(image.shape[0])
        self.mat_camera = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])
        dist_coefs = numpy.zeros((4, 1))

        if landmarks_3d is None:L3D = self.model_68_points
        else:L3D = landmarks_3d.copy()
        L3D = numpy.array([pyrr.matrix44.apply_to_vector(mat_trns,v) for v in L3D])


        if self.r_vec is None:(_, rotation_vector, translation_vector) = cv2.solvePnP(L3D, landmarks_2d, self.mat_camera,dist_coefs)
        else:(_, rotation_vector, translation_vector) = cv2.solvePnP(L3D, landmarks_2d, self.mat_camera,dist_coefs, rvec=self.r_vec, tvec=self.t_vec,useExtrinsicGuess=True)
        self.r_vec = rotation_vector.flatten()
        self.t_vec = translation_vector.flatten()



        return mat_model
# --------------------------------------------------------------------------------------------------------------------
    def draw_annotation_box(self,image, rotation_vector, translation_vector, color=(0, 128, 255), line_width=1):

        point_3d = []

        if self.model_68_points.max()-self.model_68_points.min() > 5:
            rear_size, rear_depth, front_size, front_depth = numpy.array([75, 0, 100, 100]) * 1
        else:
            rear_size,rear_depth,front_size,front_depth = numpy.array([75,0,75,75])*0.015


        dist_coefs = numpy.zeros((4, 1))
        camera_matrix = numpy.array([[image.shape[1], 0, (image.shape[1]/2)],[0, image.shape[1], (image.shape[0]/2)],[0, 0, 1]], dtype="double")
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, +rear_size, rear_depth))
        point_3d.append((+rear_size, +rear_size, rear_depth))
        point_3d.append((+rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, +front_size, front_depth))
        point_3d.append((+front_size, +front_size, front_depth))
        point_3d.append((+front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = numpy.array(point_3d, dtype=numpy.float).reshape(-1, 3)

        (point_2d, _) = cv2.projectPoints(point_3d,rotation_vector,translation_vector,camera_matrix,dist_coefs)
        point_2d = numpy.int32(point_2d.reshape(-1, 2))

        result = image.copy()
        cv2.polylines(result, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(result, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(result, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(result, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)
        return result
# --------------------------------------------------------------------------------------------------------------------
