import numpy
import cv2
import tools_image
import dlib
import tools_draw_numpy
from scipy.spatial import Delaunay
from scipy import ndimage
import math
import tools_GL
# --------------------------------------------------------------------------------------------------------------------
class detector_landmarks(object):
    def __init__(self,filename_config):
        self.name = "landmark_detector"
        self.idx_head = numpy.arange(0,27,1).tolist()
        self.idx_nose = numpy.arange(27, 36, 1).tolist()
        self.idx_eyes = numpy.arange(36, 48, 1).tolist()
        self.idx_mouth = numpy.arange(48, 68, 1).tolist()
        self.idx_removed_eyes = numpy.arange(0,68,1).tolist()
        for each in [37,38,40,41,43,44,46,47]:
            self.idx_removed_eyes.remove(each)

        self.model_68_points = self.__get_full_model_points()
        self.r_vec = -numpy.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = -numpy.array([[-14.97821226], [-10.62040383], [-2053.03596872]])


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
        '''

        raw_value = [-0.95, -0.39, -0.62,-0.94, -0.14, -0.60,-0.92, 0.10, -0.58,-0.87, 0.34, -0.56,
                     -0.78, 0.55, -0.50,-0.63, 0.73, -0.40,-0.44, 0.87, -0.24,-0.23, 0.97, -0.05,
                     0.00, 1.00, 0.01,0.23, 0.97, -0.07,0.42, 0.87, -0.25,0.60, 0.73, -0.40,
                     0.74, 0.55, -0.49,0.84, 0.33, -0.53,0.89, 0.09, -0.55,0.91, -0.15, -0.57,
                     0.93, -0.39, -0.61,-0.79, -0.64, -0.18,-0.67, -0.76, -0.09,-0.49, -0.80, -0.01,
                     -0.31, -0.79, 0.09,-0.15, -0.74, 0.16,0.16, -0.74, 0.16,0.33, -0.80, 0.09,
                     0.50, -0.81, 0.01,0.66, -0.77, -0.07,0.78, -0.65, -0.15,0.01, -0.55, 0.17,
                     0.01, -0.40, 0.27,0.01, -0.26, 0.38,0.02, -0.11, 0.48,-0.19, 0.03, 0.26,
                     -0.09, 0.06, 0.31,0.01, 0.09, 0.34,0.11, 0.06, 0.31,0.20, 0.03, 0.27,
                     -0.60, -0.49, -0.09,-0.49, -0.55, -0.04,-0.36, -0.55, -0.02,-0.25, -0.48, 0.00,
                     -0.37, -0.46, 0.00,-0.49, -0.45, -0.02,0.25, -0.48, 0.01,0.36, -0.56, -0.00,
                     0.49, -0.56, -0.02,0.59, -0.49, -0.06,0.50, -0.46, -0.00,0.37, -0.46, 0.02,
                     -0.38, 0.37, 0.03,-0.23, 0.29, 0.21,-0.09, 0.25, 0.29,0.00, 0.27, 0.31,
                     0.11, 0.25, 0.29,0.25, 0.29, 0.20,0.37, 0.36, 0.04,0.25, 0.47, 0.19,
                     0.12, 0.51, 0.29,0.00, 0.52, 0.31,-0.10, 0.52, 0.29,-0.24, 0.48, 0.20,
                     -0.32, 0.37, 0.06,-0.09, 0.33, 0.27,0.00, 0.34, 0.29,0.11, 0.33, 0.27,
                     0.32, 0.37, 0.07,0.11, 0.40, 0.27,0.00, 0.41, 0.28,-0.09, 0.40, 0.26]
        model_points = numpy.array(raw_value, dtype=numpy.float32)
        model_points = numpy.reshape(model_points, (-1, 3))
        #model_points[:, -1] *= -1
        '''

        return model_points
# --------------------------------------------------------------------------------------------------------------------



    def __get_full_model_points2(self):
        raw_value = [-0.643584631,-0.040729434,-0.101559903,
-0.639432597,0.101114133,-0.076983813,
-0.618077713,0.236126367,-0.080355559,
-0.563884198,0.367427906,-0.009524983,
-0.512362194,0.548558646,0.072390685,
-0.446010021,0.669333618,0.180580026,
-0.368168811,0.781071514,0.335318391,
-0.246436401,0.85220518,0.554088211,
0.000750884,0.851740315,0.689964006,
0.283335408,0.83245941,0.525202383,
0.401131976,0.75220725,0.32165884,
0.461429453,0.657084259,0.190228076,
0.520003823,0.542827348,0.090319993,
0.552643156,0.429349814,0.034266609,
0.617170021,0.224226726,-0.073645903,
0.646316519,0.043870799,-0.105484387,
0.650219809,-0.049596931,-0.116383293,
-0.490768634,-0.12469485,0.502494472,
-0.436396191,-0.193567065,0.57899154,
-0.335784044,-0.219559964,0.635261897,
-0.234386591,-0.216584956,0.646406959,
-0.141920618,-0.162991953,0.634866299,
0.142464833,-0.207721702,0.656438571,
0.236453363,-0.229359836,0.651291002,
0.303611233,-0.229282267,0.645036216,
0.38982755,-0.208278352,0.612991259,
0.4527858,-0.183981881,0.559680219,
0.001559081,-0.085671235,0.705738415,
0.00328822,-0.012221392,0.758940334,
0.000468683,0.06703081,0.810804683,
0.000153977,0.200680008,0.89014586,
-0.147049671,0.274609082,0.714186113,
-0.070844847,0.310495337,0.741682773,
-0.002646859,0.322703355,0.75739363,
0.093584604,0.296845698,0.738283981,
0.160130493,0.251374995,0.685388319,
-0.419948884,-0.070532366,0.533320788,
-0.366138292,-0.130714642,0.59293797,
-0.260078768,-0.129682115,0.613466895,
-0.162194008,-0.079581435,0.571885867,
-0.24118086,-0.04455248,0.597086784,
-0.354189773,-0.036853228,0.594922331,
0.163021952,-0.077893294,0.572174543,
0.245955898,-0.132864623,0.608834564,
0.346593715,-0.130632262,0.60405634,
0.420784135,-0.066053214,0.533418157,
0.360664685,-0.038750861,0.592266217,
0.25411183,-0.03890367,0.601305662,
-0.233850324,0.50130218,0.628058351,
-0.159403973,0.455185731,0.729775415,
-0.063899251,0.437224744,0.766522973,
0.000793804,0.467728381,0.764124757,
0.078695941,0.430111073,0.761131488,
0.164285609,0.459550425,0.727353498,
0.223445308,0.512631735,0.623646986,
0.163439246,0.571651073,0.701105525,
0.080313852,0.581240241,0.735043963,
-0.007032462,0.588471144,0.734582905,
-0.107203395,0.581553165,0.725262788,
-0.181796761,0.563640318,0.686808737,
-0.185952355,0.514901006,0.662936261,
-0.084193038,0.509244982,0.71862904,
0.013149973,0.500269438,0.746837464,
0.108128958,0.489355145,0.735592518,
0.194313857,0.489259519,0.687999621,
0.119592139,0.529072828,0.721016097,
0.019422827,0.541051157,0.74359893,
-0.079467253,0.531557559,0.7364661]
        model_points = numpy.array(raw_value, dtype=numpy.float32)
        model_points = numpy.reshape(model_points, (-1, 3))
        model_points[:, -1] *= -1
        # model_points = model_points[:,[0,2,1]]
        return model_points
# --------------------------------------------------------------------------------------------------------------------

    def get_pose(self, image,landmarks_2d, landmarks_3d=None):

        fx, fy = float(image.shape[1]), float(image.shape[0])
        self.mat_camera = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])

        dist_coefs = numpy.zeros((4, 1))

        if landmarks_3d is None:
            landmarks_3d = self.model_68_points

        #self.r_vec = None
        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(landmarks_3d, landmarks_2d, self.mat_camera, dist_coefs)
        else:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(landmarks_3d, landmarks_2d, self.mat_camera, dist_coefs, rvec=self.r_vec, tvec=self.t_vec, useExtrinsicGuess=True)
        self.r_vec = rotation_vector
        self.t_vec = translation_vector

        #test
        landmarks_2d_test, _ = cv2.projectPoints(landmarks_3d, rotation_vector, translation_vector, self.mat_camera, dist_coefs)

        err = numpy.sqrt(((landmarks_2d_test-landmarks_2d)**2).mean())
        tol = 0.1*landmarks_2d.mean()

        if err>tol:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(landmarks_3d, landmarks_2d, self.mat_camera, dist_coefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

            landmarks_2d_test, _ = cv2.projectPoints(landmarks_3d, rotation_vector, translation_vector, self.mat_camera,dist_coefs)
            err = numpy.sqrt(((landmarks_2d_test - landmarks_2d) ** 2).mean())

        return rotation_vector, translation_vector
# --------------------------------------------------------------------------------------------------------------------
    def draw_annotation_box(self,image, rotation_vector, translation_vector, color=(0, 128, 255), line_width=1):

        point_3d = []
        rear_size = 75
        rear_depth = 0
        front_size = 100
        front_depth = 100

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
