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

        return model_points
# --------------------------------------------------------------------------------------------------------------------

    def __get_full_model_points2(self):
        raw_value = [-0.6435846306943956,-0.04072943374418145,-0.10155990310737245,
-0.6394325968329577,0.10111413336795927,-0.07698381349802323,-0.618077712900936,0.2361263667757441,-0.08035555938010926,
-0.5638841975571798,0.3674279063528747,-0.00952498329751661,-0.5123621944292874,0.548558646131121,0.07239068484338244,
-0.44601002076425333,0.6693336181946097,0.18058002565358525,-0.3681688105378992,0.7810715144017433,0.3353183907572187,
-0.2464364006108733,0.8522051800962223,0.554088210847435,0.0007508835314467416,0.8517403154589989,0.6899640056173846,
0.2833354079128651,0.8324594102647291,0.5252023825789461,0.40113197565227904,0.7522072495493733,0.3216588399355823,
0.4614294533333237,0.6570842593730787,0.19022807578515394,0.520003822542805,0.5428273481726276,0.09031999313207179,
0.5526431556941882,0.42934981419659113,0.03426660854228627,0.6171700210742892,0.22422672647782493,-0.07364590279671002,
0.6463165193643247,0.04387079858416007,-0.1054843871697588,0.6502198088503752,-0.049596931101667116,-0.11638329317415358,
-0.4907686338256894,-0.12469484965422999,0.5024944721277029,-0.43639619066435925,-0.19356706507032134,0.578991540197562,
-0.33578404415278906,-0.21955996423034083,0.6352618967285502,-0.23438659056861633,-0.2165849563468225,0.6464069592355483,
-0.14192061758647928,-0.16299195338392097,0.6348662990217031,0.14246483286338682,-0.20772170166174844,0.6564385706568208,
0.2364533631704868,-0.22935983592817633,0.6512910021043816,0.3036112334235587,-0.2292822665426808,0.6450362164228786,
0.38982755010266495,-0.2082783515184879,0.612991258921657,0.4527857999738525,-0.18398188065693732,0.5596802189191039,
0.0015590811996136766,-0.08567123493238915,0.7057384151054282,0.0032882200829147945,-0.012221392154070275,0.7589403335191559,
0.0004686831549264303,0.06703081018044876,0.8108046828469846,0.000153976835187003,0.20068000837974226,0.8901458597509834,
-0.14704967138423755,0.27460908193277156,0.7141861133754956,-0.0708448471105979,0.31049533675242386,0.7416827727298151,
-0.0026468594993917863,0.3227033547913151,0.7573936301692297,0.09358460424049353,0.29684569836255575,0.7382839808188913,
0.16013049328357448,0.2513749948419544,0.6853883193556257,-0.41994888400646657,-0.07053236614865284,0.533320788159682,
-0.3661382917228082,-0.13071464209703387,0.5929379697178734,-0.2600787677844016,-0.12968211523162182,0.6134668954492731,
-0.16219400831666492,-0.0795814352603338,0.5718858672996282,-0.24118085971633002,-0.04455247950303412,0.5970867835548987,
-0.35418977270210883,-0.036853227937612365,0.5949223311878233,0.16302195170923395,-0.07789329443188264,0.5721745431017516,
0.24595589768335258,-0.13286462302163454,0.6088345636589514,0.3465937153889842,-0.13063226218339885,0.6040563404261559,
0.4207841352050953,-0.0660532141838165,0.5334181568579096,0.36066468461135903,-0.038750860729849124,0.592266217169844,
0.2541118296577233,-0.03890367029122892,0.6013056619605146,-0.23385032428081973,0.5013021801479194,0.628058351492573,
-0.15940397327974548,0.4551857311131614,0.7297754147721504,-0.0638992506637466,0.4372247435288682,0.766522972834131,
0.0007938037125580569,0.467728380654979,0.7641247566883991,0.07869594115150834,0.4301110733749709,0.7611314880050222,
0.16428560867663755,0.45955042496137666,0.7273534976690169,0.22344530828383335,0.5126317352454217,0.6236469862346591,
0.16343924552725905,0.5716510727248528,0.7011055248733994,0.08031385230442917,0.5812402407694596,0.7350439632306426,
-0.007032461635524692,0.5884711439119228,0.7345829053119434,-0.10720339479972739,0.5815531652654057,0.7252627884814442,
-0.1817967611990339,0.5636403180081996,0.686808736738225,-0.18595235548286154,0.5149010055902243,0.6629362613858221,
-0.08419303808608479,0.5092449820323282,0.7186290397650458,0.013149972544273636,0.5002694375899628,0.7468374640141663,
0.10812895757538908,0.48935514533963415,0.7355925176655322,0.19431385690150824,0.4892595191793757,0.6879996210776653,
0.11959213866020346,0.529072827568892,0.7210160974100568,0.01942282690017646,0.5410511565111921,0.7435989303399604,
-0.07946725327191442,0.5315575593662196,0.7364660999694181]
        model_points = numpy.array(raw_value, dtype=numpy.float32)
        model_points = numpy.reshape(model_points, (-1, 3))
        #model_points[:, -1] *= -1
        # model_points = model_points[:,[0,2,1]]
        return model_points
# --------------------------------------------------------------------------------------------------------------------

    def get_pose(self, image,landmarks_2d, landmarks_3d=None):

        fx, fy = float(image.shape[1]), float(image.shape[0])
        self.mat_camera = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])

        dist_coefs = numpy.zeros((4, 1))

        if landmarks_3d is None:
            landmarks_3d = self.model_68_points

        self.r_vec = None
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
