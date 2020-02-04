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
    def __init__(self,filename_config,H=1080,W=1920,mode='dlib'):
        #filename_config = './data/haarcascade_eye.xml'
        #filename_config ='./data/shape_predictor_68_face_landmarks.dat'
        self.W=W
        self.H=H
        self.mode = mode
        self.name = "landmark_detector"
        self.idx_head = numpy.arange(0,27,1).tolist()
        self.idx_nose = numpy.arange(27, 36, 1).tolist()
        self.idx_eyes = numpy.arange(36, 48, 1).tolist()
        self.idx_mouth = numpy.arange(48, 68, 1).tolist()
        self.idx_removed_eyes = numpy.arange(0,68,1).tolist()
        for each in [37,38,40,41,43,44,46,47]:
            self.idx_removed_eyes.remove(each)

        if mode == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(filename_config)
        else:
            self.the_cascade = cv2.CascadeClassifier(filename_config)

# --------------------------------------------------------------------------------------------------------------------
    def detect_face(self,image):
        if self.mode=='opencv':
            return self.__detect_face_opencv(image)
        else:
            return self.__detect_face_dlib(image)

# --------------------------------------------------------------------------------------------------------------------
    def draw_face(self, image):
        if self.mode == 'opencv':
            return self.__draw_face_opencv(image)
        else:
            return self.__draw_face_dlib(image)
# --------------------------------------------------------------------------------------------------------------------
    def __detect_face_dlib(self,image):
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
    def __detect_face_opencv(self, image):
        objects = self.the_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(objects) == 2:
            xmin = min(objects[0][0], objects[1][0])
            xmax = max(objects[0][0] + objects[0][2], objects[1][0] + objects[1][2])

            ymin = min(objects[0][1], objects[1][1])
            ymax = max(objects[0][1] + objects[0][3], objects[1][1] + objects[1][3])

            if xmax > xmin and ymax > ymin:
                return image[ymin:ymax, xmin:xmax, :]

        return None
# ----------------------------------------------------------------------------------------------------------------------
    def __draw_face_opencv(self, image):
        objects = self.the_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(objects) == 2:
            for (x, y, w, h) in objects: cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def __draw_face_dlib(self,image):
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
    def are_frontface_landmarks(self,landmarks):
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
    def get_position_distance_hor(self, image,landmarks):
        if len(landmarks) != 68 or numpy.sum(landmarks) == 0:
            return 1

        swp = [[0,16],[1,15],[2,14],[3,13],[4,12],[5,11],[6,10],[7,9],[8,8],[17,26],[18,25],[19,24],[20,23],[21,22],[31, 35],[32, 34]]
        #swp = [[0,16],[1,15],[2,14],[3,13],[4,12],[5,11],[6,10],[7,9],[8,8]]

        r=[]
        for each in swp:
            x0 = landmarks[each[0], 0]
            y0 = landmarks[each[0], 1]
            x1 = landmarks[each[1], 0]
            y1 = landmarks[each[1], 1]

            r.append(x0-(image.shape[1]-x1))
            r.append(x1-(image.shape[1]-x0))

        r=numpy.array(r)/image.shape[1]
        r = r**2
        q = 100*r.mean()

        return q
# --------------------------------------------------------------------------------------------------------------------
    def get_position_distance_ver(self, image,landmarks):
        landmarks = self.get_landmarks(image)
        if len(landmarks) != 68 or numpy.sum(landmarks) == 0:
            return 1

        mid = landmarks[[31, 32, 33, 34, 35], 1].mean()
        top = landmarks[:,1].min()
        bottom = landmarks[:, 1].max()

        q = (mid - top) - (bottom-mid)

        q = q/image.shape[1]
        q= 100*q**2

        return q
# --------------------------------------------------------------------------------------------------------------------