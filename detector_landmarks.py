import numpy
import cv2
import tools_image
import dlib
import tools_draw_numpy
from scipy.spatial import Delaunay
from scipy import ndimage
import math
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
        self.idx_removed_lip_line=numpy.arange(0,60,1).tolist()
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
        landmarks = self.get_landmarks_augm(image)
        if len(landmarks)!=68 or numpy.sum(landmarks)==0:
            return gray

        del_triangles = Delaunay(landmarks).vertices
        for landmark in landmarks:
            cv2.circle(gray,(landmark[0],landmark[1]),2,(0,128,255),-1)
            #gray=tools_draw_numpy.draw_circle(gray, landmark[1],landmark[0], 3, (0,128,255),alpha_transp=0.7)


        for t in del_triangles:
            p0 = (landmarks[t[0],0], landmarks[t[0],1])
            p1 = (landmarks[t[1],0], landmarks[t[1],1])
            p2 = (landmarks[t[2],0], landmarks[t[2],1])
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
    def get_landmarks_augm(self, image):

        '''
        H,W,_ = image.shape

        angles = [5,-5,10,-10]
        res = []
        res.append(self.get_landmarks(image))
        for angle in angles:
            image_augm = ndimage.rotate(image, angle)

            L_augm = self.get_landmarks(image_augm)
            L = L_augm.copy()

            CY, CX,  _ = image.shape
            CYA, CXA, _ = image_augm.shape

            L[:, 0] = CX/2+(L_augm[:, 0]-CXA/2) * math.cos(angle*math.pi/180) - (L_augm[:, 1]-CYA/2) * math.sin(angle*math.pi/180)
            L[:, 1] = CY/2+(L_augm[:, 0]-CXA/2) * math.sin(angle*math.pi/180) + (L_augm[:, 1]-CYA/2) * math.cos(angle*math.pi/180)

            res.append(L)

        res2 = numpy.array(res)
        res2 = numpy.average(res2,axis=0)

        #res_image = self.draw_landmarks_v2(image,res2)
        #cv2.imwrite('./images/output/au.png', res_image)

        #res_image = self.draw_landmarks_v2(image, res[0])
        #cv2.imwrite('./images/output/or.png', res_image)
        '''

        return self.get_landmarks(image)
# ----------------------------------------------------------------------------------------------------------------------