import numpy
import cv2
# --------------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
from CV import tools_pr_geom
# --------------------------------------------------------------------------------------------------------------------------
class OpticalFlow_LucasKanade():
    def __init__(self, image_start, face_2d=None, folder_out=None):

        self.folder_out = folder_out
        self.feature_params = dict(maxCorners=100,qualityLevel=0.1,minDistance=7,blockSize=7)
        self.lk_params = dict(winSize=(55, 55),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.colors255 = tools_draw_numpy.get_colors(255,colormap='jet',shuffle=True)
        self.mask_ROI = None
        self.cntr = 0

        self.init_start_frame(image_start)
        self.update_ROI(face_2d)
        self.face_2d_prev = face_2d
        return
# --------------------------------------------------------------------------------------------------------------------------
    def init_start_frame(self,image_start):
        if image_start is not None:
            self.gray_prev = cv2.cvtColor(image_start, cv2.COLOR_BGR2GRAY)
            self.mask_ROI = numpy.full((self.gray_prev.shape[0], self.gray_prev.shape[1]), 255, dtype=numpy.uint8)
            self.keypoints_prev = cv2.goodFeaturesToTrack(self.gray_prev, mask=self.mask_ROI, **self.feature_params)
            self.track_id_prev = numpy.arange(self.keypoints_prev.shape[0])
        return
# --------------------------------------------------------------------------------------------------------------------------
    def update_ROI(self,points_2d=None):
        if points_2d is not None:
            self.mask_ROI = numpy.full((self.gray_prev.shape[0], self.gray_prev.shape[1]), 0, dtype=numpy.uint8)
            self.mask_ROI = tools_draw_numpy.draw_convex_hull_cv(self.mask_ROI, numpy.array(points_2d).reshape((-1, 2)), color=255)
            self.mask_ROI = cv2.dilate(self.mask_ROI, numpy.ones((5, 5), numpy.uint8), iterations=10)
        return
# --------------------------------------------------------------------------------------------------------------------------
    def prune_keypoints(self,keypoints,track_ids):

        if keypoints.shape[0]<=2:
            return
        for p in keypoints:
            D = numpy.sum(abs(keypoints.reshape((-1, 2)) - p.reshape((-1, 2))), axis=1)
            if D.shape[0]>1:
                i = numpy.argsort(D)[1]
                if D[i] < 20:
                    keypoints = numpy.delete(keypoints,i,axis=0)
                    track_ids = numpy.delete(track_ids, i)
        return
# --------------------------------------------------------------------------------------------------------------------------
    def add_new_keypoints(self,gray):
        if self.keypoints_cur.shape[0]==0:
            return
        keypoints_new = cv2.goodFeaturesToTrack(gray, mask=self.mask_ROI, **self.feature_params)
        if keypoints_new is not None:
            for p in keypoints_new:
                if self.track_id_cur.shape[0] >= 255:
                    continue
                if p not in self.keypoints_cur:
                    dist = numpy.min(numpy.sum(abs(self.keypoints_cur.reshape((-1, 2)) - p.reshape((-1, 2))), axis=1))
                    if dist < 10:
                        continue
                    self.keypoints_cur = numpy.concatenate([self.keypoints_cur, p.reshape((1, 2))], axis=0)
                    self.track_id_cur = numpy.concatenate([self.track_id_cur, [self.track_id_cur.shape[0]]], axis=0)
        return
# --------------------------------------------------------------------------------------------------------------------------
    def evaluate_flow(self, image):
        self.cntr+=1
        self.gray_cur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_prev, self.gray_cur, self.keypoints_prev, None, **self.lk_params)

        M = None
        self.face_2d_cur = None
        if p1 is not None:
            idx_current_good = (st.flatten() == 1)
            H,W = self.gray_prev.shape[:2]
            idx_inside_ROI = numpy.array([0<=p[0] and p[0]<W and 0<p[1] and p[1]<H and self.mask_ROI[int(p[1]), int(p[0])]>0 for p in p1.reshape((-1, 2))])
            idx_current_good = numpy.logical_and(idx_current_good,idx_inside_ROI)

            self.keypoints_cur = p1[idx_current_good].reshape((-1, 2)).copy()
            self.track_id_cur = self.track_id_prev[idx_current_good]
            # M, _ = tools_pr_geom.fit_affine(self.keypoints_prev[idx_current_good], self.keypoints_cur)
            # if self.face_2d_prev is not None:
            #     self.face_2d_cur = cv2.transform(self.face_2d_prev.reshape((-1, 1, 2)), M).reshape((-1, 2))

            M = tools_pr_geom.fit_homography(self.keypoints_prev[idx_current_good], self.keypoints_cur)[0]
            if self.face_2d_prev is not None:
                self.face_2d_cur = cv2.perspectiveTransform(self.face_2d_prev.reshape((-1, 1, 2)).astype(float), M).reshape((-1, 2))

        if self.cntr%10 ==0:
            self.prune_keypoints(self.keypoints_cur,self.track_id_cur)
            self.add_new_keypoints(self.gray_cur)

        return M
# --------------------------------------------------------------------------------------------------------------------------
    def draw_keypoints(self, image,keypoints,track_ids):
        image = tools_draw_numpy.draw_points_fast(tools_image.desaturate(image), keypoints.reshape((-1,2)), self.colors255[track_ids], w=6)
        return image
# --------------------------------------------------------------------------------------------------------------------------
    def draw_face(self, image, points):
        image = tools_draw_numpy.draw_contours(image, numpy.array(points).reshape((-1, 2)),color=(0,0,200),w=5,transperency=0.9)
        lines = numpy.array(points).reshape((-1, 2))[[0,2,1,3]].reshape((-1,4))

        image = tools_draw_numpy.draw_lines(image,lines, color=(0, 0, 200),w=1,antialiasing=True)
        image = tools_draw_numpy.draw_points(image, numpy.array(points).reshape((-1, 2)),color=(0,0,200),w=24)
        return image
# --------------------------------------------------------------------------------------------------------------------------
    def draw_current_frame(self):
        im_result = self.draw_keypoints(tools_image.saturate(self.gray_cur),self.keypoints_cur,self.track_id_cur)
        #im_result = cv2.addWeighted(im_result, 0.6, tools_image.saturate(self.mask_ROI), 0.4, 0)
        if self.face_2d_cur is not None:
            im_result = self.draw_face(im_result, self.face_2d_cur)
        return im_result
# --------------------------------------------------------------------------------------------------------------------------
    def draw_prev_frame(self):
        im_result = self.draw_keypoints(tools_image.saturate(self.gray_prev), self.keypoints_prev, self.track_id_prev)
        #im_result = cv2.addWeighted(im_result, 0.6, tools_image.saturate(self.mask_ROI), 0.4, 0)
        if self.face_2d_prev is not None:
            im_result = self.draw_face(im_result, self.face_2d_prev)
        return im_result
# --------------------------------------------------------------------------------------------------------------------------
    def next_step(self):
        self.keypoints_prev = self.keypoints_cur.copy()
        self.gray_prev = self.gray_cur.copy()
        self.track_id_prev = self.track_id_cur.copy()
        self.face_2d_prev = self.face_2d_cur
        self.update_ROI(points_2d=self.face_2d_cur)

        if self.keypoints_prev.shape[0]==0:
            self.init_start_frame(tools_image.saturate(self.gray_cur))

        return
# ---------------------------------------------------------------------------------------------------------------------