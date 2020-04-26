import os
import math
import cv2
import numpy
import time
# ---------------------------------------------------------------------------------------------------------------------
from skimage.transform import hough_line,probabilistic_hough_line
import tools_draw_numpy
# ---------------------------------------------------------------------------------------------------------------------
from numba.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_pr_geom
import tools_render_CV
import tools_Skeletone
import tools_Hough
import tools_wavefront
import tools_filter
# ---------------------------------------------------------------------------------------------------------------------
#0,1,2,3 - outer bounds
#4,5,6 - left goal area
#7,8,9 - left penalty area
#10,11,12 - right goal area
#13,14,15 - right pen area
# ---------------------------------------------------------------------------------------------------------------------
class Soccer_Field_Processor(object):
    def __init__(self):
        self.folder_out = './images/output/'
        if not os.path.exists(self.folder_out + 'cache'):os.mkdir(self.folder_out + 'cache')

        self.do_trimming= True
        self.method_ske = 'ski'
        self.GT_Z = 0

        self.w_fild = 100.0
        self.h_fild = 75.0
        self.w_penl = 18.0
        self.h_penl = 44.0
        self.w_goal = 6.0
        self.h_goal = 20.0
        self.landmarks_GT, self.lines_GT = self.get_GT()

        self.projection_scale_factor_xy = (self.w_goal/self.h_goal)/(0.39)



        self.color_amber      = (0  , 124, 207)
        self.color_white      = (255, 255, 255)
        self.color_red        = (17 ,   6, 124)
        self.color_bright_red = (0  ,  32, 255)
        self.color_green      = (137, 233,  164)
        self.color_black      = (0  ,  0,    0)
        self.color_gray       = (20 , 20,   20)
        self.color_grass      = (31, 124, 104)

        self.Skelenonizer = tools_Skeletone.Skelenonizer()
        self.Hough = tools_Hough.Hough()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_grass_mask(self,image):

        hue = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)[:,:,0]
        mask = 255*(hue>=30) * (hue<=75)
        return mask
# ----------------------------------------------------------------------------------------------------------------------
    def get_angle(self,line):
        x1, y1, x2, y2 = line
        return 90 + math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
# ----------------------------------------------------------------------------------------------------------------------
    def get_colors(self,N,alpha_blend=None):
        colors = []
        for i in range(0, N):
            hue = int(255 * i / (N-1))
            color = cv2.cvtColor(numpy.array([hue, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)[0][0]
            if alpha_blend is not None:color = ((alpha_blend) * numpy.array((255, 255, 255)) + (1-alpha_blend) * numpy.array(color))
            colors.append((int(color[0]), int(color[1]), int(color[2])))

        return colors
# ----------------------------------------------------------------------------------------------------------------------
    def draw_lines(self,image, lines,color=(255,255,255),w=4,put_text=False):

        result = image.copy()
        H, W = image.shape[:2]
        for id,(x1,y1,x2,y2) in enumerate(lines):
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, w)
            if put_text:
                x,y = int(x1+x2)//2, int(y1+y2)//2
                cv2.putText(result, '{0}'.format(id),(min(W - 10, max(10, x)), min(H - 5, max(10, y))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


        return result
# ----------------------------------------------------------------------------------------------------------------------
    def draw_GT_lines(self, image, lines, ids, w=4,put_text=False):

        colors = self.get_colors(16,alpha_blend=0.5)
        H, W = image.shape[:2]
        result = image.copy()
        for line, id in zip(lines, ids):
            if line[0] is None or line[1] is None or line[2] is None or line[3] is None:continue
            cv2.line(result, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), colors[id], w)
            if put_text:
                x,y = int(line[0]+line[2])//2, int(line[1]+line[3])//2
                cv2.putText(result, '{0}'.format(id),(min(W - 10, max(10, x)), min(H - 5, max(10, y))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def draw_GT_lmrks(self, image, landmarks, ids, R=8, fill=-1, put_text=False):
        result = image.copy()
        if landmarks is None:return result

        H,W = image.shape[:2]
        colors = self.get_colors(36)

        for lm,id in zip(landmarks,ids):
            if lm[0] is None or lm[1] is None: continue
            cv2.circle(result, (int(lm[0]), int(lm[1])), R, colors[id], fill)
            if put_text:
                cv2.putText(result, '{0}'.format(id), (min(W-10,max(10,int(lm[0]))), min(H-5,max(10,int(lm[1])))), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(128, 128, 128), 1, cv2.LINE_AA)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def draw_playground_GT(self, w=8, R=6,scale = 1.0,put_text=True):

        padding = 10

        L_GT, lines_GT = self.get_GT()
        W = L_GT[:,0].max() - L_GT[:,0].min() + padding
        H = L_GT[:,1].max() - L_GT[:,1].min() + padding

        lines_GT[:,0]-=L_GT[:,0].min()-padding//2
        lines_GT[:,2]-=L_GT[:,0].min()-padding//2
        lines_GT[:,1]-=L_GT[:,1].min()-padding//2
        lines_GT[:,3]-=L_GT[:,1].min()-padding//2

        L_GT[:,0]-=L_GT[:,0].min()-padding//2
        L_GT[:,1]-=L_GT[:,1].min()-padding//2

        image = numpy.full((int(H*scale), int(W*scale), 3), 64, numpy.uint8)

        image = self.draw_GT_lines(image, scale*lines_GT, ids=numpy.arange(0, len(lines_GT), 1), w=w, put_text=put_text)
        image = self.draw_GT_lmrks(image, scale * L_GT  , ids=numpy.arange(0, len(L_GT)    , 1), R=R, put_text=put_text)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_playground_homography(self, image, homography, w=1, R=4):

        if homography is None: return image
        lmrks_GT, lines_GT = self.get_GT()

        lines_GT_trans = cv2.perspectiveTransform(lines_GT.reshape((-1, 1, 2)), homography).reshape((-1, 4))
        lmrks_GT_trans = cv2.perspectiveTransform(lmrks_GT.reshape((-1, 1, 2)), homography).reshape((-1, 2))


        result = self.draw_GT_lines(image, lines_GT_trans, ids=numpy.arange(0, len(lines_GT), 1), w=w)
        result = self.draw_GT_lmrks(result, lmrks_GT_trans, ids=numpy.arange(0, len(lmrks_GT), 1), R=R)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def refine_homography(self,homography,lines):

        lmrks_GT, lines_GT = self.landmarks_GT.copy(), self.lines_GT.copy()

        lines_GT_trans = cv2.perspectiveTransform(lines_GT.reshape((-1, 1, 2)), homography).reshape((-1, 4))
        is_line_visible = numpy.zeros(len(lines_GT))
        lines_GT_trans_refined = lines_GT_trans.copy()

        tol= 20
        box = (0,0,self.W,self.H)

        for i, line_gt in enumerate(lines_GT_trans):
            line_gt = tools_render_CV.trim_line_by_box(line_gt, box)
            if numpy.any(numpy.isnan(line_gt)):continue
            if numpy.linalg.norm(line_gt[2:]-line_gt[:2])<tol:continue
            is_line_visible[i]=1

            for line in lines.astype(int):
                d = tools_render_CV.distance_segment_to_line(line_gt, line)
                if d<tol:
                    p1,p2,da = tools_render_CV.distance_between_lines((line_gt[0],line_gt[1],line_gt[0]+1,line_gt[1]+1), line,clampAll=False)
                    p3,p4,db = tools_render_CV.distance_between_lines((line_gt[2],line_gt[3],line_gt[2]+1,line_gt[3]+1), line,clampAll=False)
                    if abs(self.get_angle((p2[0],p2[1],p4[0],p4[1]))-self.get_angle(lines_GT_trans_refined[i]))<10:
                        lines_GT_trans_refined[i] = [p2[0],p2[1],p4[0],p4[1]]

        landmarks = []
        list_of_lines = self.get_list_of_lines()
        for i1, i2 in list_of_lines:
            landmarks.append(tools_render_CV.line_intersection(lines_GT_trans_refined[i1], lines_GT_trans_refined[i2]))

        tol = 150
        for c in self.get_list_of_X_crosses()+self.get_list_of_T_crosses():
            if numpy.any(numpy.isnan(landmarks[c])):
                continue
            i1,i2 = list_of_lines[c]

            if (not is_line_visible[i1]) or (not is_line_visible[i2]):continue

            d1 =  numpy.linalg.norm(landmarks[c] - lines_GT_trans_refined[i1][:2])
            d2 =  numpy.linalg.norm(landmarks[c] - lines_GT_trans_refined[i1][2:])
            if min(d1,d2)<tol:
                if d1<d2:lines_GT_trans_refined[i1][:2] = landmarks[c]
                else:lines_GT_trans_refined[i1][2:] = landmarks[c]

            d1 = numpy.linalg.norm(landmarks[c] - lines_GT_trans_refined[i2][:2])
            d2 = numpy.linalg.norm(landmarks[c] - lines_GT_trans_refined[i2][2:])
            if min(d1, d2) < tol:
                if d1 < d2:lines_GT_trans_refined[i2][:2] = landmarks[c]
                else:lines_GT_trans_refined[i2][2:] = landmarks[c]

        idx = numpy.where(is_line_visible)[0]
        result = numpy.array([tools_render_CV.trim_line_by_box(lines_GT_trans_refined[i], box) for i in idx])

        return  result,idx
# ----------------------------------------------------------------------------------------------------------------------
    # lines
    # 0,1,2,3  - outer bounds
    # 4,5,6    - left  penalty area
    # 7,8,9    - left  goal area
    # 10,11,12 - right penalty area
    # 13,14,15 - right goal area
# ----------------------------------------------------------------------------------------------------------------------
    def get_list_of_lines(self):
        list3 = [[3, 0], [3, 2], [3, 4], [3, 6], [3, 7], [3, 9]]
        list5 = [[5, 0], [5, 2], [5, 4], [5, 6], [5, 7], [5, 9]]
        list8 = [[8, 0], [8, 2], [8, 4], [8, 6], [8, 7], [8, 9]]

        list1  = [[1 , 0], [ 1, 2], [ 1, 10], [ 1, 12], [ 1, 13], [ 1, 15]]
        list11 = [[11, 0], [11, 2], [11, 10], [11, 12], [11, 13], [11, 15]]
        list14 = [[14, 0], [14, 2], [14, 10], [14, 12], [14, 13], [14, 15]]
        list_of_lines = list3 + list5 + list8 + list1 + list11 + list14
        return list_of_lines
# ----------------------------------------------------------------------------------------------------------------------
    def get_list_of_X_crosses(self):
        res = [0,1,8,9,16,17,18,19,26,27,34,35]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def get_list_of_T_crosses(self):
        res = [2,3,4,5,20,21,22,23]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def get_GT(self):
        scale_factor = 20
        w_fild = self.w_fild
        h_fild = self.h_fild
        w_penl = self.w_penl
        h_penl = self.h_penl
        w_goal = self.w_goal
        h_goal = self.h_goal

        lines = [[-w_fild/2, -h_fild/2, +w_fild/2,          -h_fild/2], [+w_fild/2,          -h_fild/2, +w_fild/2         , +h_fild/2],[+w_fild/2         , +h_fild/2, -w_fild/2, +h_fild/2],[-w_fild/2,+h_fild/2,-w_fild/2,-h_fild/2],
                 [-w_fild/2, -h_penl/2, -w_fild/2 + w_penl, -h_penl/2], [-w_fild/2 + w_penl, -h_penl/2, -w_fild/2 + w_penl, +h_penl/2],[-w_fild/2 + w_penl, +h_penl/2, -w_fild/2, +h_penl/2],
                 [-w_fild/2, -h_goal/2, -w_fild/2 + w_goal, -h_goal/2], [-w_fild/2 + w_goal, -h_goal/2, -w_fild/2 + w_goal, +h_goal/2],[-w_fild/2 + w_goal, +h_goal/2, -w_fild/2, +h_goal/2],
                 [+w_fild/2, -h_penl/2, +w_fild/2 - w_penl, -h_penl/2], [+w_fild/2 - w_penl, -h_penl/2, +w_fild/2 - w_penl, +h_penl/2],[+w_fild/2 - w_penl, +h_penl/2, +w_fild/2, +h_penl/2],
                 [+w_fild/2, -h_goal/2, +w_fild/2 - w_goal, -h_goal/2], [+w_fild/2 - w_goal, -h_goal/2, +w_fild/2 - w_goal, +h_goal/2],[+w_fild/2 - w_goal, +h_goal/2, +w_fild/2, +h_goal/2]]

        lines = numpy.array(lines)

        lines[:,0]+=w_fild/2
        lines[:,1]+=h_fild/2
        lines[:,2]+=w_fild/2
        lines[:,3]+=h_fild/2

        lines = scale_factor*numpy.array(lines)

        landmarks = []
        for l1, l2 in self.get_list_of_lines():
            landmarks.append(tools_render_CV.line_intersection(lines[l1],lines[l2]))

        landmarks = numpy.array(landmarks)
        return landmarks, lines
# ----------------------------------------------------------------------------------------------------------------------
    def is_good_ratio(self,ratio_xy_cand,idx_v1,idx_v2,idx_h1,idx_h2):

        line_v1 = self.lines_GT[idx_v1]
        line_v2 = self.lines_GT[idx_v2]
        line_h1 = self.lines_GT[idx_h1]
        line_h2 = self.lines_GT[idx_h2]

        dx = (line_v1[0] - line_v2[0])
        dy = (line_h1[1] - line_h2[1])
        ratio_xy_gt = numpy.abs(float(dx/dy))

        actual_factor = ratio_xy_gt / numpy.abs(ratio_xy_cand)

        tol = 0.2
        result1 = numpy.abs(actual_factor/self.projection_scale_factor_xy) >= 1-tol
        result2 = numpy.abs(actual_factor/self.projection_scale_factor_xy) <= 1+tol

        return result1 and result2
# ----------------------------------------------------------------------------------------------------------------------
    def create_combinations(self, lines_vert, lines_horz):

        target_v  = [[3, 8],[3, 5], [8, 5]]

        target_h  = [[0,4],[0,7],[0,9],[0,6],[0,2],
                           [4,7],[4,9],[4,6],[4,2],
                                 [7,9],[7,6],[7,2],
                                       [9,6],[9,2],
                                             [6,2]]
        i=0
        idxs,combs = [],[]
        for h1 in range(len(lines_horz)-1):
            for h2 in range(h1,len(lines_horz)):
                if h1 != h2:
                    for v1 in range(len(lines_vert)-1):
                        for v2 in range(v1,len(lines_vert)):
                            if v1!=v2:
                                ratio_cand = tools_render_CV.get_ratio_4_lines(lines_vert[v1],lines_horz[h1],lines_vert[v2],lines_horz[h2],do_debug=False)

                                for (i1,i2) in target_v:
                                    for (i3,i4) in target_h:
                                        is_good_ratio = self.is_good_ratio(1/ratio_cand,i1,i2,i3,i4)
                                        if is_good_ratio:
                                            idxs.append([i1,i3,i2,i4])
                                            combs.append([lines_vert[v1], lines_horz[h1], lines_vert[v2], lines_horz[h2]])
                                        if (i == 101):
                                            i=i
                                        i+=1


        return numpy.array(combs), numpy.array(idxs)
# ----------------------------------------------------------------------------------------------------------------------
    def get_accuracy_lines(self, image, lines_GT_trans, lines_cand, weight_cand, tol=15, do_debug = False):

        pad = 50
        box = numpy.array((0+pad,0+pad,self.W-pad,self.H-pad))
        lines_GT_trimmed,weight_GT_trimmed = [],[]
        for line in lines_GT_trans:
            trimmed = tools_render_CV.trim_line_by_box(line, box)
            if not numpy.any(numpy.isnan(trimmed)):
                lines_GT_trimmed.append(trimmed)
                weight_GT_trimmed.append(numpy.linalg.norm(trimmed[:2] - trimmed[2:]))

        weight_GT_trimmed = numpy.array(weight_GT_trimmed)/numpy.sum(weight_GT_trimmed)
        weight_cand = weight_cand/numpy.sum(weight_cand)

        if len(lines_GT_trimmed)==0:
            return 0, 0, image

        lines_GT_trimmed = numpy.array(lines_GT_trimmed)

        dist = numpy.zeros((len(lines_GT_trimmed), len(lines_cand)))

        for s,segment in enumerate(lines_GT_trimmed):
                for l,line in enumerate(lines_cand):
                    dist[s,l] = tools_render_CV.distance_segment_to_line(segment, line,do_debug=False)

        hit_gt = 1*(numpy.min(dist,axis=1)<=tol)
        if numpy.sum(weight_GT_trimmed)>0:
            recall = numpy.dot(hit_gt,weight_GT_trimmed)
        else:
            recall = 0

        hit_cand = 1 * (numpy.min(dist, axis=0) <= tol)
        if numpy.sum(weight_cand) > 0:
            precision =  numpy.dot(hit_cand,weight_cand)
        else:
            precision = 0

        #if do_debug:
        #    image = self.draw_lines(image, lines_GT_trimmed[hit_gt>0],  color=(128,255,0),w=4)
        #    image = self.draw_lines(image, lines_GT_trimmed[hit_gt==0], color=(255,255,255),w=4)
        #    image = self.draw_lines(image, lines_cand, color=(0, 0, 180), w=1)

        return precision, recall, image
# ----------------------------------------------------------------------------------------------------------------------
    def get_pose_homography(self, image, landmarks, ids, lines_all, weight_all, base_name='', debug_index=None):

        homography, result = tools_pr_geom.fit_homography(self.landmarks_GT[ids], numpy.array(landmarks,dtype=float),numpy.array(self.get_colors(36))[ids])
        lines_GT_trans = cv2.perspectiveTransform(self.lines_GT.reshape((-1, 1, 2)), homography).reshape((-1, 4))

        color, transparency = (32, 90, 0), 0.6
        if debug_index is not None:
            image = tools_draw_numpy.draw_convex_hull(image, landmarks, color=color, transperency=transparency)
            image = self.draw_playground_homography(image, homography,w=4)
            image = self.draw_lines(image, lines_all, color=self.color_red, w=1)

        precision, recall = 0,0
        precision,recall, image  = self.get_accuracy_lines(image, lines_GT_trans, lines_all, weight_all, tol=15, do_debug=debug_index)

        if debug_index is not None:
            scale = 0.15
            L_GT, lines_GT = self.get_GT()
            image_pg = self.draw_playground_GT(w=1, R=2,scale=scale,put_text=False)
            image_pg = tools_draw_numpy.draw_convex_hull(image_pg, scale*L_GT[ids],color=color, transperency=transparency)

            image = tools_image.put_image(image, image_pg, 0, 0)
            cv2.imwrite(self.folder_out + base_name +'_hmgr_%03d_%03d_%03d.png' % (100 * precision, 100 * recall, debug_index), image)

        return homography, precision,recall
# ----------------------------------------------------------------------------------------------------------------------
    def get_pose_homography_inverce(self, image, landmarks, ids, lines_all, weight_all, base_name='', do_debug=False):

        landmarks_GT, lines_GT = self.get_GT()
        homography_inv, result = tools_pr_geom.fit_homography(numpy.array(landmarks, dtype=float),landmarks_GT[ids],colors=numpy.array(self.get_colors(36))[ids])
        lines_all_trans = cv2.perspectiveTransform(lines_all.reshape((-1, 1, 2)), homography_inv).reshape((-1, 4))

        color, transparency = (32, 0, 255), 0.6
        if do_debug:
            scale = 0.25
            image = tools_draw_numpy.draw_convex_hull(image, landmarks, color=color, transperency=transparency)
            image = self.draw_lines(image,lines_all,self.color_amber,w=4)

            image_debug = self.draw_playground_GT()
            image_debug = tools_draw_numpy.draw_convex_hull(image_debug, landmarks_GT[ids], color=color,transperency=0.95)
            image_debug = self.draw_lines(image_debug, lines_all_trans, color=self.color_amber, w=2)
            image_debug = tools_image.put_image(image_debug, cv2.resize(image,(int(scale*self.W),int(scale*self.H))), 0,-int(scale*self.W)-1)
            cv2.imwrite(self.folder_out + base_name + '_inv_%03d.png'%do_debug,image_debug)

        return homography_inv, 0, 0

# ----------------------------------------------------------------------------------------------------------------------
    def get_angle_range(self,angle_base,angle_delta,steps):
        if angle_base is None:
            angle_base = 90

        start = (angle_base - angle_delta) * numpy.pi / 180
        stop  = (angle_base + angle_delta) * numpy.pi / 180
        step  = steps*numpy.pi / 180
        res = numpy.arange(start, stop, step)
        return  res
# ----------------------------------------------------------------------------------------------------------------------
    def get_angle_bound2(self,lines_vert,weights_vert):

        angles = []
        for line in lines_vert:
            angles.append(self.get_angle(line))

        #[0,90]
        #[80]

        x = numpy.dot(numpy.array(angles),numpy.array(weights_vert))/weights_vert.sum()

        return 110
# ----------------------------------------------------------------------------------------------------------------------
    def trim_ROI_by_vert_lines(self, image, line_up, line_down):

        line_trimmed = tools_render_CV.line_box_intersection(line_up, (0,0,self.W,self.H))
        points = [(0, 0), (line_trimmed[0], line_trimmed[1]), (line_trimmed[2], line_trimmed[3])]
        result = tools_draw_numpy.draw_convex_hull(image, points, color=(0, 0, 0))

        line_trimmed = tools_render_CV.line_box_intersection(line_down, (0, 0, self.W, self.H))
        points = [(self.W, self.H), (line_trimmed[0], line_trimmed[1]), (line_trimmed[2], line_trimmed[3])]
        result = tools_draw_numpy.draw_convex_hull(result,points,color=(0,0,0))

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def extract_lines(self,image,base_name='',do_debug=False):

        X,success = tools_IO.load_if_exists(self.folder_out+'cache/' + base_name+'_lines.dat')
        if success:
            (lines_vert, weights_vert, lines_horz, weights_horz) = X
            skeleton = image.copy()
        else:

            filtered = image.copy()
            angle_bound1 = 70
            angle_bound2 = 110

            if self.method_ske == 'canny':
                skeleton = cv2.Canny(filtered, 5, 150, apertureSize=3)
            else:
                skeleton = self.Skelenonizer.binarized_to_skeleton2(self.Skelenonizer.binarize(filtered))

            grass_mask = self.get_grass_mask(image)
            skeleton[grass_mask == 0] = 0

            lines_vert, weights_vert = self.Hough.get_lines_ski(skeleton, max_count=4, min_weight=100,the_range=self.get_angle_range(angle_bound1, 10, 0.1))
            if self.do_trimming:
                skeleton = self.trim_ROI_by_vert_lines(skeleton, lines_vert[0], lines_vert[-1])

            #angle_bound2 = self.get_angle_bound2(lines_vert,weights_vert)
            lines_horz, weights_horz = self.Hough.get_lines_ski(skeleton, max_count=4, min_weight=0,the_range=self.get_angle_range(angle_bound2, 20, 0.1))
            tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_lines.dat',(lines_vert, weights_vert, lines_horz, weights_horz))

        if not success and do_debug:
            image_lines = 0*image
            image_lines = self.draw_lines(image_lines, lines_horz, color=(0, 128, 255), w=4)
            image_lines = self.draw_lines(image_lines, lines_vert, color=(0, 0, 255), w=4)
            image_lines[skeleton>0]=255
            cv2.imwrite(self.folder_out + base_name + '_lines_v.png', image_lines)

        return lines_vert,weights_vert, lines_horz,weights_horz
# ----------------------------------------------------------------------------------------------------------------------
    def append_annotation(self,filename,imagename,lines,IDs):

        if not os.path.isfile(filename):
            f = open(filename, 'a')
            f.write('filename x1 y1 x2 y2 class_ID\n')
            f.close()

        for line,ID in zip(lines.astype(int),IDs):
            vec = [imagename,line[0],line[1],line[2],line[3],ID]
            tools_IO.save_raw_vec(vec, filename, mode=(os.O_RDWR | os.O_APPEND), fmt='%s', delim=' ')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def estimate_homography(self,image, lines_vert, weights_vert, lines_horz, weights_horz,base_name=''):

        H, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_homography.dat')
        if success:
            return H

        combs, idxs = self.create_combinations(lines_vert, lines_horz)
        precisions, homographies = [], []
        for i, (comb, idx) in enumerate(zip(combs, idxs)):
            lines = numpy.zeros((16, 4))
            lines[idx] = comb

            lm, idx_lm = [], []
            for cnt, (l1, l2) in enumerate(self.get_list_of_lines()):
                if numpy.count_nonzero(lines[l1]) > 0 and numpy.count_nonzero(lines[l2]) > 0:
                    idx_lm.append(cnt)
                    lm.append(tools_render_CV.line_intersection(lines[l1], lines[l2]))

            homography, precision, recall = self.get_pose_homography(image, lm, idx_lm, numpy.vstack((lines_vert, lines_horz)),
                                                                     numpy.hstack((weights_vert, weights_horz)),
                                                                     base_name=base_name, debug_index=None)

            precisions.append(precision)
            homographies.append(homography)

        H = homographies[numpy.argsort(-numpy.array(precisions))[0]]
        tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_homography.dat',H)
        return H
# ----------------------------------------------------------------------------------------------------------------------
    def process_view(self, filename_in, do_debug=None):

        time_start = time.time()
        image = cv2.imread(filename_in)
        self.H, self.W = image.shape[:2]

        base_name = filename_in.split('/')[-1].split('.')[0]
        ext = filename_in.split('/')[-1].split('.')[1]

        lines_vert, weights_vert, lines_horz, weights_horz = self.extract_lines(image,base_name,do_debug=True)

        do_homography = True
        if do_homography:
            homography = self.estimate_homography(tools_image.desaturate(image),lines_vert, weights_vert, lines_horz, weights_horz,base_name=base_name)

            #result = self.draw_playground_homography(tools_image.desaturate(image),homography,w=4)

            lines_GT_trans_refined, IDs = self.refine_homography(homography, numpy.vstack((lines_vert, lines_horz)))
            result = self.draw_lines(tools_image.desaturate(image), lines_GT_trans_refined, color=self.color_green, w=4)
            self.append_annotation(self.folder_out+'annotation.txt', base_name+'.'+ext, lines_GT_trans_refined, IDs)

            result = self.draw_lines(result,lines_vert,color=self.color_red,w=1)
            result = self.draw_lines(result,lines_horz, color=self.color_amber, w=1)
            cv2.imwrite(self.folder_out + base_name + '.png', result)


        print('%s - %1.2f sec' % (base_name, (time.time() - time_start)))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def train_on_annotation(self,filename_in):

        self.H, self.W = 1080,1920

        all_filenames,all_lines,all_IDs = [],[],[]
        for each in tools_IO.load_mat(filename_in,delim=' ')[1:]:
            all_filenames.append(each[0])
            all_lines.append([int(each[1]),int(each[2]),int(each[3]),int(each[4])])
            all_IDs.append(int(each[5]))

        all_filenames = numpy.array(all_filenames)
        all_lines = numpy.array(all_lines)
        all_IDs = numpy.array(all_IDs)

        all_landmarks, all_idx_lm = [], []
        for name in numpy.unique(all_filenames):
            lines = numpy.zeros((16, 4))
            idx = numpy.where(all_filenames==name)
            lines[all_IDs[idx]] = all_lines[idx]

            for cnt, (l1, l2) in enumerate(self.get_list_of_lines()):
                if numpy.count_nonzero(lines[l1]) > 0 and numpy.count_nonzero(lines[l2]) > 0:
                    all_idx_lm.append(cnt)
                    all_landmarks.append(tools_render_CV.line_intersection(lines[l1], lines[l2]))

        all_landmarks = numpy.array(all_landmarks)
        colors = self.get_colors(36)

        image_all = numpy.zeros((255, 255, 3), dtype=numpy.uint8)
        for lmid in numpy.unique(all_idx_lm):
            image_stats = numpy.zeros((255, 255, 3), dtype=numpy.uint8)
            landmarks = all_landmarks[numpy.where(all_idx_lm == lmid)]
            landmarks_filtered = tools_filter.do_filter_kalman_2D(landmarks)
            landmarks_filtered[:,0]/=self.W/255
            landmarks_filtered[:,1]/=self.H/255

            for i in range(len(landmarks_filtered)-1):
                p1 = (int(landmarks_filtered[i  ,0]), int(landmarks_filtered[i  ,1]))
                p2 = (int(landmarks_filtered[i+1,0]), int(landmarks_filtered[i+1,1]))
                cv2.line(image_stats,p1,p2,color=colors[lmid],thickness=15)

            image_all = tools_image.put_layer_on_image(image_all,image_stats)
            cv2.imwrite(self.folder_out+'trj_LM_%02d.png'%lmid,image_stats)

        cv2.imwrite(self.folder_out + 'trj_all.png', image_all)
        return
# ----------------------------------------------------------------------------------------------------------------------