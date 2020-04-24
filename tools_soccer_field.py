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
        self.blur_kernel = 2
        self.GT_Z = 0

        self.w_fild = 100.0
        self.h_fild = 70.0
        self.w_penl = 18.0
        self.h_penl = 44.0
        self.w_goal = 6.0
        self.h_goal = 20.0
        self.landmarks_GT, self.lines_GT = self.get_GT()

        self.color_amber      = (0  , 124, 207)
        self.color_white      = (255, 255, 255)
        self.color_red        = (17 ,   6, 124)
        self.color_bright_red = (0  ,  32, 255)
        self.color_green      = (117, 134, 129)
        self.color_black      = (0  ,  0,    0)
        self.color_gray       = (20 , 20,   20)
        self.color_grass      = (31, 124, 104)

        self.Skelenonizer = tools_Skeletone.Skelenonizer()
        self.Hough = tools_Hough.Hough()

        self.scale=0.5      #trick for p3l

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_grass_mask(self,image):

        hue = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)[:,:,0]
        mask = 255*(hue>=30) * (hue<=75)
        return mask
# ----------------------------------------------------------------------------------------------------------------------
    def get_angle(self,x1, y1, x2, y2):
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
    def draw_playground_homography(self, image, homography, check_reflect=False, w=1, R=4):

        if homography is None: return image
        lmrks_GT, lines_GT = self.get_GT()

        lines_GT_trans = cv2.perspectiveTransform(lines_GT.reshape((-1, 1, 2)), homography).reshape((-1, 4))
        lmrks_GT_trans = cv2.perspectiveTransform(lmrks_GT.reshape((-1, 1, 2)), homography).reshape((-1, 2))

        if check_reflect:
            lines_GT_trans_corrected, lmrks_GT_trans_corrected = [], []
            baseline = lines_GT_trans[3]
            for i in range(len(lines_GT_trans)):
                x1,y1 = lines_GT_trans[i][0:2]
                if tools_render_CV.is_point_above_line((x1,y1),baseline,tol=-10):
                    x1,y1 = None,None
                x2,y2 = lines_GT_trans[i][2:]
                if tools_render_CV.is_point_above_line((x2,y2), baseline,tol=-10):
                    x2,y2 = None, None
                lines_GT_trans_corrected.append([x1,y1,x2,y2])
            for i in range(len(lmrks_GT_trans)):
                x,y = lmrks_GT_trans[i]
                if tools_render_CV.is_point_above_line((x,y), baseline,tol=-10):
                    x,y = None,None
                lmrks_GT_trans_corrected.append([x,y])

            lines_GT_trans = numpy.array(lines_GT_trans_corrected)
            lmrks_GT_trans = numpy.array(lmrks_GT_trans_corrected)

        result = self.draw_GT_lines(image, lines_GT_trans, ids=numpy.arange(0, len(lines_GT), 1), w=w)
        result = self.draw_GT_lmrks(result, lmrks_GT_trans, ids=numpy.arange(0, len(lmrks_GT), 1), R=R)

        return result
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
    def create_combinations(self, lines_vert, lines_horz):

        dist_GT_v,dist_GT_h = {},{}
        target_v  = [[3, 8],[3, 5], [8, 5]]
        target_h  = [[4, 6],[7, 9], [4, 9],[7,6]]

        for (v1, v2) in target_v:
            dist_GT_v[tuple((v1, v2))] = tools_render_CV.distance_between_lines(self.lines_GT[v1], self.lines_GT[v2],clampAll=True)[2]

        for (h1, h2) in target_h:
            dist_GT_h[tuple((h1, h2))] = tools_render_CV.distance_between_lines(self.lines_GT[h1], self.lines_GT[h2],clampAll=True)[2]


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

                                        ratio_GT = dist_GT_v[(i1,i2)]/dist_GT_h[(i3,i4)]

                                        is_good = 1.4 * 0.75 <= (ratio_cand / ratio_GT) and (ratio_cand / ratio_GT) <= 1.4 * 1.25
                                        if is_good:
                                            idxs.append([i1,i3,i2,i4])
                                            combs.append([lines_vert[v1], lines_horz[h1], lines_vert[v2], lines_horz[h2]])
                                        if (i == 101):
                                            i=i
                                        i+=1


        return numpy.array(combs), numpy.array(idxs)
# ----------------------------------------------------------------------------------------------------------------------
    def get_recall_lines(self, image, lines_GT, lines_cand,weight_cand,tol=15,do_debug = False):

        pad = 50
        box = numpy.array((0+pad,0+pad,self.W-pad,self.H-pad))
        lines_GT_trimmed,L = [],[]
        for line in lines_GT:
            trimmed = tools_render_CV.trim_line_by_box(line, box)
            if not numpy.any(numpy.isnan(trimmed)):
                lines_GT_trimmed.append(trimmed)
                L.append(numpy.linalg.norm(trimmed[:2] - trimmed[2:]))

        if len(lines_GT_trimmed)==0:
            return 0, 0, image

        lines_GT_trimmed = numpy.array(lines_GT_trimmed)

        dist = numpy.zeros((len(lines_GT_trimmed), len(lines_cand)))

        for s,segment in enumerate(lines_GT_trimmed):
                for l,line in enumerate(lines_cand):
                    dist[s,l] = tools_render_CV.distance_segment_to_line(segment, line,do_debug=False)

        hit_gt = 1*(numpy.min(dist,axis=1)<=tol)
        if numpy.sum(L)>0:
            recall = numpy.dot(hit_gt,L)/numpy.sum(L)
        else:
            recall = 0

        hit_cand = 1 * (numpy.min(dist, axis=0) <= tol)
        if numpy.sum(weight_cand) > 0:
            precision =  numpy.dot(hit_cand,weight_cand)/numpy.sum(weight_cand)
        else:
            precision = 0

        if do_debug:
            image = self.draw_lines(image, lines_GT_trimmed[hit_gt>0],  color=(128,255,0),w=4)
            image = self.draw_lines(image, lines_GT_trimmed[hit_gt==0], color=(255,255,255),w=4)
            image = self.draw_lines(image, lines_cand, color=(0, 0, 180), w=1)

        return recall, precision, image
# ----------------------------------------------------------------------------------------------------------------------
    def get_pose_homography(self, image, landmarks, ids, lines_all,weight_all, do_debug=None):

        landmarks_GT, lines_GT = self.get_GT()
        homography, result = tools_pr_geom.fit_homography(landmarks_GT[ids], numpy.array(landmarks,dtype=float),numpy.array(self.get_colors(36))[ids])
        lines_GT_trans = cv2.perspectiveTransform(self.lines_GT.reshape((-1, 1, 2)), homography).reshape((-1, 4))

        color, transparency = (32, 0, 255), 0.6
        if do_debug:

            image = tools_draw_numpy.draw_convex_hull(image, landmarks, color=color, transperency=transparency)
            image = self.draw_playground_homography(image, homography,w=4)

        recall, precision, image  = self.get_recall_lines(image,lines_GT_trans, lines_all,weight_all, tol=15)

        if do_debug is not None:
            scale = 0.15
            L_GT, lines_GT = self.get_GT()
            image_pg = self.draw_playground_GT(w=1, R=2,scale=scale,put_text=False)
            image_pg = tools_draw_numpy.draw_convex_hull(image_pg, scale*L_GT[ids],color=color, transperency=transparency)

            image = tools_image.put_image(image, image_pg, 0, 0)
            cv2.imwrite(self.folder_out+'check_homography_%03d_%03d_%03d.png'%(100*precision,100*recall,do_debug),image)

        return homography, precision, recall
# ----------------------------------------------------------------------------------------------------------------------
    def process_view(self, filename_in, do_debug=None):

        image = cv2.imread(filename_in)
        time_start = time.time()
        base_name = filename_in.split('/')[-1].split('.')[0]
        gray = tools_image.desaturate(image)

        self.H,self.W = image.shape[:2]

        grass_mask = self.get_grass_mask(image)
        binarized = self.Skelenonizer.binarize(image)
        skeleton = self.Skelenonizer.binarized_to_skeleton2(binarized)
        skeleton[grass_mask==0] = 0

        lines_vert,weights_vert = self.Hough.get_lines_ski(skeleton,max_count= 4,min_weight=100,the_range=numpy.arange(60*numpy.pi/180, 80*numpy.pi / 180, (numpy.pi/1800)))
        lines_horz,weights_horz = self.Hough.get_lines_ski(skeleton,max_count= 4,min_weight=  0,the_range=numpy.arange(90*numpy.pi/180,130*numpy.pi / 180, (numpy.pi/1800)))
        lines_all = numpy.vstack((lines_vert, lines_horz))
        weight_all = numpy.hstack((weights_vert, weights_horz))

        if do_debug is not None:
            #cv2.imwrite(self.folder_out + base_name + '_1_bin.png', binarized)
            #cv2.imwrite(self.folder_out + base_name + '_2_ske.png', skeleton)
            image_lines = self.draw_lines(tools_image.desaturate(image), lines_horz, color=(0, 128, 255), w=4, put_text=True)
            image_lines = self.draw_lines(image_lines, lines_vert, color=(0,   0, 255), w=4, put_text=True)
            cv2.imwrite(self.folder_out + base_name + '_4_lines_v.png', image_lines)

        combs, idxs = self.create_combinations(lines_vert, lines_horz)

        precisions, recalls,homographies = [],[],[]
        for i,(comb,idx) in enumerate(zip(combs,idxs)):
            lines = numpy.zeros((16,4))
            lines[idx]=comb

            lm,idx_lm = [],[]
            for cnt,(l1, l2) in enumerate(self.get_list_of_lines()):
                if numpy.count_nonzero(lines[l1])>0 and numpy.count_nonzero(lines[l2])>0:
                    idx_lm.append(cnt)
                    lm.append(tools_render_CV.line_intersection(lines[l1], lines[l2]))

            homography, precision, recall = self.get_pose_homography(gray,lm,idx_lm,lines_all,weight_all,do_debug=i)
            recalls.append(recall)
            precisions.append(precision)
            homographies.append(homography)

        idx = numpy.argsort(-numpy.array(precisions))[0]
        result = self.draw_playground_homography(gray, homographies[idx], w=3, R=6)
        cv2.imwrite(self.folder_out + base_name + '.png', result)

        print('%s - %1.2f sec' % (base_name, (time.time() - time_start)))

        return
# ----------------------------------------------------------------------------------------------------------------------