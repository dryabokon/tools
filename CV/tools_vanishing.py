#https://github.com/rayryeng/XiaohuLuVPDetection/blob/master/lu_vp_detect/run_vp_detect.py
#https://github.com/AngeloG98/VanishingPointCameraCalibration
#https://github.com/chsasank/Image-Rectification/blob/master/rectification.py
import math
import pandas as pd
import numpy
import cv2
import inspect
from scipy.spatial import distance as dist
# ---------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_IO
import tools_image
import tools_plot_v2
import tools_render_CV
import tools_render_GL
import tools_time_profiler
import tools_DF
from CV import tools_pr_geom
from CV import tools_alg_match
from CV import tools_Skeletone
# ---------------------------------------------------------------------------------------------------------------------
class detector_VP:
    def __init__(self, folder_out,W=None,H=None):
        self.folder_out = folder_out
        self.Ske = tools_Skeletone.Skelenonizer(folder_out=folder_out)
        self.P = tools_plot_v2.Plotter(folder_out=folder_out)


        self.TP = tools_time_profiler.Time_Profiler()

        self.H = H
        self.W = W
        self.detector = 'ORB'
        self.matchtype = 'knn'
        self.kernel_conv_vert = (7,3)
        self.kernel_conv_horz = (3,7)

        self.taret_ratio_L_W = 4685 / 1814  # vehicle ratio
        self.taret_ratio_H_W = 1449 / 1814  # vehicle ratio
        self.mean_vehicle_length = 4.685  # meters
        self.fov_x_deg_default = 18.55  #deg

        self.mean_lp_length = 0.520 # meters

        self.config_algo_ver_lines = 'LSD'
        self.tol_deg_hor_line = 10
        self.color_markup_grid = (0,0,0)
        self.width_markup_grid = 2
        self.lines_width = 2
        self.color_markup_cuboid = tools_draw_numpy.color_black
        self.transp_markup = 0.85
        self.colors_rag = tools_draw_numpy.get_colors(255,colormap = 'nipy_spectral')[120:240]
        self.font_size = 28
        self.filename_vehicle_3d_obj = './SUV1.obj' #'./temp.obj'

        return
# ---------------------------------------------------------------------------------------------------------------------
    def iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
# ---------------------------------------------------------------------------------------------------------------------
    def keep_lines_by_length(self, lines, len_min=15, len_max=200,inv=False):
        nrm = numpy.array([numpy.linalg.norm(l[:2] - l[2:]) for l in lines])
        idx = (nrm >= len_min) & (nrm <= len_max)
        if inv:
            idx=~idx
        lines_res = numpy.array(lines)[idx]
        return lines_res
# ---------------------------------------------------------------------------------------------------------------------
    def keep_lines_by_angle(self, lines, angle_deg_min, angle_deg_max,inv=False):
        angles = numpy.array([self.get_angle_deg(line) for line in lines])
        idx = (angles >= angle_deg_min) & (angles <= angle_deg_max)
        if inv:
            idx=~idx
        lines_res = numpy.array(lines)[idx]
        return lines_res
# ---------------------------------------------------------------------------------------------------------------------
    def keep_lines_above_cutoff_line(self, lines, line_cutoff, inv=False):
        if lines is None or len(lines)==0: return lines
        idx = numpy.array([tools_render_CV.is_point_above_line(line[:2],line_cutoff) and tools_render_CV.is_point_above_line(line[2:],line_cutoff) for line in lines])
        if inv:
            idx=~idx
        lines_res = numpy.array(lines)[idx]
        return lines_res
# ---------------------------------------------------------------------------------------------------------------------
    def save_lines(self,lines,filename_out):
        pd.DataFrame(lines, columns=['x1', 'y1', 'x2', 'y2']).to_csv(self.folder_out + filename_out, index=False)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def load_lines(self,filename_in):
        lines = pd.read_csv(filename_in).values
        return lines
# ---------------------------------------------------------------------------------------------------------------------
    def reshape_lines_as_paired(self,lines):
        res = numpy.array([[[l[0],l[1]],[l[2],l[3]]] for l in lines])
        return res
# ---------------------------------------------------------------------------------------------------------------------
    def reshape_lines_as_flat(self, lines):
        res = numpy.array([[l[0][0],l[0][1],l[1][0],l[1][1]] for l in lines])
        return res
# ---------------------------------------------------------------------------------------------------------------------
    def get_angle_deg(self, line):
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            angle = 0
        else:
            angle = 90 + math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
        return angle
# ----------------------------------------------------------------------------------------------------------------------
    def boxify_lines(self, lines, box, do_quick_unstable=False, do_debug=False):
        # visible lines are extended into the boarder of the box

        def is_inside_line(p, p1, p2):
            a = numpy.linalg.norm(p1 - p)
            b = numpy.linalg.norm(p2 - p)
            c = numpy.linalg.norm(p1 - p2)
            res = a + b <= c + 0.1
            return res

        def is_inside_box(p, left, top, right, bottom):
            res = (left <= p[0] <= right) and (top <= p[1] <= bottom)
            return res

        # W,H = box[2], box[3]
        tol = 2
        results, idx = [], []
        left, top, right, bottom = box[0], box[1], box[2], box[3]

        segments = [(left, top, right, top), (right, top, right, bottom), (left, bottom, right, bottom),
                    (left, top, left, bottom)]

        for l, line in enumerate(lines):
            if numpy.any(numpy.isnan(line)): continue
            if numpy.linalg.norm(line) == 0: continue
            result = []

            is_in_box = is_inside_box((line[0], line[1]), box[0], box[1], box[2], box[3]) or is_inside_box(
                (line[2], line[3]), box[0], box[1], box[2], box[3])
            if do_quick_unstable:
                x1, y1 = tools_render_CV.line_intersection_unstable(line, segments[0])
                x2, y2 = tools_render_CV.line_intersection_unstable(line, segments[1])
                x3, y3 = tools_render_CV.line_intersection_unstable(line, segments[2])
                x4, y4 = tools_render_CV.line_intersection_unstable(line, segments[3])
            else:
                x1, y1 = tools_render_CV.line_intersection(line, segments[0])
                x2, y2 = tools_render_CV.line_intersection(line, segments[1])
                x3, y3 = tools_render_CV.line_intersection(line, segments[2])
                x4, y4 = tools_render_CV.line_intersection(line, segments[3])

            if x1 is not numpy.nan and y1 is not numpy.nan:
                if left <= x1 + tol and x1 - tol <= right and abs(y1 - top) <= tol:
                    if is_in_box or is_inside_line((x1, y1), line[:2], line[2:]):
                        result.append((x1, top))

            if x2 is not numpy.nan and y2 is not numpy.nan:
                if top <= y2 + tol and y2 - tol <= bottom and abs(x2 - right) <= tol:
                    if is_in_box or is_inside_line((x2, y2), line[:2], line[2:]):
                        result.append((right, y2))

            if x3 is not numpy.nan and y3 is not numpy.nan:
                if left <= x3 + tol and x3 - tol <= right and abs(y3 - bottom) <= tol:
                    if is_in_box or is_inside_line((x3, y3), line[:2], line[2:]):
                        result.append((x3, bottom))

            if x4 is not numpy.nan and y4 is not numpy.nan:
                if top <= y4 + tol and y4 - tol <= bottom and abs(x4 - left) <= tol:
                    if is_in_box or is_inside_line((x4, y4), line[:2], line[2:]):
                        result.append((left, y4))

            if len(result) >= 2:
                results.append((result[0][0], result[0][1], result[1][0], result[1][1]))
                idx.append(l)

            if do_debug:
                image = numpy.full((bottom, right, 3), 64, dtype=numpy.uint8)
                box_p1 = tools_draw_numpy.extend_view((left, top), bottom, right, factor=4)
                box_p2 = tools_draw_numpy.extend_view((right, bottom), bottom, right, factor=4)
                line_p1 = tools_draw_numpy.extend_view((line[0], line[1]), bottom, right, factor=4)
                line_p2 = tools_draw_numpy.extend_view((line[2], line[3]), bottom, right, factor=4)

                circle_p1 = tools_draw_numpy.extend_view((x1, y1), bottom, right, factor=4)
                circle_p2 = tools_draw_numpy.extend_view((x2, y2), bottom, right, factor=4)
                circle_p3 = tools_draw_numpy.extend_view((x3, y3), bottom, right, factor=4)
                circle_p4 = tools_draw_numpy.extend_view((x4, y4), bottom, right, factor=4)

                cv2.rectangle(image, tuple(box_p1), tuple(box_p2), tools_draw_numpy.color_blue, thickness=2)
                cv2.line(image, tuple(line_p1), tuple(line_p2), tools_draw_numpy.color_red, thickness=4)

                if len(result) >= 2:
                    res_p1 = tools_draw_numpy.extend_view((result[0][0], result[0][1]), bottom, right, factor=4)
                    res_p2 = tools_draw_numpy.extend_view((result[1][0], result[1][1]), bottom, right, factor=4)
                    cv2.line(image, tuple(res_p1), tuple(res_p2), tools_draw_numpy.color_amber, thickness=1)

                cv2.circle(image, (circle_p1[0], circle_p1[1]), 10, tools_draw_numpy.color_light_gray, thickness=2)
                cv2.circle(image, (circle_p2[0], circle_p2[1]), 10, tools_draw_numpy.color_light_gray, thickness=2)
                cv2.circle(image, (circle_p3[0], circle_p3[1]), 10, tools_draw_numpy.color_light_gray, thickness=2)
                cv2.circle(image, (circle_p4[0], circle_p4[1]), 10, tools_draw_numpy.color_light_gray, thickness=2)

                cv2.imwrite(self.folder_out + 'boxify.png', image)
                uu = 0

        return numpy.array(results, dtype=numpy.int)
# -----------------------------------------------------------------------
    def get_lines_ver_candidates_static(self,folder_in,df_boxes=None,len_min=15, len_max=200,do_debug=False):

        filenames = tools_IO.get_filenames(folder_in, '*.jpg')
        lines = None

        for filename in filenames:
            image = cv2.imread(folder_in + filename)
            if df_boxes is not None:
                image_mask = numpy.full((image.shape[0],image.shape[1],3),0,dtype=numpy.uint8)
                rects = tools_DF.apply_filter(df_boxes,'ID',filename).iloc[:,3:].values
                rects = [r.reshape((2,2)) for r in rects]
                image_mask = tools_draw_numpy.draw_rects(image_mask, rects, (255,255,255), w=1, alpha_transp=0)
                image = tools_image.put_color_by_mask(image, image_mask, (128,128,128))

            if self.config_algo_ver_lines=='LSD':
                lns = self.Ske.detect_lines_LSD(image)
            else:
                img_amp = self.Ske.preprocess_amplify(image, self.kernel_conv_vert)
                img_bin = self.Ske.binarize(img_amp)
                img_ske = cv2.Canny(image=img_bin, threshold1=20, threshold2=250)
                segments = self.Ske.skeleton_to_segments(img_ske)
                segments_straight = self.Ske.sraighten_segments(segments, min_len=10)
                segments_long = self.Ske.filter_short_segments2(segments_straight, ratio=0.10)
                lns,losses = self.Ske.interpolate_segments_by_lines(segments_long)

            lns = self.keep_lines_by_length(lns, len_min, len_max)
            lns = self.keep_lines_by_angle(lns, 90  - self.tol_deg_hor_line, 90  + self.tol_deg_hor_line, inv=True)
            lns = self.keep_lines_by_angle(lns, 270 - self.tol_deg_hor_line, 270 + self.tol_deg_hor_line, inv=True)
            lns = self.keep_lines_by_angle(lns, 0, 0, inv=True)
            lns = self.keep_lines_by_angle(lns, 180, 180, inv=True)
            image = tools_draw_numpy.draw_lines(image, lns)
            cv2.imwrite(self.folder_out+filename,image)
            ii=0

            if len(lns)>0:
                lines = lns if lines is None else numpy.concatenate([lines,lns],axis=0)

        if do_debug:
            cv2.imwrite(self.folder_out + 'lines_static_ver.png',tools_draw_numpy.draw_lines(tools_image.desaturate(image), lines, color=(0, 0, 200), w=1))

        return lines
# ---------------------------------------------------------------------------------------------------------------------
    def get_lines_ver_candidates_single_image(self,image,len_min=15, len_max=200,do_debug=False):

        lines = None
        if self.config_algo_ver_lines=='LSD':
            lns = self.Ske.detect_lines_LSD(image)
        else:
            img_amp = self.Ske.preprocess_amplify(image, self.kernel_conv_vert)
            img_bin = self.Ske.binarize(img_amp)
            img_ske = cv2.Canny(image=img_bin, threshold1=20, threshold2=250)
            segments = self.Ske.skeleton_to_segments(img_ske)
            segments_straight = self.Ske.sraighten_segments(segments, min_len=10)
            segments_long = self.Ske.filter_short_segments2(segments_straight, ratio=0.10)
            lns,losses = self.Ske.interpolate_segments_by_lines(segments_long)
        lines = lns if lines is None else numpy.concatenate([lines,lns],axis=0)

        lines = self.keep_lines_by_length(lines, len_min, len_max)
        lines = self.keep_lines_by_angle(lines, 90  - self.tol_deg_hor_line, 90  + self.tol_deg_hor_line, inv=True)
        lines = self.keep_lines_by_angle(lines, 270 - self.tol_deg_hor_line, 270 + self.tol_deg_hor_line, inv=True)
        lines = self.keep_lines_by_angle(lines, 0, 0, inv=True)
        lines = self.keep_lines_by_angle(lines, 180, 180, inv=True)

        if do_debug:
            cv2.imwrite(self.folder_out + 'lines_static_ver.png',tools_draw_numpy.draw_lines(tools_image.desaturate(image), lines, color=(0, 0, 200), w=1))
        return lines
# ---------------------------------------------------------------------------------------------------------------------
    def get_lines_ver_candidates_dynamic(self,folder_in,len_min=15,len_max=200,do_debug=False):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        filenames = tools_IO.get_filenames(folder_in,'*.jpg')
        img_cur = cv2.imread(folder_in + filenames[0])

        lines = []
        for filename_cur in filenames[1:200]:
            img_prev = img_cur.copy()
            img_cur  = cv2.imread(folder_in+filename_cur)

            points1, des1 = tools_alg_match.get_keypoints_desc(img_cur, self.detector)
            points2, des2 = tools_alg_match.get_keypoints_desc(img_prev, self.detector)
            match1, match2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2,self.matchtype)
            for m1, m2 in zip(match1, match2):
                if numpy.linalg.norm(m1-m2)<len_min:continue
                if numpy.linalg.norm(m1-m2)>len_max:continue
                lines.append([m1[0], m1[1], m2[0], m2[1]])

        lines = numpy.array(lines).reshape((-1,4))
        if do_debug:
            cv2.imwrite(self.folder_out + 'lines_dynamic_ver.png',
                        tools_draw_numpy.draw_lines(tools_image.desaturate(img_cur), lines, color=(0, 0, 200), w=1))
        self.save_lines(lines,'lines_ver_dyn.csv')
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return lines

# ----------------------------------------------------------------------------------------------------------------------
    def get_lines_hor_candidates_static(self, folder_in,df_boxes=None, len_min=15,len_max=200,do_debug=False):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        filenames = tools_IO.get_filenames(folder_in, '*.jpg')
        lines = None

        tol_h = 20

        for filename in filenames[:100]:
            image = cv2.imread(folder_in + filename)
            H, W = image.shape[:2]
            if df_boxes is not None:
                image_mask = numpy.full((image.shape[0],image.shape[1],3),0,dtype=numpy.uint8)
                rects = tools_DF.apply_filter(df_boxes,'ID',filename).iloc[:,3:].values
                rects = [r.reshape((2,2)) for r in rects]
                image_mask = tools_draw_numpy.draw_rects(image_mask, rects, (255,255,255), w=1, alpha_transp=0)
                image = tools_image.put_color_by_mask(image, image_mask, (128,128,128))

            # image_skeleton = cv2.Canny(image=image, threshold1=20, threshold2=250)
            # segments = self.Ske.skeleton_to_segments(image_skeleton)
            # segments_straight = self.Ske.sraighten_segments(segments, min_len=20)
            # segments_long = self.Ske.filter_short_segments2(segments_straight, ratio=0.10)
            # lns = self.Ske.interpolate_segments_by_lines(segments_long)
            lns = self.Ske.detect_lines_LSD(image)

            if len(lns)>0:
                lns = self.keep_lines_by_length(lns, len_min, len_max)
                lns = numpy.concatenate([self.keep_lines_by_angle(lns, 270 - self.tol_deg_hor_line, 270 + self.tol_deg_hor_line),self.keep_lines_by_angle(lns, 90 - self.tol_deg_hor_line, 90 + self.tol_deg_hor_line)])
                lns = self.keep_lines_above_cutoff_line(lns, (0, H - tol_h, W, H - tol_h))
                lines = lns if lines is None else numpy.concatenate([lines, lns], axis=0)
                #image = tools_draw_numpy.draw_lines(tools_image.desaturate(image), lns)
                #cv2.imwrite(self.folder_out+filename,image)


        if do_debug:
            image = cv2.imread(folder_in + filenames[0])
            cv2.imwrite(self.folder_out + 'lines_static_hor.png',tools_draw_numpy.draw_lines(tools_image.desaturate(image), lines, color=(0, 0, 200), w=1))
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return lines

# ----------------------------------------------------------------------------------------------------------------------
    def get_lines_hor_candidates_single_image(self, image, len_min=15,len_max=200,do_debug=False):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        lines = None
        tol_h = 20
        lns = self.Ske.detect_lines_LSD(image)
        lines = lns if lines is None else numpy.concatenate([lines, lns], axis=0)

        H,W = image.shape[:2]
        lines = self.keep_lines_by_length(lines, len_min, len_max)
        lines = numpy.concatenate([self.keep_lines_by_angle(lines, 270-self.tol_deg_hor_line, 270 + self.tol_deg_hor_line),self.keep_lines_by_angle(lines, 90-self.tol_deg_hor_line, 90+self.tol_deg_hor_line)])
        lines = self.keep_lines_above_cutoff_line(lines, (0,H-tol_h,W,H-tol_h))

        if do_debug:
            cv2.imwrite(self.folder_out + 'lines_static_hor.png',
                        tools_draw_numpy.draw_lines(tools_image.desaturate(image), lines, color=(0, 0, 200), w=1))
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return lines
# ----------------------------------------------------------------------------------------------------------------------
    def get_focal_length(self, vps):

        vp_v = vps[0]
        vp_h = vps[1]

        vp_v_x = vp_v[0]
        vp_v_y = vp_v[1]
        vp_h_x = vp_h[0]
        vp_h_y = vp_h[1]

        #vp_h_y+=1000

        #Focal length of the camera in pixels
        pp = [self.W / 2, self.H / 2]
        if vp_v_x == vp_h_x:return math.fabs(pp[0] - vp_v_x)
        if vp_v_y == vp_h_y:return math.fabs(pp[1] - vp_v_y)
        k_uv = (vp_v_y - vp_h_y) / (vp_v_x - vp_h_x)
        b_uv =  vp_h_y - k_uv * vp_h_x
        pp_uv = math.fabs(k_uv * pp[0] - pp[1] + b_uv) / math.pow(k_uv * k_uv + 1, 0.5)
        lenth_uv = math.sqrt((vp_v_y - vp_h_y)**2 + (vp_v_x - vp_h_x)**2)
        lenth_pu = math.sqrt((vp_v_y -  pp[1])**2 + (vp_v_x - pp[0])**2)
        up_uv = math.sqrt(lenth_pu ** 2 - pp_uv ** 2)
        vp_uv = abs(lenth_uv - up_uv)
        dd = (up_uv * vp_uv) - ((pp_uv) ** 2)
        if dd>0:
            focal_length = math.sqrt(dd)
        else:
            focal_length = math.sqrt(-dd)
            #focal_length = 0.5 * self.W /numpy.tan(0.5*self.fov_x_deg_default *numpy.pi / 180)

        return focal_length
# -----------------------------------------------------------------------
    def calculate_metric_angle(self, current_hypothesis, lines, ignore_pts, ransac_angle_thresh):
        current_hypothesis = current_hypothesis / current_hypothesis[-1]
        hypothesis_vp_direction = current_hypothesis[:2] - lines[:, 0]
        lines_vp_direction = lines[:, 1] - lines[:, 0]
        magnitude = numpy.linalg.norm(hypothesis_vp_direction, axis=1) * numpy.linalg.norm(lines_vp_direction, axis=1)
        magnitude[magnitude == 0] = 1e-5
        cos_theta = (hypothesis_vp_direction * lines_vp_direction).sum(axis=-1) / magnitude
        theta = numpy.arccos(numpy.abs(numpy.clip(cos_theta,-1,1)))
        inliers = (theta < ransac_angle_thresh * numpy.pi / 180)
        inliers[ignore_pts] = False
        return inliers, inliers.sum()
# ----------------------------------------------------------------------------------------------------------------------
    def run_line_ransac(self,lines, ransac_iter=3000, ransac_angle_thresh=2.0, ignore_pts=None):
        best_vote_count = 0
        idx_best_inliers = None
        best_hypothesis = None
        if ignore_pts is None:
            ignore_pts = numpy.zeros((lines.shape[0])).astype('bool')
            lines_to_chose = numpy.arange(lines.shape[0])
        else:
            lines_to_chose = numpy.where(ignore_pts == 0)[0]
        for iter_count in range(ransac_iter):
            idx1, idx2 = numpy.random.choice(lines_to_chose, 2, replace=False)
            l1 = numpy.cross(numpy.append(lines[idx1][1], 1), numpy.append(lines[idx1][0], 1))
            l2 = numpy.cross(numpy.append(lines[idx2][1], 1), numpy.append(lines[idx2][0], 1))

            current_hypothesis = numpy.cross(l1, l2)
            if current_hypothesis[-1] == 0:
                continue
            idx_inliers, vote_count = self.calculate_metric_angle(current_hypothesis, lines, ignore_pts, ransac_angle_thresh)
            if vote_count > best_vote_count:
                best_vote_count = vote_count
                best_hypothesis = current_hypothesis
                idx_best_inliers = idx_inliers
        return best_hypothesis/best_hypothesis[-1], idx_best_inliers
# ----------------------------------------------------------------------------------------------------------------------
    def get_vp(self,lines,filename_out=None,image_debug=None):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        vp1, idx_vl1 = self.run_line_ransac(lines)
        vp1_lines = self.reshape_lines_as_flat(lines[idx_vl1])

        if vp1[0]>=0 and vp1[0]<=self.W:
            vp1_lines_add  = numpy.array([(vp1[0], vp1[1], x, self.H - 1) for x in numpy.linspace(0, self.W, 10)])
            vp1_lines_add2 = numpy.array([(vp1[0], vp1[1], 0, y         ) for y in numpy.linspace(0, self.H, 10)])
            vp1_lines_add3 = numpy.array([(vp1[0], vp1[1], self.W-1,   y) for y in numpy.linspace(0, self.H, 10)])
            vp1_lines_add  = numpy.concatenate([vp1_lines_add,vp1_lines_add2,vp1_lines_add3],axis=0)
        else:
            vp1_lines_add  = numpy.array([(vp1[0], vp1[1], self.W-1 if vp1[0]<0 else 0, y) for y in numpy.linspace(0, self.H, 20)])

        if filename_out is not None:
            if image_debug is None:
                image_debug = numpy.full((self.H,self.W,3),32,dtype=numpy.uint8)
            factor = 3
            H, W = self.H, self.W
            image_ext = tools_draw_numpy.extend_view_from_image(tools_image.desaturate(image_debug), factor)
            image_ext = tools_draw_numpy.draw_lines(image_ext, tools_draw_numpy.extend_view(vp1_lines, H, W, factor), w=1)
            image_ext = tools_draw_numpy.draw_lines(image_ext, tools_draw_numpy.extend_view(vp1_lines_add, H, W, factor),color=(0,100,200),w=1)
            image_ext = tools_draw_numpy.draw_points(image_ext,tools_draw_numpy.extend_view(vp1[:2], H, W, factor),color=(255, 64, 0), w=8)


            cv2.imwrite(self.folder_out + filename_out, image_ext)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return vp1,vp1_lines

# ----------------------------------------------------------------------------------------------------------------------
    def build_BEV_by_fov_van_point(self,image, cam_fov_x_deg,cam_fov_y_deg,point_van_xy_ver,point_van_xy_hor=None,do_rotation=True):

        if isinstance(image,str):
            image = tools_image.desaturate(cv2.imread(image), level=0)

        h_ipersp,target_BEV_W, target_BEV_H,rot_deg = self.get_inverce_perspective_mat_v3(image, cam_fov_x_deg,point_van_xy_ver,point_van_xy_hor)
        edges = numpy.array(([0, self.H*3/4], [0, self.H], [self.W, self.H*3/4], [self.W, self.H])).astype(numpy.float32)

        if do_rotation:
            mat_R = tools_image.get_image_affine_rotation_mat(image, rot_deg, reshape=True)
            h_ipersp = numpy.matmul(numpy.concatenate([mat_R,numpy.array([0,0,1.0]).reshape((1,-1))],axis=0),h_ipersp)
            edges_BEV = cv2.perspectiveTransform(edges.reshape((-1,1,2)), h_ipersp).reshape((-1, 2))
            target_BEV_W, target_BEV_H = numpy.max(edges_BEV, axis=0)

        image_BEV = cv2.warpPerspective(image, h_ipersp, (int(target_BEV_W), int(target_BEV_H)), borderValue=(32, 32, 32))
        center_BEV = cv2.perspectiveTransform(numpy.array(((self.W / 2, self.H / 2))).reshape((-1, 1, 2)),h_ipersp).reshape((-1, 2))[0]
        edges_BEV = cv2.perspectiveTransform(edges.reshape((-1, 1, 2)), h_ipersp)
        lines_edges_BEV = edges_BEV.reshape((-1, 4))

        p_camera_BEV_xy = tools_render_CV.line_intersection(numpy.array(lines_edges_BEV[0]),numpy.array(lines_edges_BEV[1]))
        cam_abs_offset = p_camera_BEV_xy[1] - image_BEV.shape[0]
        center_offset = image_BEV.shape[0] - center_BEV[1]
        cam_height = self.evaluate_cam_height(cam_abs_offset, center_offset, cam_fov_y_deg)

        # image_BEV = tools_draw_numpy.draw_points(image_BEV, [center_BEV], w=10,color=(0, 0, 200))
        # image_BEV = tools_draw_numpy.draw_lines(image_BEV, lines_edges_BEV, w=3, color=(0, 0, 200))

        return image_BEV, h_ipersp, cam_height, p_camera_BEV_xy, center_BEV,lines_edges_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def build_BEV_by_fov_van_point_v2(self, image, cam_fov_x_deg, cam_fov_y_deg, point_van_xy_ver,point_van_xy_hor=None, do_rotation=True):

        if isinstance(image, str):
            image = tools_image.desaturate(cv2.imread(image), level=0)

        h_ipersp, target_BEV_W, target_BEV_H, rot_deg = self.get_inverce_perspective_mat_v3(image, cam_fov_x_deg,point_van_xy_ver,point_van_xy_hor)
        image_BEV = cv2.warpPerspective(image, h_ipersp, (int(target_BEV_W), int(target_BEV_H)),borderValue=(32, 32, 32))

        edges = numpy.array(([0, self.H*3/4], [0, self.H], [self.W, self.H*3/4], [self.W, self.H])).astype(numpy.float32)
        edges_BEV = cv2.perspectiveTransform(edges.reshape((-1, 1, 2)), h_ipersp)
        center_BEV = cv2.perspectiveTransform(numpy.array(((self.W / 2, self.H / 2))).reshape((-1, 1, 2)), h_ipersp).reshape((-1, 2))[0]
        # if center_BEV[1]<0:
        #     center_BEV[1]*=-1


        lines_edges_BEV = edges_BEV.reshape((-1, 4))
        p_camera_BEV_xy = tools_render_CV.line_intersection(numpy.array(lines_edges_BEV[0]), numpy.array(lines_edges_BEV[1]))
        cam_abs_offset = p_camera_BEV_xy[1] - image_BEV.shape[0]
        center_offset = image_BEV.shape[0] - center_BEV[1]
        cam_height = self.evaluate_cam_height(cam_abs_offset, center_offset, cam_fov_y_deg)

        if do_rotation:
            mat_R = tools_image.get_image_affine_rotation_mat(image, rot_deg, reshape=True)
            h_ipersp = numpy.matmul(numpy.concatenate([mat_R, numpy.array([0, 0, 1.0]).reshape((1, -1))], axis=0),h_ipersp)
            image_BEV = tools_image.rotate_image(image_BEV, rot_deg, reshape=True, borderValue=(32, 32, 32))
            #image_BEV = cv2.warpPerspective(image, h_ipersp, (int(target_BEV_W), int(target_BEV_H)),borderValue=(32, 32, 32))
            center = (image_BEV.shape[1] / 2, image_BEV.shape[0] / 2)
            p_camera_BEV_xy = tools_image.rotate_point(p_camera_BEV_xy, center, rot_deg, reshape=True)
            center_BEV = tools_image.rotate_point(center_BEV, center, rot_deg, reshape=True)
            lines_edges_BEV = numpy.array([tools_image.rotate_point(p, center, rot_deg, reshape=True) for p in lines_edges_BEV.reshape((-1, 2))]).reshape((-1, 4))

        image_BEV = tools_draw_numpy.draw_points(image_BEV, [center_BEV],w=10, color=(0, 0, 200))
        image_BEV = tools_draw_numpy.draw_lines(image_BEV, lines_edges_BEV, w=3, color=(0, 0, 200))

        return image_BEV, h_ipersp, cam_height, p_camera_BEV_xy, center_BEV, lines_edges_BEV

# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_cam_height(self,cam_abs_offset,center_offset,fov_y_deg):
        loss_g = numpy.inf
        cam_height = numpy.nan
        for h in numpy.arange(0.1, cam_abs_offset + center_offset, 0.05):
            beta  = numpy.arctan( cam_abs_offset                 /h)*180/numpy.pi
            alpha = numpy.arctan((cam_abs_offset + center_offset)/h)*180/numpy.pi
            loss = abs((alpha-beta) - fov_y_deg/2)
            if loss < loss_g:
                loss_g = loss
                cam_height = h

        #check
        # a_pitch1 = numpy.arctan( cam_abs_offset                    / cam_height) * 180 / numpy.pi
        # a_pitch2 = numpy.arctan((cam_abs_offset +   center_offset) / cam_height) * 180 / numpy.pi
        # fact = a_pitch2-a_pitch1
        # target = fov_y_deg/2

        return cam_height
# ----------------------------------------------------------------------------------------------------------------------
    def get_RAG_by_yaw(self,angle_deg,tol_min=1,tol_max=10):
        loss = abs(angle_deg)
        if loss>=tol_max:
            return self.colors_rag[-1]
        elif loss<=tol_min:
            return self.colors_rag[0]
        else:
            L = self.colors_rag.shape[0]
            return self.colors_rag[int((loss - tol_min) / (tol_max - tol_min) * L)]
# ----------------------------------------------------------------------------------------------------------------------
    def draw_grid_at_BEV(self, image_BEV, p_camera_BEV_xy, p_center_BEV_xy, lines_edges,fov_x_deg,fov_y_deg):
        cam_abs_offset = p_camera_BEV_xy[1]-image_BEV.shape[0]
        center_offset  = image_BEV.shape[0]-p_center_BEV_xy[1]
        cam_height = self.evaluate_cam_height(cam_abs_offset,center_offset,fov_y_deg)
        image_res = image_BEV

        for a_yaw in range(0, int(fov_x_deg / 2)):
            delta = p_camera_BEV_xy[1] * numpy.tan(a_yaw * numpy.pi / 180.0)
            lines_yaw = [[p_camera_BEV_xy[0] - delta, 0, p_camera_BEV_xy[0], p_camera_BEV_xy[1]],[p_camera_BEV_xy[0] + delta, 0, p_camera_BEV_xy[0], p_camera_BEV_xy[1]]]
            image_res = tools_draw_numpy.draw_lines(image_res, lines_yaw, color=self.color_markup_grid, w=1, transperency=self.transp_markup)

        res_horizontal_pitch = []
        res_vertical_yaw = []
        for a_pitch in range(1,90):
            radius = cam_height*numpy.tan(a_pitch*numpy.pi/180.0)
            center = image_BEV.shape[0]+cam_abs_offset
            row =  center-radius
            if row<0 or row>=image_BEV.shape[0]: continue
            p = (p_camera_BEV_xy[0]-radius,center-radius,p_camera_BEV_xy[0]+radius,center+radius)
            image_res = tools_draw_numpy.draw_ellipse(image_res, p, color=None, col_edge=self.color_markup_grid, transperency=self.transp_markup)
            image_res = tools_draw_numpy.draw_text(image_res,'%d' % (90-a_pitch)+u'\u00B0', (p_camera_BEV_xy[0],row-6), color_fg=self.color_markup_grid,font_size=self.font_size)


            p1 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_edges[0,2:],lines_edges[0,:-2], full_line=False)
            p2 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_edges[1,2:],lines_edges[1,:-2],full_line=False)
            if len(p1)>0 and len(p2)>0:
                res_horizontal_pitch.append([90-a_pitch,p1[0][0],p1[0][1],p_center_BEV_xy[0],row,p2[0][0],p2[0][1]])

            for a_yaw in range(0,int(fov_x_deg/2)):
                delta = p_camera_BEV_xy[1]*numpy.tan(a_yaw*numpy.pi/180.0)
                lines_yaw = numpy.array([[p_camera_BEV_xy[0]-delta,0,p_camera_BEV_xy[0],p_camera_BEV_xy[1]],[p_camera_BEV_xy[0]+delta,0,p_camera_BEV_xy[0],p_camera_BEV_xy[1]]])
                p1 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_yaw[0, 2:], lines_yaw[0, :-2],full_line=False)
                p2 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_yaw[1, 2:], lines_yaw[1, :-2],full_line=False)
                if len(p1) > 0 and len(p2) > 0:
                    res_vertical_yaw.append([+a_yaw, p1[0][0], p1[0][1]])
                    res_vertical_yaw.append([-a_yaw, p2[0][0], p2[0][1]])

        df_horizontal = pd.DataFrame(res_horizontal_pitch)
        df_vertical = pd.DataFrame(res_vertical_yaw)
        return image_res, df_horizontal,df_vertical
# ----------------------------------------------------------------------------------------------------------------------
    def draw_meters_at_BEV(self,image_BEV,p_camera_BEV_xy,pix_per_meter_BEV,pad=50):

        rows = []
        for dist_m in numpy.arange(0,200,10):
            row = p_camera_BEV_xy[1]-dist_m*pix_per_meter_BEV
            if row>0 and row<image_BEV.shape[0]:
                rows.append(row)
                image_BEV = tools_draw_numpy.draw_text(image_BEV,'%d m'%dist_m,(pad+12,row), color_fg=(255,255,255),font_size=self.font_size)

        clr = int(128)
        for r1,r2 in zip(rows[1:],rows[:-1]):
            clr = 128 if clr==255 else 255
            image_BEV[int(r1):int(r2),pad:pad+10,:] = clr

        clr = 128 if clr == 255 else 255
        image_BEV[int(rows[0]): , pad:pad+10, :] = 128
        image_BEV[:int(rows[-1]), pad:pad+10, :] = clr




        return image_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def draw_boxes_at_BEV(self,image_BEV,h_ipersp,df_points):

        if df_points.shape[0]==0:return image_BEV

        boxes = df_points.iloc[:, -4:].values
        lines = []
        for box in boxes:
            lines.append([box[0], box[1], box[0], box[3]])
            lines.append([box[0], box[1], box[2], box[1]])
            lines.append([box[2], box[3], box[2], box[1]])
            lines.append([box[2], box[3], box[0], box[3]])

        points_BEV = cv2.perspectiveTransform(numpy.array(lines).astype(numpy.float32).reshape((-1, 1, 2)), h_ipersp).reshape((-1,2))
        image_BEV = tools_draw_numpy.draw_points(image_BEV,points_BEV,w=4)

        return image_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def draw_footprints_at_BEV(self, image_BEV, h_ipersp, df_footprints,df_metadata=None):

        if df_footprints.shape[0] == 0: return image_BEV
        color = (0,0,0)
        for r in range(df_footprints.shape[0]):
            points = df_footprints.iloc[r, :].values
            if df_metadata is not None:
                color = self.get_RAG_by_yaw(df_metadata.iloc[r,0])
            points_BEV = cv2.perspectiveTransform(numpy.array(points).astype(numpy.float32).reshape((-1, 1, 2)),h_ipersp).reshape((-1, 2))
            image_BEV = tools_draw_numpy.draw_contours(image_BEV, points_BEV.reshape((-1,2)), color=color, w=self.lines_width+1, transperency=0.60)

        return image_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def draw_grid_at_original(self, image, df_keypoints_hor,df_keypoints_ver,h_ipersp):
        image_res = image.copy()
        for r in range(df_keypoints_hor.shape[0]):
            x = df_keypoints_hor.iloc[r,[1,3,5]].values
            y = df_keypoints_hor.iloc[r,[2,4,6]].values
            label = str(df_keypoints_hor.iloc[r,0])+u'\u00B0'

            points_bev = numpy.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
            points = cv2.perspectiveTransform(points_bev.reshape((-1, 1, 2)), numpy.linalg.inv(h_ipersp)).reshape((-1,2))

            lines = tools_draw_numpy.interpolate_points_by_curve(points)
            image_res = tools_draw_numpy.draw_lines(image_res, lines, color=self.color_markup_grid, transperency=self.transp_markup,w=self.width_markup_grid)
            image_res = tools_draw_numpy.draw_text(image_res, label, (points[0]+points[-1]) / 2, color_fg=self.color_markup_grid, font_size=self.font_size)


        for y in df_keypoints_ver.iloc[:,0].unique():
            df =  tools_DF.apply_filter(df_keypoints_ver,df_keypoints_ver.columns[0],y)
            points_bev = df.iloc[:,1:].values
            points = cv2.perspectiveTransform(points_bev.reshape((-1, 1, 2)), numpy.linalg.inv(h_ipersp)).reshape((-1, 2))
            lines = tools_draw_numpy.interpolate_points_by_curve(points,trans=True)
            image_res = tools_draw_numpy.draw_lines(image_res, lines, color=self.color_markup_grid, transperency=self.transp_markup)

        return image_res
# ----------------------------------------------------------------------------------------------------------------------
    def drawes_boxes_at_original(self,image,df_points,df_metadata=None):
        for r in range(df_points.shape[0]):
            box = df_points.iloc[r,:].values
            image = tools_draw_numpy.draw_rect(image, box[0], box[1], box[2], box[3], color=(0,0,0), alpha_transp=1)
            if df_metadata is not None:
                label = ' '.join(['%s %.1f'%(d,v) for d,v in zip(df_metadata.columns.values,df_metadata.iloc[r,:].values)])
                color = self.get_RAG_by_yaw(df_metadata.iloc[r, 0])
                #image = tools_draw_numpy.draw_text(image,label,(box[0], box[1]), color_fg=(0,0,0),clr_bg=color,font_size=16)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def cut_boxes_at_original(self,image,df_points):

        boxes = df_points.iloc[:, -4:].values
        mask = numpy.full((image.shape[0],image.shape[1]),1,dtype=numpy.uint8)
        for box in boxes.astype(numpy.int):
            image[box[1]:box[3],box[0]:box[2]]=0
            mask[box[1]:box[3],box[0]:box[2]]=0

        return image,mask
# ----------------------------------------------------------------------------------------------------------------------
    def draw_cuboids_at_original(self,image, df_cuboids,idx_mode=0,color=None,df_metadata=None):

        for r in range(df_cuboids.shape[0]):
            cuboid = df_cuboids.iloc[r,:].values
            if df_metadata is not None:
                if color is None:
                    color = self.get_RAG_by_yaw(df_metadata.iloc[r, 0])

                label = ' '.join(['%s%.0f'%(d,v)+u'\u00B0'+' ' for d,v in zip(df_metadata.columns.values,df_metadata.iloc[r,:].values)])
                image = tools_draw_numpy.draw_cuboid(image, cuboid.reshape((-1, 2)), idx_mode=idx_mode,color=color,w=3)
                image = tools_draw_numpy.draw_text(image,label,(cuboid[0], cuboid[1]+self.font_size), color_fg=(0,0,0),clr_bg=color,font_size=self.font_size)
            else:
                image = tools_draw_numpy.draw_cuboid(image, cuboid.reshape((-1, 2)),idx_mode=idx_mode,color=color,w=self.lines_width)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_contour_at_original(self,image, points_2d,color=None):

        for pp in points_2d:
            image = tools_draw_numpy.draw_convex_hull(image, pp, color=color,transperency=0.50)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def prepare_cuboids_data(self, vp_ver, vp_hor, fov_x_deg, fov_y_deg, df_boxes_cars):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        image_bg = numpy.full((self.H, self.W, 3),0,dtype=numpy.uint8)
        image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.build_BEV_by_fov_van_point(image_bg, fov_x_deg, fov_y_deg,vp_ver,vp_hor,do_rotation=True)
        df_objects = self.evaluate_objects(h_ipersp, df_boxes_cars, vp_ver, vp_hor, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return df_objects
# ----------------------------------------------------------------------------------------------------------------------
    def prepare_angles_data(self, vp_ver, vp_hor, fov_x_deg, fov_y_deg, df_boxes_lps):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        image_bg = numpy.full((self.H, self.W, 3),0,dtype=numpy.uint8)
        image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.build_BEV_by_fov_van_point(image_bg, fov_x_deg, fov_y_deg,vp_ver,vp_hor,do_rotation=True)
        df_objects = self.evaluate_angles(h_ipersp, df_boxes_lps, vp_ver, vp_hor, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return df_objects
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_pix_per_meter_BEV_cars(self, df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor):

        image_bg = numpy.full((self.H,self.W,3),0,dtype=numpy.uint8)
        image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.build_BEV_by_fov_van_point(image_bg, fov_x_deg, fov_y_deg,vp_ver,vp_hor, do_rotation=True)
        points = df_objects.iloc[:,7:15]
        points_BEV = cv2.perspectiveTransform(numpy.array(points).astype(numpy.float32).reshape((-1, 1, 2)),h_ipersp).reshape((-1, 2*4))
        ws,hs = [],[]
        for r in range(points_BEV.shape[0]):
            p1 = points_BEV[r,[0,1]]
            p2 = points_BEV[r,[2,3]]
            p3 = points_BEV[r,[4,5]]
            p4 = points_BEV[r,[6,7]]
            d1 = numpy.linalg.norm(p1-p2)
            d2 = numpy.linalg.norm(p2-p3)
            d3 = numpy.linalg.norm(p3-p4)
            d4 = numpy.linalg.norm(p4-p1)
            ws.append(d1)
            ws.append(d3)
            hs.append(d2)
            hs.append(d4)

        pix_per_meter_BEV = (sum(hs)/len(hs))/(self.mean_vehicle_length)

        return pix_per_meter_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_pix_per_meter_BEV_lps(self, df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor):

        image_bg = numpy.full((self.H, self.W, 3), 0, dtype=numpy.uint8)
        image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.build_BEV_by_fov_van_point(image_bg, fov_x_deg, fov_y_deg, vp_ver, vp_hor, do_rotation=True)
        points = df_objects.iloc[:, 3:7]
        points_BEV = cv2.perspectiveTransform(numpy.array(points).astype(numpy.float32).reshape((-1, 1, 2)),h_ipersp).reshape((-1, 4))
        ls = []
        for r in range(points_BEV.shape[0]):
            p1 = points_BEV[r, [0, 1]]
            p2 = points_BEV[r, [2, 3]]
            d1 = abs(p1[0]-p2[0])
            ls.append(d1)

        pix_per_meter_BEV = (sum(ls) / len(ls)) / (self.mean_lp_length)

        return pix_per_meter_BEV

# ----------------------------------------------------------------------------------------------------------------------
    def flip_yaw_180(self, yaw_deg):
        if yaw_deg>90 and yaw_deg<=180:
            yaw_deg += 180
        if yaw_deg >= 180 and yaw_deg < 270:
            yaw_deg -= 180

        return yaw_deg
# ----------------------------------------------------------------------------------------------------------------------
    def standartize_yaw(self, yaw_deg):
        while yaw_deg<0:yaw_deg+=360
        if yaw_deg>360:yaw_deg-=360
        yaw_deg = self.flip_yaw_180(yaw_deg)
        if yaw_deg > 270 and yaw_deg <360:yaw_deg = yaw_deg-360
        return yaw_deg
# ----------------------------------------------------------------------------------------------------------------------
    def calc_footprint(self, box, cam_height, p_camera_BEV_xy, p_center_BEV_xy, point_bottom_left, point_bottom_right, point_top_right, point_top_left, points_BEV):
        cuboid_h = min(point_top_right[1], point_top_left[1]) - min(box[1], box[3])
        cuboid_w = point_bottom_right[0] - point_bottom_left[0]

        footprint = numpy.array([point_bottom_left, point_bottom_right, point_top_right, point_top_left]).reshape(1, -1)
        roofprint = footprint - numpy.array([(0, cuboid_h), (0, cuboid_h), (0, cuboid_h), (0, cuboid_h)]).reshape(1, -1)
        yaw1 = self.get_angle_deg((points_BEV[0][0], points_BEV[0][1], points_BEV[3][0], points_BEV[3][1]))
        yaw2 = self.get_angle_deg((points_BEV[1][0], points_BEV[1][1], points_BEV[2][0], points_BEV[2][1]))
        yaw_ego = (yaw1 + yaw2) / 2
        yaw_ego = self.flip_yaw_180(yaw_ego)
        yaw_cam = numpy.arctan((0.5 * (points_BEV[0][0] + points_BEV[1][0]) - p_center_BEV_xy[0]) / (p_camera_BEV_xy[1] - 0.5 * (points_BEV[0][1] + points_BEV[1][1]))) * 180 / numpy.pi
        yaw_res = -yaw_ego + yaw_cam  # with ego-compensation
        yaw_res = self.standartize_yaw(yaw_res)
        pitch_cam = 90 - numpy.arctan((p_camera_BEV_xy[1] - 0.5 * (points_BEV[0][1] + points_BEV[1][1])) / cam_height) * 180 / numpy.pi

        car_L_px = numpy.linalg.norm(points_BEV[0] - points_BEV[-1])
        car_W_px = numpy.linalg.norm(points_BEV[0] - points_BEV[1])
        car_H_px = car_W_px * cuboid_h / cuboid_w / numpy.cos(pitch_cam * numpy.pi / 180)
        metadata = numpy.array([yaw_res, pitch_cam, car_L_px, car_W_px, car_H_px]).reshape((1, -1))
        points_BEV_best = points_BEV.copy().reshape((1, -1))

        # image = cv2.imread(self.folder_out+'frames/00018.jpg')
        # image_cand = tools_draw_numpy.draw_points(image, [point_bottom_left,point_bottom_right,point_top_right,point_top_left])
        # image_cand = tools_draw_numpy.draw_contours(image_cand,numpy.array([point_bottom_left,point_bottom_right,point_top_right,point_top_left]), color=(0,0,200),transperency=0.75)
        # cv2.imwrite(self.folder_out+'F_%02d_%03d.png'%(0,point_top_right[1]),image_cand)
        return  footprint, roofprint, metadata, points_BEV_best
# ----------------------------------------------------------------------------------------------------------------------
    def calc_angles(self, point, cam_height, p_camera_BEV_xy, p_center_BEV_xy, point_BEV):

        yaw_ego = 0
        yaw_cam = numpy.arctan((point_BEV[0] - p_center_BEV_xy[0]) / (p_camera_BEV_xy[1] - point_BEV[1])) * 180 / numpy.pi
        yaw_res = -yaw_ego + yaw_cam  # with ego-compensation
        yaw_res = self.standartize_yaw(yaw_res)
        pitch_cam = 90 - numpy.arctan((p_camera_BEV_xy[1]-point_BEV[1]) / cam_height) * 180 / numpy.pi
        return yaw_res,pitch_cam
# ----------------------------------------------------------------------------------------------------------------------
    def box_to_footprint_look_upright(self,box,vp_ver, vp_hor, cam_height, p_camera_BEV_xy,p_center_BEV_xy,h_ipersp):
        footprint, roofprint, metadata, points_BEV_best, dims, ratio_best = None, None, None, None, None, None
        for point_top_right_y in range(min(box[1], box[3]), max(box[1], box[3])):
            point_top_right = (max(box[0], box[2]), point_top_right_y)
            point_bottom_right = tools_render_CV.line_intersection((min(box[0],box[2]),max(box[1],box[3]),max(box[0],box[2]),max(box[1],box[3])), (vp_ver[0], vp_ver[1], point_top_right[0], point_top_right[1]))
            point_bottom_left  = tools_render_CV.line_intersection((min(box[0],box[2]),min(box[1],box[3]),min(box[0],box[2]),max(box[1],box[3])), (vp_hor[0], vp_hor[1], point_bottom_right[0], point_bottom_right[1]))
            point_top_left     = tools_render_CV.line_intersection((vp_hor[0], vp_hor[1], point_top_right[0], point_top_right[1]),(vp_ver[0], vp_ver[1], point_bottom_left[0], point_bottom_left[1]))
            points_BEV = cv2.perspectiveTransform(numpy.array([point_bottom_left, point_bottom_right, point_top_right, point_top_left]).astype(numpy.float32).reshape((-1, 1, 2)), h_ipersp).reshape((-1, 2))

            ratio = abs(points_BEV[0][1] - points_BEV[-1][1]) / (abs(points_BEV[0][0] - points_BEV[1][0]) + 1e-4)
            if ratio_best is None or abs(ratio - self.taret_ratio_L_W) < abs(ratio_best - self.taret_ratio_L_W):
                ratio_best = ratio
                footprint, roofprint, metadata, points_BEV_best = self.calc_footprint(box, cam_height, p_camera_BEV_xy, p_center_BEV_xy, point_bottom_left, point_bottom_right, point_top_right, point_top_left, points_BEV)


        cols = ['cuboid%02d' % i for i in range(16)] + ['yaw_cam_car', 'pitch_cam','L','W','H'] + ['p_bev%02d' % i for i in range(8)]
        df = pd.DataFrame(numpy.concatenate((footprint, roofprint, metadata, points_BEV_best), axis=1).reshape((1, -1)),columns=cols)
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def box_to_footprint_look_upleft(self,box,vp_ver, vp_hor, cam_height, p_camera_BEV_xy,p_center_BEV_xy,h_ipersp):

        footprint, roofprint, metadata, points_BEV_best,dims,ratio_best = None,None,None,None,None,None
        for point_top_left_y in range(min(box[1], box[3]), max(box[1], box[3])):
            point_top_left = (min(box[0], box[2]), point_top_left_y)
            point_bottom_left  = tools_render_CV.line_intersection((min(box[0],box[2]),max(box[1],box[3]),max(box[0],box[2]),max(box[1],box[3])), (vp_ver[0], vp_ver[1], point_top_left[0], point_top_left[1]))
            point_bottom_right = tools_render_CV.line_intersection((max(box[0],box[2]),min(box[1],box[3]),max(box[0],box[2]),max(box[1],box[3])), (vp_hor[0], vp_hor[1], point_bottom_left[0], point_bottom_left[1]))
            point_top_right = tools_render_CV.line_intersection((vp_hor[0], vp_hor[1], point_top_left[0], point_top_left[1]),(vp_ver[0], vp_ver[1], point_bottom_right[0], point_bottom_right[1]))

            points_BEV = cv2.perspectiveTransform(numpy.array([point_bottom_right, point_bottom_left, point_top_left, point_top_right]).astype(numpy.float32).reshape((-1, 1, 2)), h_ipersp).reshape((-1, 2))

            ratio = abs(points_BEV[0][1] - points_BEV[-1][1]) / (abs(points_BEV[0][0] - points_BEV[1][0]) + 1e-4)
            if ratio_best is None or abs(ratio - self.taret_ratio_L_W) < abs(ratio_best - self.taret_ratio_L_W):
                ratio_best = ratio
                footprint, roofprint, metadata, points_BEV_best = self.calc_footprint(box, cam_height, p_camera_BEV_xy, p_center_BEV_xy, point_bottom_left, point_bottom_right, point_top_right, point_top_left, points_BEV)

        cols = ['cuboid%02d' % i for i in range(16)] + ['yaw_cam_car', 'pitch_cam','L','W','H'] + ['p_bev%02d' % i for i in range(8)]
        df = pd.DataFrame(numpy.concatenate((footprint, roofprint, metadata, points_BEV_best), axis=1).reshape((1, -1)),columns=cols)
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def point_to_footprint(self, point, vp_ver, vp_hor, cam_height, p_camera_BEV_xy, p_center_BEV_xy, h_ipersp):

        point_BEV = cv2.perspectiveTransform(numpy.array(point).astype(numpy.float32).reshape((-1, 1, 2)), h_ipersp).reshape(2)
        yaw_res,pitch_cam = self.calc_angles(point, cam_height, p_camera_BEV_xy, p_center_BEV_xy, point_BEV)
        df = pd.DataFrame({'yaw_cam_car': [yaw_res], 'pitch_cam': [pitch_cam],'p_bev0':point_BEV[0],'p_bev1':point_BEV[1]})
        return df
    # ----------------------------------------------------------------------------------------------------------------------
    def evaluate_objects(self, h_ipersp, df_boxes_cars, vp_ver, vp_hor, cam_height, p_camera_BEV_xy, p_center_BEV_xy):

        if df_boxes_cars.shape[0]==0:return pd.DataFrame([])
        df_cuboids_all = pd.DataFrame([])
        for box in df_boxes_cars.iloc[:, -4:].values:
            if vp_ver[0]>min(box[0], box[2]):
                df_cuboids = self.box_to_footprint_look_upright(box, vp_ver, vp_hor, cam_height, p_camera_BEV_xy,p_center_BEV_xy,h_ipersp)
            else:
                df_cuboids = self.box_to_footprint_look_upleft(box, vp_ver, vp_hor, cam_height,p_camera_BEV_xy, p_center_BEV_xy,h_ipersp)
            df_cuboids_all = pd.concat([df_cuboids_all,df_cuboids],axis=0)

        df_boxes_cars.reset_index(drop=True, inplace=True)
        df_cuboids_all.reset_index(drop=True, inplace=True)
        df_res = pd.concat([df_boxes_cars, df_cuboids_all], axis=1)

        return df_res
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_angles(self,h_ipersp, df_boxes_lps, vp_ver, vp_hor, cam_height, p_camera_BEV_xy, p_center_BEV_xy):
        if df_boxes_lps.shape[0]==0:return pd.DataFrame([])
        df_angles_all = pd.DataFrame([])
        for box in df_boxes_lps.iloc[:, -4:].values:
            point = [(box[0]+box[2])/2,(box[1]+box[3])/2]
            df_angles = self.point_to_footprint(point, vp_ver, vp_hor, cam_height, p_camera_BEV_xy, p_center_BEV_xy, h_ipersp)
            df_angles_all = df_angles_all.append(df_angles,ignore_index=True)

        df_boxes_lps.reset_index(drop=True, inplace=True)
        df_angles_all.reset_index(drop=True, inplace=True)
        df_res = pd.concat([df_boxes_lps, df_angles_all], axis=1)

        return df_res
# ----------------------------------------------------------------------------------------------------------------------
    def remove_bg(self,df_boxes,folder_in,list_of_masks='*.jpg',limit=50):

        image_S = numpy.zeros((self.H,self.W,3),dtype=numpy.long)
        image_C = numpy.zeros((self.H,self.W  ),dtype=numpy.long)

        for filename in tools_IO.get_filenames(folder_in, list_of_masks)[:limit]:
            image = cv2.imread(folder_in+filename)
            if df_boxes.shape[0]>0:
                df = tools_DF.apply_filter(df_boxes,df_boxes.columns[0],filename)
            else:
                df=df_boxes
            image_cut, mask = self.cut_boxes_at_original(image, df)
            image_S+=image_cut
            image_C+=mask

        image_S[:, :, 0] = image_S[:, :, 0] / image_C
        image_S[:, :, 1] = image_S[:, :, 1] / image_C
        image_S[:, :, 2] = image_S[:, :, 2] / image_C
        return image_S.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid(self, dim, rvec, tvec):
        dw, dh, dl = dim[0]/2, dim[1]/2, dim[2]/2
        X = [[-dw, -dh, -dl],[+dw, -dh, -dl],[+dw, -dh, +dl],[-dw, -dh, +dl],[-dw, +dh, -dl],[+dw, +dh, -dl],[+dw, +dh, +dl],[-dw, +dh, +dl]]
        X = numpy.array(X)
        X = tools_pr_geom.apply_rotation(rvec, X)
        X = tools_pr_geom.apply_translation(tvec, X)
        return X
# ----------------------------------------------------------------------------------------------------------------------
    def construct_3d_vehicle(self, dim, rvec, tvec):

        import tools_wavefront
        object = tools_wavefront.ObjLoader()
        object.load_mesh(self.filename_vehicle_3d_obj, do_autoscale=False)
        object.rotate_mesh_rvec(rvec)
        object.translate_mesh(tvec)

        return object.coord_vert
# ----------------------------------------------------------------------------------------------------------------------
    def get_vehicle_dims(self,df_objects,r,p_center_BEV_xy,pix_per_meter_BEV):
        dim1 = numpy.linalg.norm(df_objects[['p_bev00', 'p_bev01']].iloc[r].values - df_objects[['p_bev02', 'p_bev03']].iloc[r].values) / pix_per_meter_BEV
        dim2 = numpy.linalg.norm(df_objects[['p_bev02', 'p_bev03']].iloc[r].values - df_objects[['p_bev04', 'p_bev05']].iloc[r].values) / pix_per_meter_BEV
        dimW, dimL = min(dim1, dim2), max(dim1, dim2)
        dimH = self.mean_vehicle_length * self.taret_ratio_H_W / self.taret_ratio_L_W
        dims = (dimW, dimH, dimL)

        rvec_car = (0, 0, numpy.pi * df_objects['yaw_ego_deg'].iloc[r] / 180)
        centroid_x = df_objects[['p_bev00', 'p_bev02', 'p_bev04', 'p_bev06']].iloc[r].mean()
        centroid_y = df_objects[['p_bev01', 'p_bev03', 'p_bev05', 'p_bev07']].iloc[r].mean()
        centroid_x_m = -(p_center_BEV_xy[0] - centroid_x) / pix_per_meter_BEV
        centroid_y_m = (p_center_BEV_xy[1] - centroid_y) / pix_per_meter_BEV
        tvec_car = numpy.array((centroid_x_m, +dimH / 2, centroid_y_m))
        return dims, rvec_car, tvec_car
# ----------------------------------------------------------------------------------------------------------------------
    def get_cuboids(self,df_objects,p_center_BEV_xy,pix_per_meter_BEV,rvec, tvec, camera_matrix_3x3, mat_trns):

        cuboids_GL  = []
        for r in range(df_objects.shape[0]):
            dims, rvec_car, tvec_car = self.get_vehicle_dims(df_objects,r,p_center_BEV_xy,pix_per_meter_BEV)
            points_3d = self.construct_cuboid(dims, rvec_car, tvec_car)
            points_2d = tools_render_GL.project_points_rvec_tvec_GL(points_3d, rvec, tvec, camera_matrix_3x3, mat_trns)
            cuboids_GL.append(points_2d)

        return pd.DataFrame(numpy.array(cuboids_GL).reshape((-1,16)))
# ----------------------------------------------------------------------------------------------------------------------
    def get_3d_vehicles(self, df_objects, p_center_BEV_xy, pix_per_meter_BEV, rvec, tvec, camera_matrix_3x3, mat_trns):

        vehicles_2d = []
        for r in range(df_objects.shape[0]):
            dims, rvec_car, tvec_car = self.get_vehicle_dims(df_objects,r,p_center_BEV_xy,pix_per_meter_BEV)
            tvec_car[1] = 0
            points_3d = self.construct_3d_vehicle(dims, rvec_car, tvec_car)

            points_2d = tools_render_GL.project_points_rvec_tvec_GL(points_3d, rvec,tvec, camera_matrix_3x3,mat_trns)
            vehicles_2d.append(points_2d)

        return vehicles_2d
# ----------------------------------------------------------------------------------------------------------------------
    def draw_3d_vehicles(self,image0, df_objects, p_center_BEV_xy, pix_per_meter_BEV, rvec, tvec, camera_matrix_3x3, mat_trns):

        import tools_GL3D
        tg_half_fovx = camera_matrix_3x3[0, 2] / camera_matrix_3x3[0, 0]
        images = []
        for r in range(df_objects.shape[0]):
            dims, rvec_car, tvec_car = self.get_vehicle_dims(df_objects,r,p_center_BEV_xy,pix_per_meter_BEV)
            tvec_car[1] = 0
            R = tools_GL3D.render_GL3D(self.filename_vehicle_3d_obj,W=self.W//2, H=self.H//2,is_visible=False,do_normalize_model_file=False,projection_type='P',textured = False,rvec=rvec_car,tvec=tvec_car)
            image = R.get_image_perspective(rvec, tvec, tg_half_fovx, tg_half_fovx, do_debug=False, mat_view_to_1=True )
            images.append(image)
            #cv2.imwrite(self.folder_out+'xxx.png',image)

        image = numpy.mean(numpy.array(images),axis=0)
        image = cv2.resize(image.astype(numpy.uint8),(self.W, self.H))
        image = cv2.addWeighted(tools_image.desaturate(image0), 0.2, image, 1-0.2, 0)

        return image

# ----------------------------------------------------------------------------------------------------------------------
    def draw_BEVs_folder(self,df_objects_all,fov_x_deg,fov_y_deg,vp_ver,vp_hor,pix_per_meter_BEV,folder_in,list_of_masks='*.jpg',image_clear_bg=None):

        image_bg = image_clear_bg if image_clear_bg is not None else cv2.imread(folder_in+tools_IO.get_filenames(folder_in, list_of_masks)[0])
        image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.build_BEV_by_fov_van_point(image_bg, fov_x_deg, fov_y_deg,vp_ver,vp_hor, do_rotation=True)
        image_BEV, df_keypoints_pitch, df_vertical = self.draw_grid_at_BEV(image_BEV, p_camera_BEV_xy, p_center_BEV_xy,lines_edges, fov_x_deg, fov_y_deg)
        image_BEV = self.draw_meters_at_BEV(image_BEV, p_camera_BEV_xy, pix_per_meter_BEV, pad=75)

        cam_offset_dist = p_camera_BEV_xy[1]-p_center_BEV_xy[1]
        cam_offset_dist_m = cam_offset_dist/pix_per_meter_BEV
        cam_height_m = cam_height_px/pix_per_meter_BEV
        a_pitch = -numpy.arctan(cam_height_px/cam_offset_dist)

        camera_matrix_3x3, rvec, tvec, RT_GL = tools_render_GL.define_cam_position(self.W, self.H, fov_x_deg,cam_offset_dist_m, cam_height_m,a_pitch)
        mat_trns = numpy.diag([1, -1, 1, 1])
        for filename in tools_IO.get_filenames(folder_in, list_of_masks)[:1]:
            image = cv2.imread(folder_in+filename)

            df_objects = tools_DF.apply_filter(df_objects_all, 'ID', filename)

            df_boxes      = df_objects.iloc[:,3:7]
            df_footprints = df_objects.iloc[:,7:7+ 8]
            cuboids_orig    = df_objects.iloc[:,7:7+16]
            df_metadata = df_objects[['yaw_ego_deg','pitch_cam']]
            image_BEV_local = self.draw_footprints_at_BEV(image_BEV, h_ipersp, df_footprints, df_metadata)

            #cuboids_GL  = self.get_cuboids(df_objects, p_center_BEV_xy, pix_per_meter_BEV, rvec, tvec,camera_matrix_3x3, mat_trns)
            #vehicles_2d = self.get_3d_vehicles(df_objects, p_center_BEV_xy, pix_per_meter_BEV, rvec, tvec,camera_matrix_3x3, mat_trns)
            image_res = self.draw_3d_vehicles(image,df_objects, p_center_BEV_xy, pix_per_meter_BEV, rvec, tvec, camera_matrix_3x3,mat_trns)

            #image_res = self.draw_cuboids_at_original(image_res, cuboids_GL,color=(0,0,255))
            #image_res = self.draw_contour_at_original(image_res,vehicles_2d,color=(0,128,255))
            #image_res = self.draw_cuboids_at_original(image_res, cuboids_orig,color=(255,128,0))
            image_res = self.drawes_boxes_at_original(image_res, df_boxes)
            image_res = self.draw_grid_at_original(image_res, df_keypoints_pitch, df_vertical, h_ipersp)
            image_res = tools_draw_numpy.draw_text(image_res,'cam fov=%.f'%fov_x_deg+u'\u00B0'+'\nheight=%.1f m'%(cam_height_px/pix_per_meter_BEV),(0,self.H-50), color_fg=(255,255,255),clr_bg=None,font_size=40)
            cv2.imwrite(self.folder_out +'%.02d_'%fov_x_deg +filename, tools_image.hstack_images(image_res,image_BEV_local))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_four_point_transform_mat(self, p_src, target_width, target_height):
        def order_points(pts):
            xSorted = pts[numpy.argsort(pts[:, 0]), :]
            leftMost = xSorted[:2, :]
            rightMost = xSorted[2:, :]
            leftMost = leftMost[numpy.argsort(leftMost[:, 1]), :]
            (tl, bl) = leftMost
            D = dist.cdist(tl[numpy.newaxis], rightMost, "euclidean")[0]
            (br, tr) = rightMost[numpy.argsort(D)[::-1], :]
            return numpy.array([tl, tr, br, bl], dtype=numpy.float32)

        p_src_ordered = order_points(p_src)

        # tl, tr, br, bl = p_src_ordered
        # widthA = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        # widthB = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        # maxWidth = max(int(widthA), int(widthB))
        # heightA = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        # heightB = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        # maxHeight = max(int(heightA), int(heightB))

        (maxWidth, maxHeight) = (target_width, target_height)

        p_dst = numpy.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
                            dtype=numpy.float32)
        M = cv2.getPerspectiveTransform(p_src_ordered, p_dst)

        return M

# ---------------------------------------------------------------------------------------------------------------------
    def get_inverce_perspective_mat(self,image,point_vanishing,line_up,line_bottom,target_width,target_height):
        if line_bottom is None:
            line_bottom = numpy.array((0,image.shape[0],image.shape[1],image.shape[0]),dtype=numpy.float32)

        p1a = (line_up[0], line_up[1])
        p2a = (line_up[2], line_up[3])
        if p1a[0]>p2a[0]:p1a,p2a = p2a,p1a

        p3a = (line_bottom[0], line_bottom[1])
        p4a = (line_bottom[2], line_bottom[3])
        if p3a[0]>p4a[0]:p3a,p4a = p4a,p3a

        p1b = tools_render_CV.line_intersection(line_up    ,(point_vanishing[0],point_vanishing[1],p3a[0],p3a[1]))
        if p1a[0]<p1b[0]:p1 = p1a
        else:p1 = p1b

        p2b = tools_render_CV.line_intersection(line_up    ,(point_vanishing[0],point_vanishing[1],p4a[0],p4a[1]))
        if p2a[0]>p2b[0]:p2 = p2a
        else:p2 = p2b


        line1 = (point_vanishing[0],point_vanishing[1],p1[0],p1[1])
        line2 = (point_vanishing[0],point_vanishing[1],p2[0],p2[1])

        p3 = tools_render_CV.line_intersection(line_bottom,line1)
        p4 = tools_render_CV.line_intersection(line_bottom,line2)

        M = self.get_four_point_transform_mat(numpy.array((p1, p2, p3, p4), dtype=numpy.float32), target_width, target_height)

        #res = tools_draw_numpy.draw_points(image,[p1,p2,p3,p4],w=16)

        return M
# ---------------------------------------------------------------------------------------------------------------------
    def get_inverce_perspective_mat_v2(self,image,target_W,target_H,point_van_xy,tol_up = 100,pad_left=0,pad_right=0,do_check=False):
        H, W = image.shape[:2]


        tol_bottom = 0
        upper_line = (0, point_van_xy[1] + tol_up, W, point_van_xy[1] + tol_up)
        bottom_line = (0, H - tol_bottom, W, H - tol_bottom)

        line1 = (   -W*pad_left , H, point_van_xy[0], point_van_xy[1])
        line2 = (W + W*pad_right, H, point_van_xy[0], point_van_xy[1])
        p1 = tools_render_CV.line_intersection(upper_line, line1)
        p2 = tools_render_CV.line_intersection(upper_line, line2)
        p3 = tools_render_CV.line_intersection(bottom_line, line1)
        p4 = tools_render_CV.line_intersection(bottom_line, line2)
        src = numpy.array([(p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]), (p4[0], p4[1])], dtype=numpy.float32)
        dst = numpy.array([(0, 0), (target_W, 0), (0, target_H), (target_W, target_H)], dtype=numpy.float32)
        h_ipersp = cv2.getPerspectiveTransform(src, dst)

        if do_check:
            # dst_check = cv2.perspectiveTransform(src.reshape((-1, 1, 2)), h_ipersp).reshape((-1, 2))
            image = tools_draw_numpy.draw_convex_hull(image, numpy.array([p1, p2, p3, p4]), color=(36, 10, 255),transperency=0.5)


        return h_ipersp
# ----------------------------------------------------------------------------------------------------------------------
    def get_inverce_perspective_mat_v3(self,image,cam_fov_deg,point_van_xy_ver,point_van_xy_hor=None,do_debug=False):
        H, W = image.shape[:2]
        target_H = H

        tol_up = 0.15 * (H - point_van_xy_ver[1])
        if point_van_xy_ver[1] + tol_up<0:
            tol_up=-point_van_xy_ver[1]

        if point_van_xy_hor is None:
            upper_line = (0, point_van_xy_ver[1] + tol_up, W, point_van_xy_ver[1] + tol_up)
            bottom_line = (0, H, W, H)
        else:
            upper_line  = (point_van_xy_hor[0], point_van_xy_hor[1],W/2, point_van_xy_ver[1] + tol_up)
            #bottom_line = (point_van_xy_hor[0], point_van_xy_hor[1],W if point_van_xy_hor[0]<0 else 0, H)
            bottom_line = (0, H, W, H)


        line_van_left = (0, H, point_van_xy_ver[0], point_van_xy_ver[1])
        line_van_right = (W, H, point_van_xy_ver[0], point_van_xy_ver[1])

        p1 = tools_render_CV.line_intersection(upper_line, line_van_left)
        p2 = tools_render_CV.line_intersection(upper_line, line_van_right)
        p3 = tools_render_CV.line_intersection(bottom_line, line_van_left)
        p4 = tools_render_CV.line_intersection(bottom_line, line_van_right)
        p5 = numpy.array((0, p1[1])).astype(numpy.float32)
        p6 = numpy.array((W, p2[1])).astype(numpy.float32)
        p7 = numpy.array((0, H)).astype(numpy.float32)
        p8 = numpy.array((W, H)).astype(numpy.float32)

        src = numpy.array((p1,p2,p3,p4), dtype=numpy.float32).reshape((-1,2))

        if do_debug:
            im = tools_draw_numpy.draw_points(tools_image.desaturate(image), src.astype(numpy.int), color=(0, 0, 200), w=4, put_text=True)
            im = tools_draw_numpy.draw_convex_hull(im, src.astype(numpy.int),color=(0, 0, 200),transperency=0.9)
            im = tools_draw_numpy.draw_lines(im,[upper_line,bottom_line,line_van_left,line_van_right],color=(0, 0, 200))
            cv2.imwrite('./images/output/ppp.png',im)

        min_delta = numpy.inf
        h_ipersp_best = None
        best_target_W = None
        best_target_H = None
        rotation_deg = 0

        for target_W in range(int(target_H*0.1),int(target_H*2.5)):

            dst = numpy.array([(0, 0), (target_W,0),(0, target_H),(target_W,target_H)], dtype=numpy.float32).reshape((-1,2))

            #h_ipersp = cv2.getPerspectiveTransform(src, dst)
            h_ipersp, loss = tools_pr_geom.fit_homography(src, dst)

            # ppp = src.reshape((-1, 1, 2))
            # print(ppp.flatten(), cv2.perspectiveTransform(ppp, h_ipersp).flatten())
            # ppp = numpy.array(((self.W / 2, self.H / 2))).reshape((-1, 1, 2))
            # print(ppp.flatten(), cv2.perspectiveTransform(ppp, h_ipersp).flatten())

            #cv2.imwrite('./images/output/hh_%03d.png'%target_W, cv2.warpPerspective(image, h_ipersp, (target_W, target_H), borderValue=(32, 32, 32)))

            p5_new = cv2.perspectiveTransform(p5.reshape((-1,1,2)),h_ipersp).flatten()
            p6_new = cv2.perspectiveTransform(p6.reshape((-1,1,2)),h_ipersp).flatten()
            p7_new = cv2.perspectiveTransform(p7.reshape((-1,1,2)),h_ipersp).flatten()
            p8_new = cv2.perspectiveTransform(p8.reshape((-1,1,2)),h_ipersp).flatten()

            line_van_left  = (p5_new[0], p5_new[1], p7_new[0], p7_new[1])
            line_van_right = (p6_new[0], p6_new[1], p8_new[0], p8_new[1])

            a = tools_render_CV.angle_between_lines(line_van_left,line_van_right)
            target_W=target_W+1 if a>cam_fov_deg else target_W-1

            #print(target_W,a,abs(a-cam_fov_deg))
            if abs(a-cam_fov_deg)<min_delta:
                min_delta = abs(a - cam_fov_deg)
                best_target_H = target_H
                best_target_W = int(p6_new[0]-p5_new[0])
                new_dst = dst.copy()
                new_dst[:,0]+=abs(p5_new[0])
                #h_ipersp_best = cv2.getPerspectiveTransform(src, new_dst)
                h_ipersp_best, r = tools_pr_geom.fit_homography(src, new_dst)

                central_line = numpy.array(([self.W/2, 0], [self.W/2, self.H])).astype(numpy.float32)
                central_line = cv2.perspectiveTransform(central_line.reshape((-1, 1, 2)), h_ipersp).reshape((-1, 2))
                p_up, p_dn = central_line[0], central_line[1]
                rotation_deg = -math.degrees(numpy.arctan((p_up[0] - p_dn[0]) / (p_up[1] - p_dn[1])))

                # edges = numpy.array(([0, 0], [0, self.H], [self.W, 0], [self.W, self.H])).astype(numpy.float32)
                # edges_BEV = cv2.perspectiveTransform(edges.reshape((-1, 1, 2)), h_ipersp).reshape((-1,2))
                #
                # p_up,p_dn = edges_BEV[0],edges_BEV[1]
                # rotation_deg1 = -math.degrees(numpy.arctan((p_up[0] - p_dn[0]) / (p_up[1] - p_dn[1])))
                # p_up,p_dn = edges_BEV[2],edges_BEV[3]
                # rotation_deg2 = -math.degrees(numpy.arctan((p_up[0] - p_dn[0]) / (p_up[1] - p_dn[1])))
                # rotation_deg = (rotation_deg1+rotation_deg2)/2

                if do_debug:
                    image_BEV = cv2.warpPerspective(image, h_ipersp_best, (best_target_W, best_target_H), borderValue=(32, 32, 32))
                    image_BEV = tools_draw_numpy.draw_lines(image_BEV,[(p_dn[0],p_dn[1],p_up[0],p_up[1])],w=1,color=(128,0,255))
                    cv2.imwrite('./images/output/xx_%03d.png'%target_W, image_BEV)
            else:
                break

        return h_ipersp_best,best_target_W, best_target_H,rotation_deg
# ----------------------------------------------------------------------------------------------------------------------
