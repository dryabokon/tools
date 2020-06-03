import math
import os
import cv2
import numpy
import subprocess
import uuid
from scipy.special import ellipeinc
from scipy import optimize
from sklearn.linear_model import LinearRegression
# ----------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------
import utils_draw
# ----------------------------------------------------------------------------------------------------------------
class Fornaciari(object):
    def __init__(self,folder_out):
        self.name = "Fornaciari"
        self.folder_out = folder_out
        self.bin_name = './../_weights/FastEllipse.exe'

        return
# ----------------------------------------------------------------------------------------------------------------
    def iou(self,boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
# ----------------------------------------------------------------------------------------------------------------------
    def generate_ellipse_points(self,center_xy=(20, 10), Rxy=(5.0, 10.0), rotation_angle_deg=0, N=10):
        def generate_ellipse_angles(num, a, b):

            angles = 2 * numpy.pi * numpy.arange(num) / num
            if a < b:
                e = float((1.0 - a ** 2 / b ** 2) ** 0.5)
                tot_size = ellipeinc(2.0 * numpy.pi, e)
                arc_size = tot_size / num
                arcs = numpy.arange(num) * arc_size
                res = optimize.root(lambda x: (ellipeinc(x, e) - arcs), angles)
                angles = res.x
            elif b < a:
                e = float((1.0 - b ** 2 / a ** 2) ** 0.5)
                tot_size = ellipeinc(2.0 * numpy.pi, e)
                arc_size = tot_size / num
                arcs = numpy.arange(num) * arc_size
                res = optimize.root(lambda x: (ellipeinc(x, e) - arcs), angles)
                angles = numpy.pi / 2 + res.x
            else:
                numpy.arange(0, 2 * numpy.pi, 2 * numpy.pi / num)
            return angles

        rotation_angle = rotation_angle_deg * numpy.pi / 180

        phi = generate_ellipse_angles(N, Rxy[0], Rxy[1])
        points = numpy.vstack((Rxy[0] * numpy.sin(phi), Rxy[1] * numpy.cos(phi))).T
        M = numpy.array([[numpy.cos(rotation_angle), -numpy.sin(rotation_angle), 0],
                         [numpy.sin(rotation_angle), numpy.cos(rotation_angle), 0], [0, 0, 1]])
        points2 = cv2.transform(points.reshape(-1, 1, 2), M).reshape(-1, 3)[:, :2]
        points2[:, 0] += center_xy[0]
        points2[:, 1] += center_xy[1]

        return points2.astype(numpy.int)
# ----------------------------------------------------------------------------------------------------------------
    def get_bbox_ellipse(self,ellipse):
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        Rxy = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        rotation_angle = ellipse[2] * numpy.pi / 180
        M = numpy.array([[numpy.cos(rotation_angle), -numpy.sin(rotation_angle), 0],
                         [numpy.sin(rotation_angle), numpy.cos(rotation_angle), 0], [0, 0, 1]])

        points = numpy.array([[-Rxy[0],-Rxy[1]],[+Rxy[0],+Rxy[1]]],dtype=numpy.float32)

        points2 = cv2.transform(points.reshape((-1, 1, 2)), M).reshape(-1, 3)[:, :2]
        points2[:, 0] += center[0]
        points2[:, 1] += center[1]

        points2 = points2.flatten()

        left = min(points2[0], points2[2])
        right = max(points2[0], points2[2])
        top = min(points2[1], points2[3])
        bottom = max(points2[1], points2[3])
        bbox = numpy.array([left, top, right, bottom])

        return bbox
# ----------------------------------------------------------------------------------------------------------------
    def extract_segments(self, image, min_len=1,line_upper_bound=None):

        temp_sgm = str(uuid.uuid4())
        temp_dbg = str(uuid.uuid4())

        if (type(image) == numpy.ndarray):
            self.H, self.W = image.shape[:2]
            temp_png = str(uuid.uuid4())
            cv2.imwrite(self.folder_out + temp_png + '.png',image)
            command = [self.bin_name, self.folder_out + temp_png + '.png', self.folder_out + temp_sgm + '.txt',self.folder_out + temp_dbg + '.png']
        else:
            temp_png = image
            command = [self.bin_name, image, self.folder_out + temp_sgm + '.txt',self.folder_out + temp_dbg + '.png']

        subprocess.call(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        image_segm = cv2.imread(self.folder_out + temp_dbg + '.png')

        segments = []
        if os.path.isfile(self.folder_out + temp_sgm + '.txt'):
            data = tools_IO.load_mat_var_size(self.folder_out + temp_sgm + '.txt',dtype=numpy.int,delim=' ')
            for segment in data:
                if len(segment)<min_len:continue
                if line_upper_bound is not None and tools_render_CV.is_point_above_line(segment[0], line_upper_bound):continue
                points = numpy.array(segment, dtype=numpy.int).reshape((-1,2))
                segments.append(points)

        tools_IO.remove_file(self.folder_out + temp_sgm + '.txt')
        tools_IO.remove_file(self.folder_out + temp_dbg + '.png')
        if (type(image) == numpy.ndarray):tools_IO.remove_file(self.folder_out + temp_png + '.png')

        segments, index_reg = self.regularize(segments, min_len=20)

        return segments,index_reg,image_segm
# ----------------------------------------------------------------------------------------------------------------
    def is_good_ellipse(self,ellipse):

        if ellipse is None: return False

        if numpy.any(numpy.isnan(ellipse[0])): return False
        if numpy.any(numpy.isnan(ellipse[1])): return False

        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        rotation_angle = ellipse[2]

        if axes[0] < 20 or axes[0] > 150: return False
        if axes[1] < 50 or axes[1] > 600: return False
        if axes[1] / axes[0] < 4.00: return False
        if axes[1] / axes[0] > 6.00: return False
        if not ((90-4<= rotation_angle <=90+4) or (270-4<= rotation_angle <=270+4)): return False

        return True
# ----------------------------------------------------------------------------------------------------------------
    def get_angle(self, line):
        x1, y1, x2, y2 = line
        return 90 + math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
# ----------------------------------------------------------------------------------------------------------------
    def match_segment_ellipse(self, points, ellipse_mask, N=4):

        idx = [-1] + numpy.arange(0, len(points), len(points) // N).tolist()
        pp = numpy.array(points[idx]).astype(int)
        res = ellipse_mask[pp[:, 1], pp[:, 0]]
        res = numpy.all(res > 0)

        return res
# ----------------------------------------------------------------------------------------------------------------
    def refine_ellipse(self,segments,idx):
        if len(idx)==0:return None
        X,Y=[],[]
        for i in idx:
            X+=segments[i][:,0].tolist()
            Y+=segments[i][:,1].tolist()

        seg = numpy.vstack((X, Y),).T
        ellipse = cv2.fitEllipseAMS(seg.astype(numpy.float32))
        return ellipse
# ----------------------------------------------------------------------------------------------------------------
    def estimate_ellipse(self,segments,s1,s2=None):
        if s2 is not None:
            seg = numpy.vstack((segments[s1], segments[s2]))
        else:
            seg = segments[s1]

        ellipse = cv2.fitEllipseAMS(seg)
        if not self.is_good_ellipse(ellipse): return None,[]
        ellipse_mask = self.get_ellipse_mask(ellipse)

        #match1 = self.match_segment_ellipse(segments[s1], ellipse_mask)
        #match2 = self.match_segment_ellipse(segments[s2], ellipse_mask)
        #if not (match1 and match2): return None,[]

        is_match = numpy.array([self.match_segment_ellipse(segment, ellipse_mask) for segment in segments])
        idx_match = numpy.where(is_match)[0]
        return ellipse, idx_match
# ----------------------------------------------------------------------------------------------------------------
    def get_ellipse_mask(self, ellipse,tol=10):
        ellipse_mask = numpy.zeros((self.H, self.W), dtype=numpy.uint8)
        if ellipse is not None:
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
            rotation_angle = ellipse[2]
            cv2.ellipse(ellipse_mask, center, axes, rotation_angle, startAngle=0, endAngle=360, color=255, thickness=tol)
        return ellipse_mask
# ----------------------------------------------------------------------------------------------------------------
    def filter_segments(self, segments, line_upper_bound, line_midfield):
        filtered = []
        if line_upper_bound is not None:
            th_row = 50 + (line_upper_bound[1] + line_upper_bound[3]) / 2
            angle = self.get_angle(line_upper_bound)
        else:
            th_row = None
            angle = None

        for segment in segments:
            box = cv2.boundingRect(segment)

            # bottom check
            if box[1] + box[3] > self.H - 50: continue

            # top check
            if th_row is not None and box[1] < th_row: continue

            # left-right check
            if line_midfield is not None:
                if box[0] > line_midfield[0] + self.W / 4: continue
                if box[0] + box[2] < line_midfield[0] - self.W / 4: continue
            else:
                if angle is not None and angle < 90 and box[0] > 1 * self.W / 3: continue
                if angle is not None and angle > 90 and box[0] < 2 * self.W / 3: continue

            filtered.append(segment)

        return filtered
# ----------------------------------------------------------------------------------------------------------------
    def segments_to_ellipse(self, image, segments, base_name=None, do_debug=False):

        X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + 'ellipse.dat')
        if (not do_debug) and success:
            return X


        self.H,self.W = image.shape[:2]
        weights = numpy.array([len(s) for s in segments],dtype=numpy.float)
        weights=weights/weights.sum()

        processed = numpy.zeros((len(segments),len(segments)))

        Q,E,Cands = {},{},{}
        for s1 in range(len(segments) - 1):
            bbox1 = numpy.array(cv2.boundingRect(segments[s1]))
            bbox1[2:] += bbox1[:2]
            for s2 in range(s1 + 1, len(segments)):
                if processed[s1,s2]==1:continue
                processed[s1, s2] = 1

                bbox2 = numpy.array(cv2.boundingRect(segments[s2]))
                bbox2[2:]+=bbox2[:2]
                if self.iou(bbox1,bbox2)>0.2:continue

                ellipse, idx_match = self.estimate_ellipse(segments,s1,s2)
                if ellipse is None:continue

                for i1 in idx_match:
                    for i2 in idx_match:
                        processed[i1,i2]=1

                E[(s1, s2)] = ellipse
                Q[(s1, s2)] = weights[idx_match].sum()
                Cands[(s1, s2)] = idx_match

                if do_debug:
                    image_debug  = tools_image.desaturate(image)
                    image_debug = utils_draw.draw_segments(image_debug, [segments[s1]], color=(90,0,255), w=4)
                    image_debug = utils_draw.draw_segments(image_debug, [segments[s2]], color=(0,90,255), w=4)

                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
                    rotation_angle = ellipse[2]
                    cv2.ellipse(image_debug, center, axes, rotation_angle, startAngle=0, endAngle=360, color=(0, 0, 190), thickness=1)
                    cv2.imwrite(self.folder_out+'ellips_%03d_%03d.png'%(s1,s2),image_debug)

        ellipse,quality = None,0
        if len(Q)>0:
            key = tools_IO.max_element_by_value(Q)[0]
            cands = Cands[key]
            ellipse = self.refine_ellipse(segments,cands)
            if self.is_good_ellipse(ellipse):
                quality = 100 * Q[key]
                if quality<0.5:
                    ellipse = None
            else:
                ellipse = None

        tools_IO.write_cache(self.folder_out + 'cache/' + base_name + 'ellipse.dat',ellipse)

        return ellipse
# ----------------------------------------------------------------------------------------------------------------
    def are_same_segments(self,segment1,segment2):
        result = False

        match1 = segment2 == segment1[ 0]
        match2 = segment2 == segment1[-1]
        match3 = segment1 == segment2[0]
        match4 = segment1 == segment2[-1]

        match1 = numpy.min(1*match1,axis=1)
        match2 = numpy.min(1*match2,axis=1)
        match3 = numpy.min(1*match3,axis=1)
        match4 = numpy.min(1*match4,axis=1)

        if numpy.sum(match1)+numpy.sum(match2)+numpy.sum(match3)+numpy.sum(match4)>0:
            result = True


        return result
# ----------------------------------------------------------------------------------------------------------------
    def regularize(self, segments,min_len=10):

        th=4
        segments_reg = []
        idx_original = []
        for i,segment in enumerate(segments):
            if i==65:
                i=i
            dx = (segment[:,0]-numpy.roll(segment[:,0],-1))[:-1]
            dy = (segment[:,1]-numpy.roll(segment[:,1],-1))[:-1]
            idx = numpy.where(numpy.abs(dx)>th)[0]
            idy = numpy.where(numpy.abs(dy)>th)[0]
            if len(idx)>0 and len(idy)>0:
                parts = numpy.intersect1d(idx,idy)
                if len(parts)>0:
                    cur = 0
                    for p in parts:
                        segments_reg.append(segment[cur:p+1])
                        idx_original.append(i)
                        cur = p+1
                    segments_reg.append(segment[cur:])
                    idx_original.append(i)
                else:
                    segments_reg.append(segment)
                    idx_original.append(i)
            else:
                segments_reg.append(segment)
                idx_original.append(i)

        res = numpy.array([(r,i) for r,i in zip(segments_reg,idx_original) if len(r)>min_len])
        segments_reg = res[:,0]
        index_reg = res[:,1]

        return segments_reg, index_reg
# ----------------------------------------------------------------------------------------------------------------
    def has_common_range(self,segment1,segment2,line1,line2):
        min_x1 = numpy.min(segment1[:, 0])
        min_y1 = numpy.min(segment1[:, 1])
        min_x2 = numpy.min(segment2[:, 0])
        min_y2 = numpy.min(segment2[:, 1])

        max_x1 = numpy.max(segment1[:, 0])
        max_y1 = numpy.max(segment1[:, 1])
        max_x2 = numpy.max(segment2[:, 0])
        max_y2 = numpy.max(segment2[:, 1])

        check_x = (max_x2 <= min_x1 and max_x2 <= max_x1) or (min_x2 >= min_x1 and min_x2 >= max_x1)
        check_y = (max_y2 <= min_y1 and max_y2 <= max_y1) or (min_y2 >= min_y1 and min_y2 >= max_y1)
        if (check_x) and (check_y): return False

        return True
# ----------------------------------------------------------------------------------------------------------------
    def interpolate_segment_by_line(self, XY):
        reg = LinearRegression()
        X = numpy.array([XY[:, 0]]).astype(numpy.float).T
        Y = numpy.array([XY[:, 1]]).astype(numpy.float).T

        if (X.max() - X.min()) > (Y.max() - Y.min()):
            reg.fit(X, Y)
            X_inter = numpy.array([(X.min(), X.max())]).T
            Y_inter = reg.predict(X_inter)
            line = numpy.array([X_inter[0], Y_inter[0], X_inter[1], Y_inter[1]]).flatten()
        else:
            reg.fit(Y, X)
            Y_inter = numpy.array([(Y.min(), Y.max())]).T
            X_inter = reg.predict(Y_inter)
            line = numpy.array([X_inter[0], Y_inter[0], X_inter[1], Y_inter[1]]).flatten()

        return line
# ----------------------------------------------------------------------------------------------------------------------
    def keep_double_segments(self, segments, index_reg, line_upper_bound, base_name=None, do_debug=False):

        tol_max = 12
        tol_min = 2

        lines=[]
        for i, segment in enumerate(segments):
            if line_upper_bound is None or tools_render_CV.is_point_above_line(segment[0], line_upper_bound):
                lines.append(self.interpolate_segment_by_line((segment)))
            else:
                lines.append((numpy.nan,numpy.nan,numpy.nan,numpy.nan))

        lines = numpy.array(lines)

        has_pair = numpy.zeros(len(lines))

        for l1 in range(len(lines) - 1):
            line1 = lines[l1]
            segment1 = segments[l1]
            if numpy.any(numpy.isnan(line1)):continue
            for l2 in range(l1 + 1, len(lines)):
                if index_reg[l1]==index_reg[l2]:continue
                line2 = lines[l2]
                segment2 = segments[l2]
                if numpy.any(numpy.isnan(line2)): continue
                if self.are_same_segments(segment1,segment2):continue
                if not self.has_common_range(segment1,segment2,line1,line2):continue

                d1 = tools_render_CV.distance_point_to_line(line1, line2[:2])
                d2 = tools_render_CV.distance_point_to_line(line1, line2[2:])
                d3 = tools_render_CV.distance_point_to_line(line2, line1[:2])
                d4 = tools_render_CV.distance_point_to_line(line2, line1[2:])
                if numpy.any(numpy.isnan((d1,d2,d3,d4))):continue
                if (d1>tol_max or d2>tol_max):continue
                if (d3>tol_max or d4>tol_max):continue
                if d1 < tol_min and d2 < tol_min and d3 < tol_min and d4 < tol_min: continue

                len2 = numpy.linalg.norm(line2[:2] - line2[2:])
                len1 = numpy.linalg.norm(line1[:2] - line1[2:])
                d12 = max(numpy.linalg.norm(line1[:2] - line2[2:]), numpy.linalg.norm(line1[2:] - line2[2:]))
                d21 = max(numpy.linalg.norm(line1[:2] - line2[2:]), numpy.linalg.norm(line1[2:] - line2[2:]))

                if (d12>len2+5) and (d21>len1+5):continue

                p1,p2,d = tools_render_CV.distance_between_lines(line1,line2,clampAll=True)
                if d>tol_max:continue

                has_pair[l1]=1
                has_pair[l2]=1

        idx_good = (has_pair>0)
        segments = numpy.array(segments)

        if do_debug:
            #colors_all = tools_IO.get_colors(len(segments), shuffle=True)
            #image_segm = self.Fornaciari.draw_segments(32+0*image_edges, segments, colors_all,w=1,put_text=True)
            #cv2.imwrite(self.folder_out + base_name+'_1_segm.png', image_segm)

            #for i,segment in enumerate(segments):
            #    image_segm = self.Fornaciari.draw_segments(32 + 0 * image_edges, [segment], colors_all[i].tolist(), w=1)
            #    cv2.imwrite(self.folder_out + base_name + '_segm_%03d.png'%i, image_segm)

            image_segm = numpy.full((self.H, self.W, 3), 32, dtype=numpy.uint8)
            image_segm = utils_draw.draw_segments(image_segm, segments,(90, 90, 90), w=1)
            segments[~idx_good] = None
            image_segm = utils_draw.draw_segments(image_segm, segments, (0, 0, 255), w=1)
            image_segm = utils_draw.draw_lines(image_segm, [line_upper_bound], (0, 0, 255), w=1)
            cv2.imwrite(self.folder_out + base_name+'_2_fltr.png', image_segm)

        return segments[idx_good], lines[idx_good]
# ----------------------------------------------------------------------------------------------------------------