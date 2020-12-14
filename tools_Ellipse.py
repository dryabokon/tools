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
import tools_Skeletone
# ----------------------------------------------------------------------------------------------------------------
class Ellipse_Processor(object):
    def __init__(self,folder_out=None):
        self.folder_out = folder_out

        if folder_out is not None:
            self.folder_cache = self.folder_out + 'cache/'
        else:
            self.folder_cache = None

        self.name = "Ellipse processor"

        self.Ske = tools_Skeletone.Skelenonizer(folder_out)
        return
# ----------------------------------------------------------------------------------------------------------------
    def draw_segments(self,image, segments, color=(255, 255, 255), w=4, put_text=False):

        result = image.copy()
        H, W = image.shape[:2]
        if len(segments) == 0: return result

        for id, segment in enumerate(segments):
            if len(numpy.array(color).shape) == 1:
                clr = color
            else:
                clr = color[id].tolist()

            if segment is None: continue
            if numpy.any(numpy.isnan(segment)): continue

            for point in segment:
                cv2.circle(result, (int(point[0]), int(point[1])), 1, clr, w)
                #if 0 <= point[1] < H and 0 <= point[0] < W:
                #    result[int(point[1]), int(point[0])] = clr

            if put_text:
                x, y = int(segment[:, 0].mean() + -30 + 60 * numpy.random.rand()), int(
                    segment[:, 1].mean() - 30 + 60 * numpy.random.rand())
                cv2.putText(result, '{0}'.format(id), (min(W - 10, max(10, x)), min(H - 5, max(10, y))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)
        return result

# ----------------------------------------------------------------------------------------------------------------------
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
    def get_ellipse_lines(self,ellipse):
        if ellipse is None:
            return None,None
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        rotation_angle = ellipse[2] * numpy.pi / 180

        ct = numpy.cos(rotation_angle)
        st = numpy.sin(rotation_angle)

        p1 = (int(center[0] - axes[0] * ct), int(center[1] - axes[0] * st))
        p2 = (int(center[0] + axes[0] * ct), int(center[1] + axes[0] * st))

        p3 = (int(center[0] - axes[1] * st), int(center[1] + axes[1] * ct))
        p4 = (int(center[0] + axes[1] * st), int(center[1] - axes[1] * ct))

        line1 = (p1[0], p1[1],p2[0], p2[1])
        line2 = (p3[0], p3[1],p4[0], p4[1])


        return line1,line2
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
        if x2 - x1==0:
            angle = 0
        else:
            angle = 90 + math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
        return angle
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
    def preprocess_for_ellipse(self, segments, line_upper_bound=None, line_midfield=None):
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

            if line_midfield is not None:
                if tools_render_CV.distance_point_to_line(line_midfield,segment[0])<10 and tools_render_CV.distance_point_to_line(line_midfield, segment[-1]) < 10: continue

            filtered.append(segment)

        return filtered
# ----------------------------------------------------------------------------------------------------------------
    def segments_to_ellipse(self, image, line_upper_bound, line_midfield, base_name=None, do_debug=False):

        X, success = tools_IO.load_if_exists(self.folder_cache,base_name , '_ellipse.dat')
        if (not do_debug) and success:
            return X


        segments_all = self.Ske.extract_segments(image, 30)

        segments_double = self.Ske.keep_double_segments(segments_all, line_upper_bound, base_name)
        if line_upper_bound is not None:
            segments = self.preprocess_for_ellipse(segments_all, line_upper_bound, line_midfield)

        else:
            segments = segments_double.copy()


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

                if ellipse is not None:
                    for i1 in idx_match:
                        for i2 in idx_match:
                            processed[i1,i2]=1

                    E[(s1, s2)] = ellipse
                    Q[(s1, s2)] = weights[idx_match].sum()
                    Cands[(s1, s2)] = idx_match

                if do_debug and self.folder_out is not None:
                    image_debug  = tools_image.desaturate(image)
                    image_debug = self.draw_segments(image_debug, segments, color=(0, 0, 128), w=8)
                    image_debug = self.draw_segments(image_debug, [segments[s1]], color=(90,0,255), w=8)
                    image_debug = self.draw_segments(image_debug, [segments[s2]], color=(0,90,255), w=8)

                    if ellipse is not None:
                        center = (int(ellipse[0][0]), int(ellipse[0][1]))
                        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
                        rotation_angle = ellipse[2]
                        cv2.ellipse(image_debug, center, axes, rotation_angle, startAngle=0, endAngle=360, color=(0, 0, 190), thickness=1)
                    cv2.imwrite(self.folder_out+base_name+'ellips_%03d_%03d.png'%(s1,s2),image_debug)

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

        tools_IO.write_cache(self.folder_cache , base_name , '_ellipse.dat',ellipse)

        return ellipse
# ----------------------------------------------------------------------------------------------------------------
