import os
import cv2
import numpy
import subprocess
import uuid
# ----------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
# ----------------------------------------------------------------------------------------------------------------
from utils_primities import Lines
import utils_filtering
# ----------------------------------------------------------------------------------------------------------------
class Fornaciari(object):
    def __init__(self,folder_out):
        self.name = "Fornaciari"
        self.folder_out = folder_out
        self.bin_name = './../_weights/FastEllipse.exe'
        self.Filter = utils_filtering.Soccer_Field_Lines_Filter(folder_out,None,None)

        self.th_valid_position_X_min = 0.0/1280
        self.th_valid_position_X_max = 1200.0/1280
        self.th_valid_position_Y_min = 200.0/720
        self.th_valid_position_Y_max = 600.0/720
        self.th_valid_rotation_min = 40
        self.th_valid_rotation_max = 140

        return
# ----------------------------------------------------------------------------------------------------------------
    def draw_segments(self,image, segments,color=(255,255,255),w=4,put_text=False):

        result = image.copy()
        H, W = image.shape[:2]
        for id, segment in enumerate(segments):
            if len(numpy.array(color).shape) == 1:
                clr = color
            else:
                clr = color[id].tolist()

            if numpy.any(numpy.isnan(segment)): continue
            for point in segment: cv2.circle(result, (int(point[0]), int(point[1])), 1, clr, w)
            if put_text:
                x, y = int(segment[0][0] + segment[-1][0]) // 2, int(segment[0][1] + segment[-1][1]) // 2
                cv2.putText(result, '{0}'.format(id), (min(W - 10, max(10, x)), min(H - 5, max(10, y))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)
        return result
# ----------------------------------------------------------------------------------------------------------------
    def extract_segments(self, image,min_size=1,GT_ellps=None,save_stats = False):

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

        if save_stats:
            gt_ellipse_mask = self.get_ellipse_mask(GT_ellps)

        segments = []
        if os.path.isfile(self.folder_out + temp_sgm + '.txt'):
            data = tools_IO.load_mat_var_size(self.folder_out + temp_sgm + '.txt',dtype=numpy.int,delim=' ')
            for segment in data:
                points = numpy.array(segment, dtype=numpy.int).reshape((-1,2))
                if save_stats:
                    is_TP = self.match_segment_ellipse(points, gt_ellipse_mask)
                    (r1, t1x, t1y, sc1) = self.Filter.line_to_RTS(self.Filter.interpolate_points_by_line(points))
                    tools_IO.save_raw_vec([1 * is_TP, r1, t1x, t1y, sc1],self.folder_out + 'elli_good_bad.txt', fmt='%f', delim='\t')

                if not self.is_valid_position(points):continue
                segments.append(points)

        tools_IO.remove_file(self.folder_out + temp_sgm + '.txt')
        tools_IO.remove_file(self.folder_out + temp_dbg + '.png')
        if (type(image) == numpy.ndarray):tools_IO.remove_file(self.folder_out + temp_png + '.png')

        return segments
# ----------------------------------------------------------------------------------------------------------------
    def is_good_ellipse(self,ellipse):

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
    def is_valid_position(self, points):
        #X = numpy.mean(points[:, 0])
        #Y = numpy.mean(points[:, 1])
        #X/= self.W
        #Y/= self.H
        #if not (self.th_valid_position_X_min <= X and X <= self.th_valid_position_X_max): return False
        #if not (self.th_valid_position_Y_min <= Y and Y <= self.th_valid_position_Y_max): return False

        #line = self.Filter.interpolate_points_by_line(points)
        #(rotation, tx, ty, scale) = self.Filter.line_to_RTS(line)
        #if not (self.th_valid_rotation_min <= rotation and rotation <= self.th_valid_rotation_max): return False

        return True
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
    def estimate_ellipse(self,segments,s1,s2):

        seg = numpy.vstack((segments[s1], segments[s2]))
        ellipse = cv2.fitEllipseAMS(seg)
        if not self.is_good_ellipse(ellipse): return None,[]

        ellipse_mask = self.get_ellipse_mask(ellipse)
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
    def segments_to_ellipse(self, image, segments, do_debug=False):

        self.H,self.W = image.shape[:2]
        weights = numpy.array([len(s) for s in segments],dtype=numpy.float)
        weights=weights/weights.sum()

        Q,E,Cands = {},{},{}
        for s1 in range(len(segments) - 1):
            for s2 in range(s1 + 1, len(segments)):
                ellipse, idx_match = self.estimate_ellipse(segments,s1,s2)
                if ellipse is None:continue

                E[(s1, s2)] = ellipse
                Q[(s1, s2)] = weights[idx_match].sum()
                Cands[(s1, s2)] = idx_match

                if do_debug:
                    image_debug  = tools_image.desaturate(image)
                    image_debug = self.draw_segments(image_debug, [segments[s1]], color=(90,0,255), w=4)
                    image_debug = self.draw_segments(image_debug, [segments[s2]], color=(0,90,255), w=4)

                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
                    rotation_angle = ellipse[2]
                    cv2.ellipse(image_debug, center, axes, rotation_angle, startAngle=0, endAngle=360, color=(0, 0, 190), thickness=1)
                    cv2.imwrite(self.folder_out+'ellips_%03d_%03d.png'%(s1,s2),image_debug)

        ellipse, cands = None, None
        if len(Q)>0:
            key = tools_IO.max_element_by_value(Q)[0]
            cands = Cands[key]
            #ellipse = E[key]
            ellipse = self.refine_ellipse(segments,cands)

        return ellipse, cands
# ----------------------------------------------------------------------------------------------------------------