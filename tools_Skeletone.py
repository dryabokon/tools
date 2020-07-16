import cv2
import numpy
from skimage.morphology import skeletonize
from sklearn.linear_model import LinearRegression
# ----------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_render_CV
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------
class Skelenonizer(object):
    def __init__(self,folder_out):
        self.name = "Skelenonizer"
        self.folder_out = folder_out
        self.nodes = None
        self.W,self.H = None,None
        return
# ----------------------------------------------------------------------------------------------------------------
    def binarize(self,image):
        if len(image.shape)==3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 0)
        binarized[gray>230]=255
        binarized[gray<25 ]=0

        return binarized
# ----------------------------------------------------------------------------------------------------------------
    def morph(self,image,kernel_h=3,kernel_w=3,n_dilate=1,n_erode=1):
        kernel = numpy.ones((kernel_h, kernel_w), numpy.uint8)
        result = cv2.dilate(image, kernel, iterations=n_dilate)
        result = cv2.erode(result, kernel, iterations=n_erode)
        return result
# ----------------------------------------------------------------------------------------------------------------
    def binarized_to_skeleton(self, binarized):
        return 255*skeletonize(binarized > 0).astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------
    def remove_joints(self,skeleton):
        mask = tools_image.sliding_2d(skeleton,-1,1,-1,1,'sum')
        mask = (mask == (255 * 3)).astype(numpy.uint8)
        mask = self.morph(mask, 2, 2, 1, 0)
        result  = numpy.multiply(skeleton,1-mask).astype(numpy.uint8)
        return result
# ----------------------------------------------------------------------------------------------------------------
    def skeleton_to_segments(self,skeleton,min_len=10):

        image_cleaned = self.remove_joints(skeleton)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(image_cleaned)

        i=0
        dct_seg={}
        for l in range(1,ret):
            top,left, h, w, N = stats[l]
            if N>min_len and (h**2 + w**2)>min_len**2:
                dct_seg[l]=i
                i+=1

        segments = [[] for c in range(i)]
        for r in range(labels.shape[0]):
            for c in range(labels.shape[1]):
                label = labels[r,c]
                if label in dct_seg:
                    number = dct_seg[label]
                    segments[number].append((c,r))

        segments = [numpy.array(s) for s in segments]
        return segments
# ----------------------------------------------------------------------------------------------------------------
    def extract_segments(self,image,min_len=10):

        image_edges = cv2.Canny(image,20,80)
        segments = self.skeleton_to_segments(image_edges,min_len)

        return segments
# ----------------------------------------------------------------------------------------------------------------
    def get_length_segment(self,segment):
        xmin = numpy.min(segment[:,0])
        xmax = numpy.max(segment[:,0])
        ymin = numpy.min(segment[:,1])
        ymax = numpy.max(segment[:,1])
        return numpy.linalg.norm((xmax-xmin,ymax-ymin))
# ----------------------------------------------------------------------------------------------------------------
    def sraighten_segments(self, segments,min_len=10):
        result = []
        for i, s in enumerate(segments):
            straighten  = self.sraighten_segment(s,min_len)
            for each in straighten:result.append(each)
        return result
# ----------------------------------------------------------------------------------------------------------------
    def sraighten_segment(self,segment,min_len=10,do_debug=False):

        if self.get_length_segment(segment)<min_len:
            return []

        idx_DL = numpy.lexsort((-segment[:, 0], segment[:, 1]))  # traverce from top
        idx_DR = numpy.lexsort(( segment[:, 0], segment[:, 1]))  # traverce from top
        idx_RD = numpy.lexsort(( segment[:, 1], segment[:, 0]))  # traverce from left
        idx_RU = numpy.lexsort((-segment[:, 1], segment[:, 0]))  # traverce from left

        segment_DL = segment[idx_DL]
        segment_DR = segment[idx_DR]
        segment_RD = segment[idx_RD]
        segment_RU = segment[idx_RU]

        sorted_x_DL = segment[idx_DL, 0]  # sould decrease
        sorted_x_DR = segment[idx_DR, 0]  # sould increase
        sorted_y_RD = segment[idx_RD, 1]  # sould increase
        sorted_y_RU = segment[idx_RU, 1]  # sould decrease

        dx_DL = (-sorted_x_DL + numpy.roll(sorted_x_DL, -1))[:-1]  # should be [-1..0]
        dx_DR = (-sorted_x_DR + numpy.roll(sorted_x_DR, -1))[:-1]  # should be [0..+1]
        dy_RD = (-sorted_y_RD + numpy.roll(sorted_y_RD, -1))[:-1]  # should be [0..+1]
        dy_RU = (-sorted_y_RU + numpy.roll(sorted_y_RU, -1))[:-1]  # should be [-1..0]

        th = 1

        dx_DL_good = numpy.array(-th<=dx_DL) & numpy.array(dx_DL<=0)
        dx_DR_good = numpy.array(0  <=dx_DR) & numpy.array(dx_DR<=th)
        dy_RD_good = numpy.array(0  <=dy_RD) & numpy.array(dy_RD<=th)
        dy_RU_good = numpy.array(-th<=dy_RU) & numpy.array(dy_RU<=0)

        if numpy.all(dx_DL_good) or numpy.all(dx_DR_good) or numpy.all(dy_RD_good) or numpy.all(dy_RU_good):
            return [segment]

        #search best cut
        pos_DL, L_DL = tools_IO.get_longest_run_position_len(dx_DL_good)
        pos_DR, L_DR = tools_IO.get_longest_run_position_len(dx_DR_good)
        pos_RD, L_RD = tools_IO.get_longest_run_position_len(dy_RD_good)
        pos_RU, L_RU = tools_IO.get_longest_run_position_len(dy_RU_good)

        best_s = int(numpy.argmax([L_DL,L_DR,L_RD,L_RU]))
        pos_best = [pos_DL,pos_DR,pos_RD,pos_RU][best_s]
        len_best = [L_DL, L_DR, L_RD, L_RU][best_s]
        segment_best = [segment_DL, segment_DR, segment_RD, segment_RU][best_s]

        if do_debug:
            cv2.imwrite(self.folder_out + 'DL_bin.png', 255 * dx_DL_good.reshape((1, -1)))
            cv2.imwrite(self.folder_out + 'DR_bin.png', 255 * dx_DR_good.reshape((1, -1)))
            cv2.imwrite(self.folder_out + 'RD_bin.png', 255 * dy_RD_good.reshape((1, -1)))
            cv2.imwrite(self.folder_out + 'RU_bin.png', 255 * dy_RU_good.reshape((1, -1)))

            image = numpy.full((720,1280,3),64,dtype=numpy.uint8)
            cv2.imwrite(self.folder_out + 'DL.png',tools_draw_numpy.draw_points(image, segment_DL, tools_draw_numpy.get_colors(len(segment_DL)),w=1))
            cv2.imwrite(self.folder_out + 'DR.png',tools_draw_numpy.draw_points(image, segment_DR, tools_draw_numpy.get_colors(len(segment_DR)),w=1))
            cv2.imwrite(self.folder_out + 'RD.png',tools_draw_numpy.draw_points(image, segment_RD, tools_draw_numpy.get_colors(len(segment_RD)),w=1))
            cv2.imwrite(self.folder_out + 'RU.png',tools_draw_numpy.draw_points(image, segment_RU, tools_draw_numpy.get_colors(len(segment_RU)),w=1))
            #print('len_in=%d pos_best=%d, len_best=%d'%(len(segment), pos_best,len_best))

        s1,s2,s3 = [],[],[]

        if len_best>min_len:
            s1 = [segment_best[pos_best:pos_best + len_best]]

        if pos_best>0:
            s2 = self.sraighten_segment(segment_best[:pos_best],min_len)

        if (pos_best>=0) and (pos_best + len_best<len(segment_best)):
            s3 = self.sraighten_segment(segment_best[pos_best + len_best:],min_len)
        return s1 + s2 + s3
# ----------------------------------------------------------------------------------------------------------------
    def filter_short_segments(self,segments,min_len):
        result = []
        for s in segments:
            if len(s)>min_len:
                result.append(s)

        return result
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
    def keep_double_segments(self, segments, line_upper_bound, base_name=None, do_debug=False):

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
                if l1==42 and l2==50:
                    yy=0
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
            empty = numpy.full((self.H,self.W,3),32,dtype=numpy.uint8)
            colors_all = tools_draw_numpy.get_colors(len(segments), shuffle=True)
            image_segm = tools_draw_numpy.draw_segments(empty, segments, colors_all,w=1,put_text=True)
            cv2.imwrite(self.folder_out + base_name+'_1_segm.png', image_segm)

            #for i,segment in enumerate(segments):
            #    image_segm = tools_draw_numpy.draw_segments(empty, [segment], colors_all[i].tolist(), w=1)
            #    cv2.imwrite(self.folder_out + base_name + '_segm_%03d.png'%i, image_segm)

            image_segm = numpy.full((self.H, self.W, 3), 32, dtype=numpy.uint8)
            image_segm = tools_draw_numpy.draw_segments(image_segm, segments,(90, 90, 90), w=1)
            segments[~idx_good] = None
            image_segm = tools_draw_numpy.draw_segments(image_segm, segments, (0, 0, 255), w=1)
            #image_segm = utils_draw.draw_lines(image_segm, [line_upper_bound], (0, 0, 255), w=1)
            cv2.imwrite(self.folder_out + base_name+'_2_fltr.png', image_segm)

        return segments[idx_good]#, lines[idx_good]
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
