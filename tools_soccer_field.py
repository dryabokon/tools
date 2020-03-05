import math
import cv2
import numpy
from skimage.morphology import skeletonize,remove_small_holes, remove_small_objects
import sknw
from sklearn.cluster import KMeans
# ---------------------------------------------------------------------------------------------------------------------
from numba.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_filter
# ---------------------------------------------------------------------------------------------------------------------
class Soccer_Field_Processor(object):
    def __init__(self):
        self.folder_out = './data/output/'
        self.blur_kernel = 2
        return
# ---------------------------------------------------------------------------------------------------------------------
    def line_length(self,x1, y1, x2, y2):return numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
# ---------------------------------------------------------------------------------------------------------------------
    def line_intersection(self,line1, line2):
        def det(a, b): return a[0] * b[1] - a[1] * b[0]

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        x, y = None,None
        div = det(xdiff, ydiff)
        if div == 0:return x, y
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
# ---------------------------------------------------------------------------------------------------------------------
    def skelenonize_fast(self, image,do_debug=False):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = tools_filter.sliding_2d(gray, self.blur_kernel, self.blur_kernel, stat='avg', mode='reflect').astype(numpy.uint8)
        threshholded = 255 - cv2.adaptiveThreshold(255 - blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 0)

        #edges = cv2.Canny(threshholded, 10, 50, apertureSize=3)
        edges = (255 * skeletonize(threshholded / 255)).astype(numpy.uint8)

        lines = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 40, int(image.shape[0]*0.1))

        result = image.copy()
        result = tools_image.desaturate(result)
        skeleton = numpy.zeros(image.shape, dtype=numpy.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                result = cv2.line(result, (x1, y1), (x2, y2), (0, 32, 255), 1)
                skeleton = cv2.line(skeleton, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if do_debug:
            cv2.imwrite(self.folder_out + '1-blur.png', blur)
            cv2.imwrite(self.folder_out + '2-threshholded.png', threshholded)
            cv2.imwrite(self.folder_out + '3-edges.png', edges)
            cv2.imwrite(self.folder_out + '4-res.png', result)
            cv2.imwrite(self.folder_out + 'skeleton.png', skeleton)
        return skeleton
# ---------------------------------------------------------------------------------------------------------------------
    def skelenonize_slow(self, image, do_debug=False):

        L = image.shape[0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = tools_image.desaturate(image.copy(), 0.8)
        skeleton = numpy.full(image.shape, 0, dtype=numpy.uint8)

        filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 0)
        ske = skeletonize(filtered>0).astype(numpy.uint8)

        graph = sknw.build_sknw(ske)
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            xx = ps[:, 1]
            yy = ps[:, 0]
            if self.line_length(xx[0],yy[0],xx[-1],yy[-1]) > int(L/20.0):
                for i in range(len(xx)-1):
                    result = cv2.line(result, (xx[i],yy[i]), (xx[i+1],yy[i+1]), (0, 168, 255),thickness=4)
                    skeleton = cv2.line(skeleton, (xx[i], yy[i]), (xx[i + 1], yy[i + 1]), (255, 255, 255), thickness=4)
                cv2.circle(result, (xx[0] , yy[ 0]), 2, (0, 0, 255), -1)
                cv2.circle(result, (xx[-1], yy[-1]), 2, (0, 0, 255), -1)

        if do_debug:
            cv2.imwrite(self.folder_out + '1-filtered.png', filtered)
            cv2.imwrite(self.folder_out + '2-skimage_skelet.png', ske*255)
            cv2.imwrite(self.folder_out + '3-lines_skelet.png', result)
            cv2.imwrite(self.folder_out + '4-skelenon.png', skeleton)

        return skeleton
# ---------------------------------------------------------------------------------------------------------------------
    def get_hough_lines(self, image, do_debug=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, numpy.pi / 180, int(image.shape[0]*0.1))
        L = max(image.shape[0], image.shape[1])

        result_lines = []
        result_image = 0*image.copy()
        if lines is None or len(lines)==0:
            return result_lines

        for line in lines:
            rho, theta = line[0][0],line[0][1]
            a = numpy.cos(theta)
            b = numpy.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 - L* b)
            y1 = int(y0 + L* a)
            x2 = int(x0 + L* b)
            y2 = int(y0 - L* a)

            result_lines.append([x1, y1, x2, y2])
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 128, 0), 4)

        if do_debug:
            result_image = tools_image.put_layer_on_image(result_image,image,(0,0,0))
            cv2.imwrite(self.folder_out + 'lines_hough.png', result_image)
        return result_lines
# ----------------------------------------------------------------------------------------------------------------------
    def filter_lines(self,H,W,lines,N,a_min,a_max,do_debug=False):

        image_map = numpy.full((H,180,3),0,dtype=numpy.uint8)
        result_image = numpy.full((H, W, 3), 0, dtype=numpy.uint8)
        candidate_lines,result_lines = [],[]

        if lines is None or len(lines)==0:
            return result_lines


        for x1, y1, x2, y2 in lines:
            if x2==x1:continue
            angle = 90+math.atan((y2-y1)/(x2-x1))*180/math.pi
            if angle<=a_min or angle>=a_max:continue
            inters_mid_hor = y1 + (y2 - y1) * (W / 2 - x1) / (x2 - x1)
            candidate_lines.append([angle,inters_mid_hor])

            if do_debug:
                cv2.circle(image_map,(int(angle),int(inters_mid_hor)),3,(255,255,255),-1)
                cv2.line(result_image, (x1, y1), (x2, y2), (255, 128, 0), 4)

        if len(candidate_lines)<6:
            return result_lines

        candidate_lines=numpy.array(candidate_lines)
        kmeans_model = KMeans(n_clusters=5).fit(candidate_lines)
        centers = numpy.array(kmeans_model.cluster_centers_)[:N]
        idx = numpy.argsort(centers[:,1])
        centers = centers[idx,:]


        for center in centers:
            x1,x2=0,W-1
            dy=W/2*math.tan((90-center[0])*math.pi/180)
            y1=center[1]+dy
            y2=center[1]-dy
            result_lines.append([x1,y1,x2,y2])
            if do_debug:
                cv2.line(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 12, 255), 4)

        if do_debug:
            for center in centers:cv2.circle(image_map, (int(center[0]), int(center[1])), 3, (0, 12, 255), -1)
            cv2.imwrite(self.folder_out + 'map.png',image_map)
            cv2.imwrite(self.folder_out + 'lns.png', result_image)

        result_lines = numpy.array(result_lines)

        return result_lines
# ----------------------------------------------------------------------------------------------------------------------
    def four_lines_to_rect(self,side1,side2,side3,side4,do_debug=False):

        xA, yA = self.line_intersection(side1.reshape((2, 2)), side2.reshape((2, 2)))
        xB, yB = self.line_intersection(side2.reshape((2, 2)), side3.reshape((2, 2)))
        xC, yC = self.line_intersection(side3.reshape((2, 2)), side4.reshape((2, 2)))
        xD, yD = self.line_intersection(side4.reshape((2, 2)), side1.reshape((2, 2)))

        return numpy.array([[xA, yA,xD, yD],[xA, yA,xB, yB],[xB, yB,xC, yC],[xC, yC,xD, yD]])
# ----------------------------------------------------------------------------------------------------------------------
    def draw_lines(self,image, lines,color=(0,0,255),w=4):

        result = image.copy()
        for x1,y1,x2,y2 in lines:
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, w)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def process_left_view(self,image, do_debug=False):
        skeleton = self.skelenonize_slow(image, do_debug=do_debug)
        all_lines = self.get_hough_lines(skeleton, do_debug=do_debug)
        left_lines_long = self.filter_lines(image.shape[0], image.shape[1], all_lines, 3, 50, 82, do_debug=do_debug)
        left_lines_short = self.filter_lines(image.shape[0], image.shape[1], all_lines, 4, 91, 120, do_debug=do_debug)
        if len(left_lines_long)==3 and len(left_lines_short)==4:
            goal_lines = self.four_lines_to_rect(left_lines_long[0], left_lines_short[1], left_lines_long[1],left_lines_short[2])
            pen_lines = self.four_lines_to_rect(left_lines_long[0], left_lines_short[0], left_lines_long[2], left_lines_short[3])
            result = self.draw_lines(tools_image.desaturate(image, 0.9), goal_lines, color=(0, 180, 255))
            result = self.draw_lines(result, pen_lines, color=(0, 180, 255))
            result = self.draw_lines(result, [left_lines_long[0]], color=(0, 180, 255))
        else:
            return image
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def process_folder(self,folder_in, folder_out):
        tools_IO.remove_files(folder_out, create=True)
        local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')

        for local_filename in local_filenames:
            image = cv2.imread(folder_in + local_filename)
            result = self.process_left_view(image)
            cv2.imwrite(folder_out + local_filename, result)
            print(local_filename)
        return
# ----------------------------------------------------------------------------------------------------------------------
