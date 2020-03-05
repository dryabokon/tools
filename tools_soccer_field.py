import math
import cv2
import numpy
from skimage.morphology import skeletonize,remove_small_holes, remove_small_objects
import sknw
import imutils
# ---------------------------------------------------------------------------------------------------------------------
from numba.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
# ---------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_filter
import tools_draw_numpy
# ---------------------------------------------------------------------------------------------------------------------
class Soccer_Field_Processor(object):
    def __init__(self):
        self.folder_out = './data/output/'
        self.blur_kernel = 2
        return
# ---------------------------------------------------------------------------------------------------------------------
    def line_length(self,x1, y1, x2, y2):return numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
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
        result = 0*image.copy()
        result = tools_image.desaturate(result)

        lines = cv2.HoughLines(edges, 1, numpy.pi / 180, int(image.shape[0]*0.1))

        L = max(image.shape[0], image.shape[1])

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

            #tan = (x2 - x1) / (y2 - y1)
            #print(math.atan(tan) * 180 / math.pi)
            cv2.line(result, (x1, y1), (x2, y2), (255, 128, 0), 4)

        if do_debug:
            result = tools_image.put_layer_on_image(result,image,(0,0,0))
            cv2.imwrite(self.folder_out + 'lines_hough.png', result)
        return
# ----------------------------------------------------------------------------------------------------------------------