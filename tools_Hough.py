import cv2
import numpy
from skimage.transform import hough_line,probabilistic_hough_line
# ----------------------------------------------------------------------------------------------------------------
import tools_image
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------
class Hough(object):
    def __init__(self):
        self.name = "Hough"
        self.folder_out = './images/output/'
        return
# ----------------------------------------------------------------------------------------------------------------
    def preprocess(self,image, min_length=100):

        gray = tools_image.desaturate_2d(image)
        lines = probabilistic_hough_line(gray, line_length=min_length)

        result = numpy.zeros_like(gray)
        for line in lines:
            cv2.line(result, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])),
                     color=(255, 255, 255), thickness=4)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def get_lines_cv(self, skeleton,the_range,min_weight,max_count=10):

        candidates = cv2.HoughLines(skeleton, 10, (numpy.pi/180)/20, threshold=0,min_theta=the_range[0],max_theta=the_range[-1])
        #candidates= cv2.HoughLinesP(gray, 1, numpy.pi / 180, int(image.shape[0]*0.1),minLineLength=20)
        L = max(skeleton.shape[0], skeleton.shape[1])

        lines, weights = [],[]
        for line in candidates:
            if len(lines) < max_count:
                rho, theta = line[0][0],line[0][1]
                a = numpy.cos(theta)
                b = numpy.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 - L* b)
                y1 = int(y0 + L* a)
                x2 = int(x0 + L* b)
                y2 = int(y0 - L* a)
                if self.line_exists((x1, y1, x2, y2), lines): continue
                lines.append([x1, y1, x2, y2])


        weights = numpy.full(len(lines),255)

        return lines, weights
# ----------------------------------------------------------------------------------------------------------------------
    def sort_lines_and_weights(self,lines,weights):
        if len(lines)<=1:
            return lines,weights

        xmin = min(lines[:,0].max(),lines[:,2].min())
        xmax = max(lines[:,0].min(),lines[:,2].max())
        ymin = min(lines[:,1].max(),lines[:,3].min())
        ymax = max(lines[:,1].min(),lines[:,3].max())

        middle = numpy.array([(xmin+xmax)//2,ymin,(xmin+xmax)//2,ymax])
        inter=[]
        for line in lines:
            inter.append(tools_render_CV.line_intersection(middle,line))

        inter = numpy.array(inter)
        idx = numpy.argsort(inter[:,1])
        return lines[idx], weights[idx]
# ----------------------------------------------------------------------------------------------------------------------
    def get_lines_ski(self,skeleton,the_range,min_weight,max_count=10):

        hspace, angles, dist = hough_line(skeleton,the_range)
        #hspace, angles, dist = probabilistic_hough_line(gray,theta=the_range)

        hspace = numpy.array(hspace,dtype=numpy.float)
        hspace *= 255 / hspace.max()

        idx_all = numpy.dstack(numpy.unravel_index(numpy.argsort(-hspace.ravel()), (hspace.shape[0], hspace.shape[1])))[0]

        lines,weights = [],[]
        L = (skeleton.shape[0] + skeleton.shape[1]) // 2

        for n,idx in enumerate(idx_all):
            w = hspace[idx[0], idx[1]]
            th = angles[idx[1]]
            dst = dist[idx[0]]

            if w>=min_weight and len(lines)<max_count:

                a = numpy.cos(th)
                b = numpy.sin(th)
                tan = numpy.tan(th)
                x0 = a * dst
                y0 = b * dst
                x1 = (x0 - L * b)
                y1 = (y0 + L * a)
                x2 = (x0 + L * b)
                y2 = (y0 - L * a)

                if self.is_boarder_line((x1,y1,x2,y2),skeleton.shape[0],skeleton.shape[1]):continue
                if self.line_exists((x1, y1, x2, y2),lines):continue
                lines.append((x1, y1, x2, y2))
                weights.append(w)
            else:
                break

        lines,weights = self.sort_lines_and_weights(numpy.array(lines),numpy.array(weights))

        return lines, weights
# ----------------------------------------------------------------------------------------------------------------------
    def get_lines_ski_segments(self, image, the_range, min_weight, max_count=10):

        gray = tools_image.desaturate_2d(image)

        candidates = probabilistic_hough_line(gray,threshold=10, line_length=50, line_gap=10,theta=the_range)
        lines, weights = [],[]
        for line in candidates[:max_count]:
            lines.append([line[0][0], line[0][1], line[1][0], line[1][1]])


        weights = numpy.full(len(lines),255)
        return lines, weights
# ----------------------------------------------------------------------------------------------------------------------
    def is_boarder_line(self,line,H,W):
        x1, y1, x2, y2 = line
        if y1 >=H*0.95 and y2>=H*0.95:
            return True
        return
# ----------------------------------------------------------------------------------------------------------------------
    def line_exists0(self,point,angle,points,angles):

        for p,a in zip(points,angles):
            delta_p = numpy.abs(numpy.array(p)-numpy.array(point))
            if delta_p.max()<10 and numpy.abs(a*180/numpy.pi - 180*angle/numpy.pi)<2:
                return True

        return False
# ----------------------------------------------------------------------------------------------------------------------
    def line_exists(self,candidate,lines):

        tol = 10
        for line in lines:
            d = tools_render_CV.distance_between_lines(candidate, line, clampA0=True, clampA1=True, clampB0=False,clampB1=False)[2]
            if d<tol:
                return True

        return False
# ----------------------------------------------------------------------------------------------------------------------