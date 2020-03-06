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

        self.color_amber = (0, 168, 255)
        self.color_white = (255, 255, 255)
        self.color_red = (0, 32, 255)
        self.color_blue= (255, 128, 0)
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
                result = cv2.line(result, (x1, y1), (x2, y2), self.color_red, 1)
                skeleton = cv2.line(skeleton, (x1, y1), (x2, y2), self.color_while, 1)

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
                    result = cv2.line(result, (xx[i],yy[i]), (xx[i+1],yy[i+1]), self.color_amber,thickness=4)
                    skeleton = cv2.line(skeleton, (xx[i], yy[i]), (xx[i + 1], yy[i + 1]), self.color_white, thickness=4)
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
        #linesp = cv2.HoughLinesP(edges, 1, numpy.pi / 180, int(image.shape[0]*0.1))

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
            cv2.line(result_image, (x1, y1), (x2, y2), self.color_blue, 4)

        if do_debug:
            result_image = tools_image.put_layer_on_image(result_image,image,(0,0,0))
            cv2.imwrite(self.folder_out + '6-lines_hough.png', result_image)
        return result_lines
# ----------------------------------------------------------------------------------------------------------------------
    def get_lines_params(self,lines, skeleton):

        w, a, c = [], [], []
        if lines is None or len(lines)==0:return w, a, c

        for x1, y1, x2, y2 in lines:
            if x2==x1:
                w.append(0)
                a.append(90)
                c.append(0)
                continue
            angle = 90+math.atan((y2-y1)/(x2-x1))*180/math.pi
            cross = y1 + (y2 - y1) * (self.W / 2 - x1) / (x2 - x1)
            a.append(angle)
            c.append(cross)

            B = []
            for x in range(0,self.W):
                y = int(y1 + (y2 - y1) * (x - x1) / (x2 - x1))
                if y>=0 and y <self.H:B.append(skeleton[y,x,0])

            w.append(numpy.array(B).mean())

        return numpy.array(w),numpy.array(a),numpy.array(c)
# ----------------------------------------------------------------------------------------------------------------------
    def get_longest_line(self, lines, weights, angles, crosses, do_debug=False):

        if lines is None or len(lines)==0:return []
        idx = numpy.argsort(-numpy.array(weights))[0]

        result_lines = []
        for line, weight, angle, cross in zip(lines,weights,angles,crosses):
            x1, y1, x2, y2 = line
            if angles[idx] < 90:
                if angle<90:
                    if cross>crosses[idx]:
                        result_lines.append(line)
                else:
                    result_lines.append(line)


        if do_debug:
            result_image = numpy.full((self.H,self.W,3),0,numpy.uint8)
            result_image = self.draw_lines(result_image, result_lines,color=self.color_blue,w=2)
            result_image = self.draw_lines(result_image, [lines[idx]], color=self.color_amber, w=2)
            cv2.imwrite(self.folder_out + '5-longest_lines.png', result_image)

        return [lines[idx]]
# ----------------------------------------------------------------------------------------------------------------------
    def trim_lines(self,lines):
        result = []
        for line in lines:
            x1, y1, x2, y2 = line

            y_begin = int(y1 + (y2 - y1) * (0 - x1) / (x2 - x1))
            if y_begin>=0 and y_begin<self.W:
                x1, y1 = 0,y_begin







        return
# ----------------------------------------------------------------------------------------------------------------------
    def clean_skeleton(self,skeleton, line,do_debug=False):
        result = skeleton.copy()

        x1, y1, x2, y2 = line[0]
        angle = 90 + math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi

        if angle<90:
            pts = numpy.array([[x1, y1-5], [x2, y2-5], [0, 0]],dtype=numpy.int)
        else:
            pts = numpy.array([[x1, y1-5], [x2, y2-5], [self.W, 0]],dtype=numpy.int)

        cv2.drawContours(result, [pts], 0, (0, 0, 0),-1)
        if do_debug:
            cv2.imwrite(self.folder_out + '5-skelenon_clean.png', result)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def get_centers(self,candidates,max_C=4):

        kmeans_model = KMeans(n_clusters=2*max_C).fit(numpy.array(candidates))


        best_N = 2
        N_candidates = numpy.arange(2, max_C + 1, 1)
        for N in N_candidates:
            centers = numpy.array(kmeans_model.cluster_centers_)[:N]
            centers = centers[numpy.argsort(centers[:,1]),:]

            d = []
            for i in range(len(centers)-1):
                for j in range(i+1,len(centers)):
                    xxx = (numpy.array(centers[i]) - numpy.array(centers[j]))
                    d.append( math.sqrt((xxx**2).mean()) )

            dmin = numpy.array(d).min()
            dmax = numpy.array(d).max()
            if dmax/dmin < 5:
                best_N = N

        kmeans_model = KMeans(n_clusters=best_N*2).fit(numpy.array(candidates))
        centers = numpy.array(kmeans_model.cluster_centers_)[:best_N]
        centers = centers[numpy.argsort(centers[:, 1]), :]

        idx_out = []
        for center in centers:
            d = [((numpy.array(candidate)-numpy.array(center))**2).mean() for candidate in candidates]
            idx_out.append(numpy.argsort(d)[0])

        return centers, idx_out
# ----------------------------------------------------------------------------------------------------------------------
    def filter_lines(self,lines,lines_weights, line_angles, lines_crosses,a_min,a_max,do_debug=False):

        image_map = numpy.full((self.H,180,3),0,dtype=numpy.uint8)
        result_image = numpy.full((self.H, self.W, 3), 0, dtype=numpy.uint8)
        candidates, idx_in, result_lines = [],[],[]

        if lines is None or len(lines)==0:return result_lines

        for line,weight,angle,cross,i in zip(lines,lines_weights, line_angles, lines_crosses,numpy.arange(0,len(lines),1)):

            if angle<=a_min or angle>=a_max:continue
            candidates.append([angle,cross])
            idx_in.append(i)

            if do_debug:
                cv2.circle(image_map,(int(angle),int(cross)),3,self.color_white,-1)
                self.draw_lines(result_image,[line],self.color_blue, 4)

        if len(candidates)<=2:
            return result_lines

        centers,idxs_out = self.get_centers(candidates)
        result_lines = numpy.array(lines)[numpy.array(idx_in)[idxs_out]]

        if do_debug:
            result_image = self.draw_lines(result_image, numpy.array(lines)[idx_in], self.color_blue, 4)
            result_image = self.draw_lines(result_image, result_lines, self.color_red, 4)
            for center in centers:cv2.circle(image_map, (int(center[0]), int(center[1])), 3, (0, 12, 255), -1)
            #cv2.imwrite(self.folder_out + 'map_%02d.png'%a_min,image_map)
            cv2.imwrite(self.folder_out + 'lns_%02d.png'%a_min, result_image)

        return result_lines
# ----------------------------------------------------------------------------------------------------------------------
    def four_lines_to_rect(self,side1,side2,side3,side4,do_debug=False):

        xA, yA = self.line_intersection(side1.reshape((2, 2)), side2.reshape((2, 2)))
        xB, yB = self.line_intersection(side2.reshape((2, 2)), side3.reshape((2, 2)))
        xC, yC = self.line_intersection(side3.reshape((2, 2)), side4.reshape((2, 2)))
        xD, yD = self.line_intersection(side4.reshape((2, 2)), side1.reshape((2, 2)))

        return numpy.array([[xA, yA,xD, yD],[xA, yA,xB, yB],[xB, yB,xC, yC],[xC, yC,xD, yD]])
# ----------------------------------------------------------------------------------------------------------------------
    def lines_to_rects(self,lines1,lines2,skeleton):

        for i1 in range(len(lines1)-1):
            for i2 in range(i1+1,len(lines1)):
                for j1 in range(len(lines2) - 1):
                    for j2 in range(j1 + 1, len(lines2)):
                        rect = self.four_lines_to_rect(lines1[i1],lines2[j1],lines1[i2],lines2[j2])
                        w,a,c = self.get_lines_params(rect,skeleton)
                        hhh=0


        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_lines(self,image, lines,color=(255,255,255),w=4):

        result = image.copy()
        for x1,y1,x2,y2 in lines:
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, w)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def process_left_view(self,image, do_debug=False):
        self.H,self.W = image.shape[:2]
        skeleton = self.skelenonize_slow(image, do_debug=do_debug)
        lines_coord = self.get_hough_lines(skeleton, do_debug=do_debug)
        lines_weights,line_angles,lines_crosses = self.get_lines_params(lines_coord, skeleton)

        longest_line = self.get_longest_line(lines_coord, lines_weights, line_angles, lines_crosses, do_debug=do_debug)
        skeleton = self.clean_skeleton(skeleton,longest_line,do_debug=do_debug)
        lines_coord = self.get_hough_lines(skeleton, do_debug=do_debug)
        lines_weights, line_angles, lines_crosses = self.get_lines_params(lines_coord, skeleton)


        lines_long  = self.filter_lines(lines_coord,lines_weights, line_angles, lines_crosses, 50,  82, do_debug=do_debug)
        lines_short = self.filter_lines(lines_coord,lines_weights, line_angles, lines_crosses, 91, 120, do_debug=do_debug)

        lines_long = self.trim_lines(lines_long)
        lines_short = self.trim_lines(lines_short)

        lines_rects = self.lines_to_rects(lines_long,lines_short,skeleton)

        result = self.draw_lines(tools_image.desaturate(image, 0.9), lines_long, color=self.color_red)
        result = self.draw_lines(result, lines_short, color=self.color_amber)


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
