import math
import cv2
import numpy
from skimage.morphology import skeletonize,remove_small_holes, remove_small_objects
import sknw
import pyrr
import tools_wavefront
from sklearn.cluster import KMeans
# ---------------------------------------------------------------------------------------------------------------------
from numba.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_filter
import tools_pr_geom
import tools_render_CV
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
        self.knn_th_max_min = 10
        self.method = 'homography' #'pnp'

        self.w_fild = 100
        self.h_fild = 65
        self.w_penl = 18
        self.h_penl = 44
        self.w_goal = 6
        self.h_goal = 20

        self.color_amber      = (0  , 124, 207)
        self.color_white      = (255, 255, 255)
        self.color_red        = (17 ,   6, 124)
        self.color_bright_red = (0  ,  32, 255)
        self.color_green      = (117, 134, 129)
        self.color_black      = (0  ,  0,    0)
        self.color_gray       = (20 , 20,   20)
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
    def skelenonize_slow(self, image, do_debug=None):

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

        if do_debug is not None:
            cv2.imwrite(self.folder_out + '1-filtered.png', filtered)
            cv2.imwrite(self.folder_out + '2-skimage_skelet.png', ske*255)
            cv2.imwrite(self.folder_out + '3-lines_skelet.png', result)
            cv2.imwrite(self.folder_out + '4-skelenon.png', skeleton)



        return skeleton
# ---------------------------------------------------------------------------------------------------------------------
    def get_hough_lines(self, image, do_debug=None):
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
            cv2.line(result_image, (x1, y1), (x2, y2), self.color_green, 4)

        if do_debug is not None:
            result_image = tools_image.put_layer_on_image(result_image,image,(0,0,0))
            cv2.imwrite(self.folder_out + '%02d-lines_hough.png'%do_debug, result_image)
        return result_lines
# ----------------------------------------------------------------------------------------------------------------------
    def get_angle(self,x1, y1, x2, y2):
        return 90 + math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
# ----------------------------------------------------------------------------------------------------------------------
    def get_lines_params(self,lines, skeleton,do_debug=None):

        w, a, c = [], [], []
        if lines is None or len(lines)==0:return w, a, c

        for x1, y1, x2, y2 in lines:
            if x2==x1:
                w.append(0)
                a.append(90)
                c.append(0)
                continue
            a.append(self.get_angle(x1, y1, x2, y2))
            c.append(y1 + (y2 - y1) * (self.W / 2 - x1) / (x2 - x1))

            B = []
            for x in range(0,self.W):
                y = int(y1 + (y2 - y1) * (x - x1) / (x2 - x1))
                if y>=0 and y <self.H:B.append(skeleton[y,x,0])

            w.append(numpy.array(B).sum())

        w = numpy.array(w,dtype=numpy.float)
        if w.max()>0:w*=255/w.max()

        if do_debug is not None:
            image = numpy.full((self.H,self.W,3),0,dtype=numpy.uint8)
            for i in numpy.argsort(w):image = self.draw_lines(image,[lines[i]],(int(w[i]),int(w[i]),int(w[i])),w=3)
            cv2.imwrite(self.folder_out+'weights.png',tools_image.hitmap2d_to_jet(image))

        return numpy.array(w),numpy.array(a),numpy.array(c)
# ----------------------------------------------------------------------------------------------------------------------
    def get_longest_line(self, lines, weights, angles, crosses, do_debug=None):

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


        if do_debug is not None:
            result_image = numpy.full((self.H,self.W,3),0,numpy.uint8)
            result_image = self.draw_lines(result_image, result_lines, color=self.color_green, w=2)
            result_image = self.draw_lines(result_image, [lines[idx]], color=self.color_amber, w=2)
            cv2.imwrite(self.folder_out + '%02d-longest_lines.png'%do_debug, result_image)

        return [lines[idx]]
# ----------------------------------------------------------------------------------------------------------------------
    def clean_skeleton(self,skeleton,line,mode,do_debug=None):
        result = skeleton.copy()
        tol = 25

        x1, y1, x2, y2 = line[0]
        angle = self.get_angle(x1, y1, x2, y2)

        if mode=='up':
            if angle<90:
                pts = numpy.array([[x1, y1-tol], [x2, y2-tol], [0, 0]],dtype=numpy.int)
            else:
                pts = numpy.array([[x1, y1-tol], [x2, y2-tol], [self.W, 0]],dtype=numpy.int)
        else:
            if angle<90:
                pts = numpy.array([[x1, y1+tol], [x2, y2+tol], [self.W, self.H]],dtype=numpy.int)
            else:
                pts = numpy.array([[x1, y1+tol], [x2, y2+tol], [0, self.H]],dtype=numpy.int)



        if do_debug is not None:
            temp = skeleton.copy()
            temp = cv2.drawContours(temp, [pts], 0, self.color_gray, -1)
            temp = self.draw_lines(temp, line, color=self.color_red, w=2)
            cv2.imwrite(self.folder_out + '%02d-skelenon_clean.png'%do_debug, temp)

        cv2.drawContours(result, [pts], 0, self.color_black, -1)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def get_centers(self, candidates, weights, max_C=4):

        best_N = 2
        N_candidates = numpy.arange(2, max_C + 1, 1)
        for N in N_candidates:
            if len(candidates)<N+2:
                continue
            kmeans_model = KMeans(n_clusters=N+2).fit(numpy.array(candidates))
            centers = numpy.array(kmeans_model.cluster_centers_)[:N]
            centers = centers[numpy.argsort(centers[:,1]),:]

            d = []
            for i in range(len(centers)-1):
                for j in range(i+1,len(centers)):
                    xxx = (numpy.array(centers[i]) - numpy.array(centers[j]))
                    d.append( math.sqrt((xxx**2).mean()) )

            dmin = numpy.array(d).min()
            dmax = numpy.array(d).max()
            if dmax/dmin < self.knn_th_max_min:
                best_N = N

        kmeans_model = KMeans(n_clusters=best_N).fit(numpy.array(candidates))

        idx_out = []
        Y = kmeans_model.predict(candidates)
        for n in range(best_N):
            temp_weights = weights.copy()
            temp_weights[numpy.where(Y!=n)]=0
            best = numpy.argsort(-temp_weights)[0]
            idx_out.append(best)

        return idx_out
# ----------------------------------------------------------------------------------------------------------------------
    def remove_intersected_line(self,lines, weights):

        N = len(lines)
        if N<=1:return False, lines, weights

        idx = numpy.argsort(weights)
        flag = False

        for i in range(N-1):
            for j in range(i+1,N):
                x,y = tools_render_CV.line_line_intersection(lines[idx[i]],lines[idx[j]])
                if x is not None and x>=0 and x<self.W and y>=0 and y<self.H:
                    return True,numpy.delete(lines,idx[i],axis=0),numpy.delete(weights,idx[i],axis=0)


        return flag,lines, weights
# ----------------------------------------------------------------------------------------------------------------------
    def filter_lines_by_angle_pos(self, skeleton, lines, lines_weights, line_angles, lines_crosses, a_min, a_max, do_debug=None):

        result_image = numpy.full((self.H, self.W, 3), 0, dtype=numpy.uint8)
        candidates, idx_in, result_lines = [],[],[]

        if lines is None or len(lines)==0:return result_lines

        for line,weight,angle,cross,i in zip(lines,lines_weights, line_angles, lines_crosses,numpy.arange(0,len(lines),1)):
            if angle<=a_min or angle>=a_max:continue
            candidates.append([angle,cross])
            idx_in.append(i)

        if len(candidates)>2:
            idxs_out = self.get_centers(candidates,numpy.array(lines_weights)[idx_in])
            result_lines   = numpy.array(lines        )[numpy.array(idx_in)[idxs_out]]
            result_crosses = numpy.array(lines_crosses)[numpy.array(idx_in)[idxs_out]]
            result_weights = numpy.array(lines_weights)[numpy.array(idx_in)[idxs_out]]

            idx = numpy.argsort(result_crosses)
            result_lines = result_lines[idx]
            result_weights = result_weights[idx]

            success = True
            while success:
                success, result_lines, result_weights = self.remove_intersected_line(result_lines, result_weights)

        else:
            result_lines = numpy.array(lines)[idx_in]


        if do_debug is not None:
            result_image = self.draw_lines(result_image, numpy.array(lines)[idx_in], self.color_green, 4)
            result_image = tools_image.put_layer_on_image(result_image, skeleton, self.color_black)
            result_image = self.draw_lines(result_image, result_lines, self.color_amber, 3)
            cv2.imwrite(self.folder_out + '%02d-lines.png' % do_debug, result_image)

        return result_lines
# ----------------------------------------------------------------------------------------------------------------------
    def get_colors(self,N,do_blend=False):
        colors = []
        for i in range(0, N):
            hue = int(255 * i / (N-1))
            color = cv2.cvtColor(numpy.array([hue, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)[0][0]
            if do_blend:color = (0.25 * numpy.array((255, 255, 255)) + 0.75 * numpy.array(color))
            colors.append((int(color[0]), int(color[1]), int(color[2])))

        return colors
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_lines(self,image, lines,color=(255,255,255),w=4):

        result = image.copy()
        for x1,y1,x2,y2 in lines:
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, w)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def draw_GT_lines(self, image, lines, ids, w=4,put_text=False):

        colors = self.get_colors(16,do_blend=True)
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
    def draw_playground_predict(self, image, landmarks, ids, w=8, R=8):
        result = image.copy()
        if landmarks is None: return result

        colors = self.get_colors(36)
        all_landmarks = numpy.full((36,2),-1,dtype=numpy.int)
        all_landmarks[ids]=landmarks[:]

        idxs = [[0,18,19,1],[2, 8, 9, 3],[4,16,17,5],[26,20,21,27],[34,22,23,35]]

        for idx in idxs:
            cv2.polylines(result, [all_landmarks[idx]], isClosed=True, color=(0,0,190),thickness=w)
            #cv2.fillPoly(result, [points], color=(255,0,0),lineType=cv2.LINE_AA)

        for lm,id in zip(landmarks,ids):
            cv2.circle(result, (int(lm[0]), int(lm[1])), R, colors[id], thickness=-1)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def draw_playground_GT(self, w=8, R=6):
        scale = 4

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

        image = self.draw_GT_lines(image, scale*lines_GT, ids=numpy.arange(0, len(lines_GT), 1), w=w, put_text=True)
        image = self.draw_GT_lmrks(image, scale * L_GT  , ids=numpy.arange(0, len(L_GT)    , 1), R=R, put_text=True)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_playground_homography(self, H, W, homography, check_reflect=False, w=1, R=4):
        image = numpy.full((H, W, 3), 0, numpy.uint8)
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

        image = self.draw_GT_lines(image, lines_GT_trans, ids=numpy.arange(0, len(lines_GT), 1), w=w)
        image = self.draw_GT_lmrks(image, lmrks_GT_trans, ids=numpy.arange(0, len(lmrks_GT), 1), R=R)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_playground_homography_v2(self, H, W, homography,do_reflect=False, w=1, R=4):

        image = numpy.full((H, W, 3), 0, numpy.uint8)
        if homography is None: return image
        lmrks_GT, lines_GT = self.get_GT()

        lines_GT_trans = cv2.perspectiveTransform(lines_GT.reshape((-1, 1, 2)), homography).reshape((-1, 4))
        lmrks_GT_trans = cv2.perspectiveTransform(lmrks_GT.reshape((-1, 1, 2)), homography).reshape((-1, 2))

        padding = 10
        xmin = int(lines_GT_trans[:,0].min())-padding
        xmax = int(lines_GT_trans[:,0].max())+padding
        ymin = int(lines_GT_trans[:,1].min())-padding
        ymax = int(lines_GT_trans[:,1].max())+padding
        image_debug = numpy.full((ymax-ymin,xmax-xmin,3),32,dtype=numpy.uint8)
        colors = self.get_colors(16, do_blend=True)

        for line, id in zip(lines_GT_trans, numpy.arange(0,16,1)):
            cv2.line(image_debug, (int(line[0]-xmin), int(line[1])-ymin), (int(line[2]-xmin), int(line[3])-ymin), colors[id], w)

        image_debug = cv2.resize(image_debug,(image_debug.shape[1]//4,image_debug.shape[0]//4))

        return image_debug
# ----------------------------------------------------------------------------------------------------------------------
    def export_playground(self,filename_obj_out,filename_texture_out):
        Obj = tools_wavefront.ObjLoader()
        landmarks_GT, lines = self.get_GT()
        #landmarks_GT = landmarks_GT[[0,1,2,-1]]

        landmarks_GT3 = numpy.full((landmarks_GT.shape[0], 3), self.GT_Z, dtype=numpy.float)
        landmarks_GT3[:, :2] = landmarks_GT

        coord_texture = (landmarks_GT.copy()).astype(numpy.float)

        coord_texture[:,0]-=coord_texture[:,0].min()
        coord_texture[:,0]/=coord_texture[:,0].max()

        coord_texture[:,1]-= coord_texture[:, 1].min()
        coord_texture[:,1]/= coord_texture[:, 1].max()

        Obj.export_mesh(filename_obj_out,landmarks_GT3,coord_texture)

        cv2.imwrite(filename_texture_out, self.draw_playground_GT())

        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_lines_long(self, lines_long, skeleton, do_debug=None):

        best_loss = None
        best_transform = None
        best_params = None

        if len(lines_long.shape) == 1: lines_long = [lines_long]

        for i, line_long in enumerate(lines_long):
            lines_long_hypot = self.get_hypot_lines_long(skeleton, [line_long])
            lines_short_hypot = self.get_hypot_lines_short(skeleton, [line_long])
            angle = self.get_angle(lines_long_hypot[0][0], lines_long_hypot[0][1], lines_long_hypot[0][2],lines_long_hypot[0][3])

            long_combinations = self.create_long_lines_combinations(lines_long_hypot, angle)
            short_combinations = self.create_short_lines_combinations(lines_short_hypot, angle)

            cmb = -1
            for long_combination in long_combinations:
                for short_combination in short_combinations:
                    cmb += 1
                    comb = numpy.zeros((16, 4))
                    for c in range(16):
                        if numpy.count_nonzero(long_combination[c]) > 0: comb[c] = long_combination[c]
                        if numpy.count_nonzero(short_combination[c])> 0: comb[c] = short_combination[c]

                    landmarks, landm_ids = self.get_landmarks_field(comb)

                    if self.method=='homography':
                        H,params,landmarks_fit  = self.get_homography(landmarks, landm_ids)
                    else:
                        H,params,landmarks_fit = self.get_pose_pnp(landmarks, landm_ids)

                    loss = ((landmarks_fit - landmarks) ** 2).mean()
                    if best_loss is None or loss < best_loss:
                        best_transform, best_params, best_loss = H, params, loss

                    if do_debug is not None:
                        image_comb = self.draw_GT_lines(skeleton.copy(), comb, numpy.arange(0, 16, 1), w=6)
                        image_comb = self.draw_GT_lmrks(image_comb, landmarks, landm_ids, 16, 3)
                        image_comb = self.draw_GT_lmrks(image_comb, landmarks_fit, landm_ids, 6, -1)
                        cv2.imwrite(self.folder_out + 'comb_%03d_%03d_%d.png' % (i,cmb,loss), image_comb)

        return best_transform,best_params
# ----------------------------------------------------------------------------------------------------------------------
    def process_view(self, image, do_debug=None):

        self.H,self.W = image.shape[:2]
        #skeleton = self.skelenonize_slow(image)

        if do_debug is not None:
            #cv2.imwrite(self.folder_out + 'skeleton.png',skeleton)
            skeleton = cv2.imread(self.folder_out + 'skeleton.png')


        lines_coord = self.get_hough_lines(skeleton)
        lines_weights,line_angles,lines_crosses = self.get_lines_params(lines_coord, skeleton)

        longest_line = self.get_longest_line(lines_coord, lines_weights, line_angles, lines_crosses)
        angle = self.get_angle(longest_line[0][0],longest_line[0][1],longest_line[0][2],longest_line[0][3])
        lines_long = self.filter_lines_by_angle_pos(skeleton, lines_coord, lines_weights, line_angles, lines_crosses, angle - 10, angle + 10)

        if do_debug is not None:
            temp = self.draw_lines(skeleton.copy(), lines_long, self.color_red, 4)
            cv2.imwrite(self.folder_out + 'hypotesis.png', temp)

        tools_IO.save_mat(lines_long,self.folder_out + 'lines_long.txt')
        lines_long = tools_IO.load_mat(self.folder_out + 'lines_long.txt',dtype=numpy.int)

        homography, params = self.process_lines_long(lines_long, skeleton, do_debug)

        self.set_playground_params(params)
        playground = self.draw_playground_homography(self.H, self.W, homography, check_reflect=True, w=3, R=6)
        result = tools_image.put_layer_on_image(tools_image.desaturate(image), playground, background_color=(0, 0, 0))

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def get_hypot_lines_long(self, skeleton, longest_line, do_debug=None):
        skeleton = self.clean_skeleton(skeleton, longest_line, mode='up')
        lines_coord = self.get_hough_lines(skeleton)
        lines_weights, line_angles, lines_crosses = self.get_lines_params(lines_coord, skeleton)
        angle = self.get_angle(longest_line[0][0], longest_line[0][1], longest_line[0][2], longest_line[0][3])
        lines_long = self.filter_lines_by_angle_pos(skeleton, lines_coord, lines_weights, line_angles, lines_crosses, angle - 5, angle + 5, do_debug=do_debug)
        return lines_long
# ----------------------------------------------------------------------------------------------------------------------
    def get_hypot_lines_short(self, skeleton, longest_line, do_debug=None):
        skeleton = self.clean_skeleton(skeleton, longest_line, mode='up')
        angle = self.get_angle(longest_line[0][0], longest_line[0][1], longest_line[0][2], longest_line[0][3])
        if angle>90:angle_min,angle_max =45,88
        else:angle_min,angle_max = 92,90+45
        lines_coord = self.get_hough_lines(skeleton)
        lines_weights, line_angles, lines_crosses = self.get_lines_params(lines_coord, skeleton)
        lines_short = self.filter_lines_by_angle_pos(skeleton, lines_coord, lines_weights, line_angles, lines_crosses, angle_min, angle_max, do_debug=do_debug)
        return lines_short
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
        scale_factor = 0.05
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

        #lines[:,0]+=w_fild/2
        #lines[:,1]+=h_fild/2
        #lines[:,2]+=w_fild/2
        #lines[:,3]+=h_fild/2

        lines = scale_factor*numpy.array(lines)

        landmarks = []
        for l1, l2 in self.get_list_of_lines():
            landmarks.append(self.line_intersection(lines[l1].reshape(2,2),lines[l2].reshape(2,2)))

        landmarks = numpy.array(landmarks)
        return landmarks, lines
# ----------------------------------------------------------------------------------------------------------------------
    def get_landmarks_field(self,lines,visible_only=False):

        landmarks,ids = [],[]
        i=0
        for l1, l2 in self.get_list_of_lines():
            if numpy.count_nonzero(lines[l1])>0 and numpy.count_nonzero(lines[l2])>0:
                cross = self.line_intersection(lines[l1].reshape(2, 2), lines[l2].reshape(2, 2))
                if cross[0] is not None and cross[1] is not None:
                    if (not visible_only) or (cross[0]>=0 and cross[0]<=self.W and cross[1]>=0 and cross[1]<self.H):
                        ids.append(i)
                        landmarks.append(cross)
            i+=1

        landmarks=numpy.array(landmarks)
        ids = numpy.array(ids)

        return landmarks, ids
# ----------------------------------------------------------------------------------------------------------------------
    def create_long_lines_combinations(self,lines_long, angle):
        C = []
        if len(lines_long)==3:
            if angle < 90:
                idx_target3 = [[3, 8, 5]]
                idx_target2 = [[3, 5], [3, 8], [5, 8]]
            else:
                idx_target3 = [[1, 14, 11]]
                idx_target2 = [[1, 11], [1, 14], [11, 14]]

            C.append(self.assign_lines(lines_long, [[0, 1, 2]], idx_target3))
            #C.append(self.assign_lines(lines_long, [[0, 1], [1, 2], [0, 2]], idx_target2))

        if len(lines_long)==2:
            if angle > 90:
                idx_target2 = [[1, 11],[1, 14],[11, 14]]
            else:
                idx_target2 = [[3, 5],[3, 8],[5, 8]]

            C.append(self.assign_lines(lines_long, [[0, 1]], idx_target2))

        C = numpy.array(C)
        if len(C)>0:
            C = numpy.concatenate(C, axis=0)

        return C
# ----------------------------------------------------------------------------------------------------------------------
    def create_short_lines_combinations(self, lines_short, angle):
        C,idx_src, idx_target = [],[],[]
        if len(lines_short) == 3:
            idx_src = [[0, 1, 2]]
            if angle < 90:idx_target = [[0, 4, 7]]
            else:idx_target = [[0, 10, 15]]

        if len(lines_short) == 4:
            idx_src = [[0, 1, 2, 3]]
            if angle < 90:
                idx_target = [[0, 4, 7, 9],[4, 7, 9, 6]]
            else:
                idx_target = [[0, 10, 13, 15]]

        C.append(self.assign_lines(lines_short, idx_src, idx_target))
        C = numpy.array(C).reshape((-1, 16, 4))
        return C
# ----------------------------------------------------------------------------------------------------------------------
    def assign_lines(self,lines, idx_source,idx_target):
        result = []
        for s in range(len(idx_source)):
            for t in range(len(idx_target)):
                L = numpy.zeros((16,4),dtype=numpy.float)
                L[idx_target[t]]=numpy.array(lines)[idx_source[s]]
                result.append(L)

        result = numpy.array(result).reshape((-1,16,4))
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def set_playground_params(self,params):
        self.h_fild = params
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_playground_params(self):
        params = self.h_fild
        return params
# ----------------------------------------------------------------------------------------------------------------------
    def get_pose_pnp(self, landmarks, ids):

        best_H, best_loss,result = None, None,None
        best_params = self.get_playground_params()
        if len(ids) <= 3: return best_H,best_params,result

        landmarks_GT, lines_GT = self.get_GT()
        landmarks_GT3 = numpy.full((landmarks_GT.shape[0], 3), self.GT_Z, dtype=numpy.float)
        landmarks_GT3[:, :2] = landmarks_GT

        for params_h in numpy.arange(70 - 5, 70 + 6, 0.25):
            self.set_playground_params(params_h)
            for aperture in numpy.arange(0.50, 2.60, 0.01):
                mat_camera_3x3 = tools_pr_geom.compose_projection_mat_3x3(self.W, self.H, aperture, aperture)
                r_vec, t_vec, result = tools_pr_geom.fit_pnp(landmarks_GT3[ids], landmarks, mat_camera_3x3)
                loss = numpy.sqrt(((result - landmarks) ** 2).mean())
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_H = r_vec, t_vec, aperture
                    best_params = params_h

        self.set_playground_params(best_params)

        return best_H, best_params, result
# ----------------------------------------------------------------------------------------------------------------------
    def get_homography(self, landmarks, ids, do_debug=None):
        best_H, best_loss,result = None, None,None
        best_params = self.get_playground_params()
        if len(ids) <= 3: return best_H,best_params,result


        cnt=0
        for params_h in numpy.arange(70 - 5, 70 + 6, 0.25):
            self.set_playground_params(params_h)

            landmarks_GT, lines_GT = self.get_GT()
            homography, result = tools_pr_geom.fit_homography(landmarks_GT[ids], landmarks,numpy.array(self.get_colors(36))[ids])
            loss =  ((result-landmarks)**2).mean()
            if best_loss is None or loss < best_loss:
                best_H, best_loss= homography, loss
                best_params = params_h
            cnt += 1

        self.set_playground_params(best_params)

        if do_debug is not None:
            image = self.draw_playground_homography(self.H, self.W, best_H, w=3, R=6)
            cv2.imwrite(self.folder_out+'fit_best_%02d.png'%do_debug,image)

        return best_H, best_params, result
# ----------------------------------------------------------------------------------------------------------------------
    def process_folder(self,folder_in, folder_out):
        tools_IO.remove_files(folder_out, create=True)
        local_filenames = tools_IO.get_filenames(folder_in, '*.jpg')

        for local_filename in local_filenames:
            image = cv2.imread(folder_in + local_filename)
            result = self.process_view(image)
            cv2.imwrite(folder_out + local_filename, result)
            print(local_filename)

        return
# ----------------------------------------------------------------------------------------------------------------------
