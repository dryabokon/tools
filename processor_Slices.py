import numpy
import cv2
import time
from scipy import ndimage
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_IO
import tools_Skeletone
import tools_filter
import tools_plot
# --------------------------------------------------------------------------------------------------------------------
class processor_Slices(object):
    def __init__(self,folder_out):
        self.name = "detector_Zebra"
        self.folder_out = folder_out
        self.th = 200
        self.W_min = 3
        self.W_max = 27

        self.L = 10
        self.pad = 2
        self.zebra_rotations = 3
        self.gray_prev = None

        self.init_zebras('./images/ex_bin/cached.dat')
        self.Skelenonizer = tools_Skeletone.Skelenonizer()
        return
# ----------------------------------------------------------------------------------------------------------------
    def init_zebras(self,filename_cached):

        self.zebras,success = tools_IO.load_if_exists(filename_cached)
        if success:
            if len(self.zebras) != self.W_max - self.W_min+1:
                success = False
            elif len(self.zebras[0])!= self.zebra_rotations:
                success = False
            if success:
                return

        self.zebras = []
        for w in range(self.W_min,self.W_max+1):

            temp_list = []
            image_zebra = self.create_zebra(W=w, L=self.L, pad=self.pad)

            for r in range(self.zebra_rotations):
                if r == 0:
                    image_zebra_rotated = image_zebra.copy()
                elif r == self.zebra_rotations - 1:
                    image_zebra_rotated = image_zebra.copy().T
                else:
                    angle = 90 * r / (self.zebra_rotations - 1)
                    image_zebra_rotated = tools_image.rotate_image(image_zebra, angle)
                temp_list.append(image_zebra_rotated)

            self.zebras.append(temp_list)

        tools_IO.write_cache(filename_cached,self.zebras)

        return
# ----------------------------------------------------------------------------------------------------------------
    def create_zebra(self, W, L, pad):
        image = numpy.full((L,W+2*pad),0,dtype=numpy.uint8)
        image[:,pad:-pad] = 255
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def convolve_with_mask(self,image_large,image_mask):

        if image_large.max()>1:image_large=image_large/255
        image_large = image_large.astype(numpy.uint8)

        if image_mask.max() > 1: image_mask = image_mask / 255
        image_mask = image_mask.astype(numpy.uint8)

        mask_white = image_mask.copy()
        mask_white[:, :] = 0
        mask_white[image_mask>0]=255

        N_white = numpy.count_nonzero(mask_white)
        N_black = numpy.count_nonzero(255-mask_white)

        loss_white = cv2.matchTemplate(255-255*image_large, 255*image_mask, method=cv2.TM_SQDIFF, mask=mask_white)
        loss_black = cv2.matchTemplate(255-255*image_large, 255*image_mask, method=cv2.TM_SQDIFF, mask=255-mask_white)

        hitmap_gray= 255*(loss_white/N_white)*(loss_black/N_black)
        hitmap_gray = tools_image.canvas_extrapolate(hitmap_gray, image_large.shape[0], image_large.shape[1])

        return hitmap_gray
# ----------------------------------------------------------------------------------------------------------------------
    def compose_debug_image(self, image_large, image_prob, image_zebra):

        image_prob[image_prob < self.th] = 0
        image_prob[image_prob >= self.th] = 255
        image_prob = tools_image.saturate(image_prob)
        image_prob[:, :, [0, 1]] = 0

        scale = 1
        if image_large.max() == 1: scale = 255

        is_bw = numpy.count_nonzero(image_large==0)> image_large.shape[0]*image_large.shape[1]*0.1

        image_bg = tools_image.saturate(image_large * scale)
        if is_bw:
            image_bg[image_bg != 0] = 128
            image_bg[image_bg == 0] = 32

        result = tools_image.put_layer_on_image(image_bg, image_prob, background_color=(0, 0, 0))

        temp = numpy.full((image_zebra.shape[0]*2,image_zebra.shape[1]*2,3),(0),dtype=numpy.uint8)
        temp[:,:] = [128,64,0]
        result = tools_image.put_image(result, temp, 0, 0)
        result = tools_image.put_image(result, tools_image.saturate(image_zebra), 0, 0)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def compose_debug_image_v2(self,binarized, image_prob,image_zebra):

        # result  = tools_image.hitmap2d_to_jet(image_prob)
        result = tools_image.saturate(image_prob)

        temp = numpy.full((image_zebra.shape[0] * 2, image_zebra.shape[1] * 2, 3), (0), dtype=numpy.uint8)
        temp[:, :] = [128, 64, 0]
        result = tools_image.put_image(result, temp, 0, 0)
        result = tools_image.put_image(result, tools_image.saturate(image_zebra), 0, 0)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def detect_light_areas0(self,image, min_size):

        image2 = numpy.array(image.copy(),dtype=numpy.long)
        image2 = image2**2

        mean_x  = tools_filter.sliding_2d(image, -min_size,min_size,-min_size,min_size, stat='avg')
        mean_xx = tools_filter.sliding_2d(image2,-min_size, min_size, -min_size, min_size, stat='avg')

        A = numpy.sqrt(mean_xx - mean_x**2).astype(numpy.uint8)
        mask = 255*(A<3.0)

        labeled_image = (ndimage.label(mask)[0]).astype(numpy.float)
        labeled_image*=255/labeled_image.max()

        cv2.imwrite(self.folder_out + 'A.png', mean_x)
        cv2.imwrite(self.folder_out + 'M.png',tools_image.hitmap2d_to_viridis(mask))
        cv2.imwrite(self.folder_out + 'L.png', tools_image.hitmap2d_to_jet(labeled_image))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_labeled_image(self,mask):

        kernel = numpy.ones((3, 3), numpy.uint8)
        mask = cv2.erode(mask, kernel,iterations=1)
        mask = cv2.dilate(mask, kernel,iterations=1)

        #labels = (skimage.measure.label(mask, connectivity=2)).astype(numpy.float)
        #nb_labels = int(labels.max()+1)

        labels,nb_labels = ndimage.label(mask)

        sizes = ndimage.sum(mask, labels, range(nb_labels + 1))
        #H = numpy.histogram(labels.flatten(),bins=numpy.arange(0,nb_labels+1))
        labels_new = 0*labels.copy()

        for l in range(nb_labels):
            labels_new[labels==l]=sizes[l]

        labels = labels_new.astype(numpy.float)
        labels *= 255 / labels.max()
        labels = tools_image.hitmap2d_to_jet(labels)
        return labels
# ----------------------------------------------------------------------------------------------------------------------
    def get_granules0(self,mask,do_debug=False):

        granules = {}
        the_range = numpy.arange(2, 26, 1)
        colors = tools_IO.get_colors(len(the_range), colormap='viridis')
        image_result = numpy.zeros((mask.shape[0],mask.shape[1],3),dtype=numpy.uint8)
        responces = []

        for n in the_range:
            struct = numpy.zeros((n + 2, 2 * n),dtype=numpy.bool)
            struct[1:-1, :] = True
            responces.append(ndimage.binary_opening(mask, structure=struct))

        for i in range(len(the_range)):
            image = tools_image.saturate(255*responces[i])
            idx = numpy.where(image > 0)
            image[idx[0],idx[1],:]=colors[i]
            image_result = tools_image.put_layer_on_image(image_result,image,(0,0,0))

        for i in range(len(the_range)):
            granules[the_range[i]] = len(numpy.where(image_result == colors[i])[0])

        if do_debug:
            cv2.imwrite(self.folder_out + 'orig_all.png', image_result)


            for i in range(len(the_range)):
                image_temp = numpy.zeros((mask.shape[0], mask.shape[1], 3), dtype=numpy.uint8)
                idx = numpy.where(image_result == colors[i])
                image_temp[idx[0], idx[1], :] = colors[i]

                image_temp = tools_image.put_layer_on_image(tools_image.saturate(mask),image_temp)
                cv2.imwrite(self.folder_out + 'grain_%02d.png'%the_range[i], image_temp)


        return granules
# ----------------------------------------------------------------------------------------------------------------------
    def get_granules(self, binarized, do_debug=False):

        granules = {}
        the_range = numpy.arange(2, 26, 1)
        colors = tools_IO.get_colors(len(the_range), colormap='viridis')
        empty = numpy.zeros((binarized.shape[0], binarized.shape[1], 3), dtype=numpy.uint8)
        image_result = empty.copy()
        kernel = numpy.full((3, 3), 255, numpy.uint8)

        for i in range(len(the_range)):
            n = the_range[i]
            A = tools_filter.sliding_2d(binarized,-n,n,-n,n).astype(numpy.uint8)
            A = cv2.dilate(A,kernel=kernel,iterations=n-1)
            idx = numpy.where(A==255)
            image_temp = empty.copy()
            image_temp[idx[0], idx[1], :] = colors[i]
            image_result = tools_image.put_layer_on_image(image_result, image_temp)

            if do_debug:
                cv2.imwrite(self.folder_out + 'grain_%02d.png' % the_range[i], tools_image.put_layer_on_image(tools_image.saturate(binarized), image_temp))

        for i in range(len(the_range)):granules[the_range[i]] = len(numpy.where(image_result == colors[i])[0])

        if do_debug:
            cv2.imwrite(self.folder_out + 'grain.png', image_result)

        return granules
# ----------------------------------------------------------------------------------------------------------------------
    def process_file_ske(self,filename_in,do_debug=False):

        base_name = filename_in.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_in)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray,(7,7))
        binarized = self.Skelenonizer.binarize(gray)

        time_start = time.time()
        self.Skelenonizer.binarized_to_skeleton_large(binarized,do_inverce=True)
        self.Skelenonizer.remove_edges(min_length=10)
        self.Skelenonizer.remove_clusters(min_count=4, min_length=50)
        histo = self.Skelenonizer.get_width_distribution(binarized, by_column=True)

        if do_debug:
            cv2.imwrite(self.folder_out + base_name + '_gre.png', gray)
            cv2.imwrite(self.folder_out + base_name + '_bin.png', binarized)
            tools_plot.plot_histo(histo,self.folder_out + 'histo.png')
            cv2.imwrite(self.folder_out + base_name + '_bin.png',self.Skelenonizer.draw_skeleton(image_bg=binarized, draw_thin_lines=True))

        print('%s - %1.2f sec' % (base_name, (time.time() - time_start)))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_file_granules(self,filename_in,do_debug=False):
        base_name = filename_in.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_in)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (7, 7))
        binarized = self.Skelenonizer.binarize(gray)

        time_start = time.time()
        histo = self.get_granules(binarized, do_debug=do_debug)

        if do_debug:
            colors = tools_IO.get_colors(len(histo), colormap='viridis')
            tools_plot.plot_histo(histo, self.folder_out + 'histo.png',colors=colors)

        print('%s - %1.2f sec' % (base_name, (time.time() - time_start)))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_flow_direction_sparce(self, flow,filename_out):

        h, w = flow.shape[:2]
        step = h//10

        y, x = numpy.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1)
        fx, fy = flow[y, x].T
        lines = numpy.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = numpy.int32(lines + 0.5)
        lines = lines.reshape((lines.shape[0], -1))
        fig = plt.figure(figsize=(w//100, h//100),dpi=100,facecolor=(0.5, 0.5, 0.5))
        plt.xlim([0, w])
        plt.ylim([h, 0])
        plt.tight_layout()
        #plt.tick_params(axis='off', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',labelright='off', labelbottom='off')
        plt.axis('off')
        #ax = plt.gca()
        #ax.set_facecolor((0.5, 0.5, 0.5))

        idx = numpy.random.choice(len(lines),100)
        for line in lines[idx,:]:tools_plot.plot_gradient_rbg_pairs((line[0],line[1]), (line[2],line[3]), (1,1,1),(1,0,0))
        plt.savefig(filename_out)
        plt.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_flow_direction(self, flow, filename_out):

        result_angle = numpy.full((flow.shape[0],flow.shape[1]),0,dtype=numpy.float)
        idx = numpy.where(flow[:,:,0]>0)
        result_angle[idx[0],idx[1]] = (flow[idx[0],idx[1],1]/flow[idx[0],idx[1],0]).astype(numpy.float)
        result_angle[idx[0],idx[1]] = numpy.arctan(result_angle[idx[0],idx[1]])   + numpy.pi/2

        idx = numpy.where(flow[:, :, 0] < 0)
        result_angle[idx[0], idx[1]] = (flow[idx[0], idx[1], 1] / flow[idx[0], idx[1], 0]).astype(numpy.float)
        result_angle[idx[0], idx[1]] = numpy.arctan(result_angle[idx[0], idx[1]]) +  3*numpy.pi/2

        result_angle/=(2*numpy.pi)
        result_angle*=255

        result_angle = numpy.array(result_angle,dtype=numpy.uint8)
        cv2.imwrite(filename_out, result_angle)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_flow_velocity(self, flow, filename_out,norm = 10):
        result_velocity = numpy.sqrt(flow[:,:,1]**2 + flow[:,:,0]**2)
        numpy.clip(result_velocity/(norm/255),0,255).astype(numpy.uint8)
        cv2.imwrite(filename_out, result_velocity)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_file_flow(self,filename_in,do_debug=False):

        base_name = filename_in.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_in)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.gray_prev is None:
            self.gray_prev = gray
        else:
            flow = cv2.calcOpticalFlowFarneback(self.gray_prev, gray,flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            self.gray_prev = gray.copy()
            flow = numpy.array(flow,dtype=numpy.float)

            self.draw_flow_direction(flow, self.folder_out + base_name + '_angle.png')
            #self.draw_flow_velocity(flow, self.folder_out + base_name + '_velocity.png')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def generate_flow_animation(self):



        return
# ----------------------------------------------------------------------------------------------------------------------
