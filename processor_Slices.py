import numpy
import cv2
import time
from scipy import ndimage
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

        self.Skelenonizer = tools_Skeletone.Skelenonizer()
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
    def get_granules0(self, binarized):

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
            cv2.imwrite(self.folder_out + 'grain_%02d.png' % the_range[i], tools_image.put_layer_on_image(tools_image.saturate(binarized), image_temp))

        for i in range(len(the_range)):granules[the_range[i]] = len(numpy.where(image_result == colors[i])[0])

        return granules, image_result
# ----------------------------------------------------------------------------------------------------------------------
    def get_granules(self, binarized):

        granules = {}
        the_range = numpy.arange(2, 26, 1)
        colors = tools_IO.get_colors(len(the_range), colormap='viridis')
        kernel = numpy.full((3, 3), 255, numpy.uint8)
        empty = numpy.zeros((binarized.shape[0], binarized.shape[1], 3), dtype=numpy.uint8)
        image_result = empty.copy()

        responce = numpy.zeros((len(the_range),binarized.shape[0], binarized.shape[1]), dtype=numpy.uint8)

        for i in range(len(the_range)):
            n = the_range[i]
            A = tools_filter.sliding_2d(binarized, -n, n, -n, n).astype(numpy.uint8)
            A = cv2.dilate(A, kernel=kernel, iterations=n - 1)
            responce[i]=1*(A == 255)

        mask = 0*responce[-1]

        for i in reversed(range(len(the_range))):
            local_responce = responce[i] ^ mask
            granules[the_range[i]] = (local_responce).sum()
            image_result[local_responce > 0] = colors[i]
            mask = mask | responce[i]
        return granules, image_result

# ----------------------------------------------------------------------------------------------------------------------
    def process_file_granules(self,filename_in,do_debug=False):
        base_name = filename_in.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_in)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (7, 7))
        binarized = self.Skelenonizer.binarize(gray)

        time_start = time.time()
        histo, image_grains = self.get_granules(binarized)

        if do_debug:
            #tools_plot.plot_histo(histo, self.folder_out + base_name + '_histo.png',colors=tools_IO.get_colors(len(histo), colormap='viridis'))
            cv2.imwrite(self.folder_out + base_name + '_grains.png',image_grains)

        print('%s - %1.2f sec' % (base_name, (time.time() - time_start)))

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
            self.draw_flow_velocity(flow, self.folder_out + base_name + '_velocity.png')

        return
# ----------------------------------------------------------------------------------------------------------------------
