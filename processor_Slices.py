import numpy
import cv2
import time
import imutils
from scipy import ndimage
# --------------------------------------------------------------------------------------------------------------------
import tools_plot
import tools_image
import tools_Skeletone
import tools_filter
import tools_draw_numpy
# --------------------------------------------------------------------------------------------------------------------
class processor_Slices(object):
    def __init__(self,folder_out):
        self.name = "detector_Zebra"
        self.folder_out = folder_out
        self.Ske = tools_Skeletone.Skelenonizer(folder_out)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_file_granules(self, filename_in, do_debug=False):
        base_name = filename_in.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_in)
        time_start = time.time()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (11, 11))
        binarized = self.Ske.binarize(gray)

        histo, image_result = self.detect_slices(binarized,do_debug)
        if do_debug:
            cv2.imwrite(self.folder_out + '1-gray.png', gray)
            cv2.imwrite(self.folder_out + '2-gray_bin.png', binarized)

        tools_plot.plot_histo(histo, self.folder_out + '_histo.png',colors=tools_draw_numpy.get_colors(len(histo), colormap='viridis'))

        print('%s - %1.2f sec' % (base_name, (time.time() - time_start)))

        return

# ----------------------------------------------------------------------------------------------------------------------
    def get_kernel(self, W, H=10, angle=0):

        kernel = numpy.full((W,W),0,dtype=numpy.uint8)
        kernel[W//2-H//4:W//2+H//4,:W]=255
        if angle!=0:
            kernel = tools_image.rotate_image(kernel,angle)
        return kernel
# ----------------------------------------------------------------------------------------------------------------------
    def get_responce(self,binarized,kernel):
        return ndimage.binary_opening(binarized > 0, structure=kernel > 0)
# ----------------------------------------------------------------------------------------------------------------------
    def get_responc2(self,binarized,kernel):
        #A = tools_filter.sliding_2d(binarized, -n, n, -n, n).astype(numpy.uint8)
        #B = cv2.dilate(A, kernel=kernel, iterations=n - 1)
        #responce = 1 * (B == 255)


        return responce
# ----------------------------------------------------------------------------------------------------------------------
    def detect_slices(self, binarized,do_debug=False):

        granules = {}
        #the_range = numpy.arange(10, 90, 5)
        the_range = [80]
        colors = tools_draw_numpy.get_colors(len(the_range), colormap='viridis')
        responce = numpy.zeros((len(the_range), binarized.shape[0], binarized.shape[1]), dtype=numpy.uint8)
        image_result = numpy.zeros((binarized.shape[0], binarized.shape[1], 3), dtype=numpy.uint8)

        for i,W in enumerate(the_range):
            for angle in range(0,180,10):
                kernel = self.get_kernel(W,angle=angle)
                responce[i][self.get_responce(binarized,kernel)>0]=255
                cv2.imwrite(self.folder_out + 'K.png', kernel)
                cv2.imwrite(self.folder_out + 'resp.png',responce[i])

                cv2.imwrite(self.folder_out + 'local_%03d.png'%W, responce[i])

        mask = 0 * responce[-1]
        for i in reversed(range(len(the_range))):
            local_responce = responce[i] ^ mask
            #granules[the_range[i]] = (local_responce).sum()
            image_result[local_responce > 0] = colors[i]
            mask = mask | responce[i]

        cv2.imwrite(self.folder_out + 'result.png', image_result)


        return granules, image_result
# ----------------------------------------------------------------------------------------------------------------------
