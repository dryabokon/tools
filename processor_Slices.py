import numpy
import cv2
import time
import imutils
# --------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_Skeletone
import tools_filter
import tools_draw_numpy
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

        self.Ske = tools_Skeletone.Skelenonizer(folder_out)
        self.kernel = self.get_kernel()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_kernel(self):
        W,H = 16,36
        step_while = 8
        step_black = 8
        kernel = numpy.full((H, W), +1, dtype=numpy.float32)

        sign = +1
        for c in range(0, W):
            kernel[:, c] = sign
            if sign>0 and c%(step_while+step_black)%(step_while)==step_while-1:
                sign = -1
            if sign<0 and c%(step_while+step_black) == step_while +step_black-1:
                sign = +1

        return kernel
# ----------------------------------------------------------------------------------------------------------------------
    def process_file_granules(self, filename_in, do_debug=False):
        base_name = filename_in.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_in)
        time_start = time.time()

        #image = imutils.rotate(image, angle=45)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (11, 11))
        binarized = self.Ske.binarize(gray)

        self.detect_slices(binarized,do_debug)
        if do_debug:
            cv2.imwrite(self.folder_out + '1-gray.png', gray)
            cv2.imwrite(self.folder_out + '2-gray_bin.png', binarized)

        print('%s - %1.2f sec' % (base_name, (time.time() - time_start)))

        return

# ----------------------------------------------------------------------------------------------------------------------
    def detect_slices(self,binarized,do_debug=False):

        #binarized = self.Ske.morph(binarized,kernel_h=3,kernel_w=3,n_dilate=3,n_erode=3)
        gray =  tools_image.saturate(binarized) / 2

        image_ske = self.Ske.binarized_to_skeleton(binarized)
        #image_ske = cv2.dilate(image_ske, kernel= cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2, 2)), iterations=1)
        if do_debug:
            cv2.imwrite(self.folder_out + '3-Ske.png', image_ske)

        segm = self.Ske.skeleton_to_segments(image_ske)


        if do_debug:
            image_segm = tools_draw_numpy.draw_segments(gray, segm,color=self.Ske.get_segment_colors(segm), w=2)
            cv2.imwrite(self.folder_out + '4-Segm.png', image_segm)

        segm2 = self.Ske.sraighten_segments(segm)


        if do_debug:
            image_segm2 = tools_draw_numpy.draw_segments(gray, segm2,color=self.Ske.get_segment_colors(segm2), w=2)
            cv2.imwrite(self.folder_out + '5-Segm_str.png', image_segm2)

        lines = [self.Ske.interpolate_segment_by_line(s) for s in segm2]
        if do_debug:
            image_lines = tools_draw_numpy.draw_lines(gray, lines,color=self.Ske.get_segment_colors(segm2), w=2)
            cv2.imwrite(self.folder_out + '6-Segm_smooth.png', image_lines)

        return
# ----------------------------------------------------------------------------------------------------------------------