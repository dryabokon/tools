# Silin number: # of slices longer > 1 cm per 100g
# Swedish number: (mass of slices>5 cm) /(mass of slices < 1cm)
import numpy
import cv2
import time
import json
# --------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_Skeletone
import tools_draw_numpy
import tools_plot
import tools_IO
import tools_filter
# --------------------------------------------------------------------------------------------------------------------
class processor_Slices(object):
    def __init__(self,folder_out):
        self.name = "detector_Zebra"
        self.folder_out = folder_out
        self.folder_cache = folder_out
        self.Ske = tools_Skeletone.Skelenonizer(folder_out)

        self.kernel_H = 8
        #self.range_length = numpy.array([100,50,20])
        self.range_length = numpy.arange(100, 10, -20)
        self.range_angle  = numpy.arange(0, 180, 10)
        self.roi_top, self.roi_left, self.roi_bottom, self.roi_right = None, None,None,None
        #self.roi_top, self.roi_left, self.roi_bottom, self.roi_right = 100,100,600,600

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_kernel(self, W, H, angle=0):

        kernel = numpy.full((W, W), 0,dtype=numpy.uint8)
        kernel[W // 2 - H // 4:W // 2 + H // 4, :W] = +1
        if angle != 0:
            kernel = tools_image.rotate_image(kernel, angle)
        return kernel
# ----------------------------------------------------------------------------------------------------------------------
    def get_responce(self,binarized,kernel):

        min_value = 0
        max_value = 255*kernel.sum()

        A = tools_image.convolve_with_mask(binarized,kernel,min_value,max_value)
        A[A < 255] = 0

        B = cv2.dilate(A, kernel=255*kernel, iterations=1)
        return B
# ----------------------------------------------------------------------------------------------------------------------
    def detect_slices(self, binarized,base_name,do_debug=False):

        granules = {}

        colors = tools_draw_numpy.get_colors(len(self.range_length), colormap='viridis')
        image_result = numpy.zeros((binarized.shape[0], binarized.shape[1], 3), dtype=numpy.uint8)
        longer_responces = numpy.zeros_like(binarized)

        norm = 255*binarized.shape[0]*binarized.shape[1]/100

        for i,W in enumerate(self.range_length):
            local_responce = numpy.zeros_like(binarized)
            for angle in self.range_angle:
                local_responce |= self.get_responce(binarized, self.get_kernel(W, self.kernel_H, angle=angle))

            local_responce ^= longer_responces
            longer_responces |= local_responce
            image_result[local_responce > 0] = colors[len(self.range_length) - i - 1]
            granules[str(W)]=str(int(local_responce.sum()/norm))

        return granules, image_result
# ----------------------------------------------------------------------------------------------------------------------
    def detect_slices_unstable_slow(self, binarized, base_name, do_debug=False):

        granules = {}

        colors = tools_draw_numpy.get_colors(len(self.range_length), colormap='viridis')

        H,W = binarized.shape[:2]
        R = numpy.sqrt( (W/2)**2 + (H/2)**2)
        pad_w = int(R-W/2)
        pad_h = int(R-H/2)
        padded = numpy.pad(binarized,((pad_h,pad_h),(pad_w,pad_w)),'constant')

        H = 10

        local_responce = numpy.zeros((len(self.range_angle),padded.shape[0],padded.shape[1]),dtype=numpy.uint8)

        for angle in self.range_angle:
            image_rotated = 1*(tools_image.rotate_image(padded, angle)>0)
            #cv2.imwrite(self.folder_out + 'Rot_a_%03d.png' % angle, 255*image_rotated)
            image_I = tools_filter.integral_2d(image_rotated)

            for i, W in enumerate(self.range_length):
                kernel = 255 * self.get_kernel(W, H=H, angle=0).astype(numpy.uint8)

                conv = tools_filter.sliding_I_2d(image_I, -H//2, +H//2, -W//2, +W//2, pad=10, stat='avg', mode='constant')
                conv = ((conv==1)*255).astype(numpy.uint8)
                conv_d = cv2.dilate(conv, kernel=kernel, iterations=1)
                conv_d_rot = tools_image.rotate_image(conv_d, -angle)

                local_responce[i]|=conv_d_rot

        norm = 255 * binarized.shape[0] * binarized.shape[1] / 100
        mask = numpy.zeros_like(local_responce[0])
        image_result = numpy.zeros((padded.shape[0],padded.shape[1],3),dtype=numpy.uint8)
        for i,W in enumerate(self.range_length):

            local_responce[i] ^= mask
            mask |= local_responce[i]
            image_result[local_responce[i] > 0] = colors[len(self.range_length) - i - 1]
            granules[str(W)]=str(int(local_responce.sum()/norm))

        return granules, image_result
# ----------------------------------------------------------------------------------------------------------------------
    def process_file_granules(self, filename_in, do_debug=False):
        base_name = filename_in.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_in)
        if self.roi_top is not None:
            image = tools_image.crop_image(image, self.roi_top, self.roi_left, self.roi_bottom, self.roi_right)

        time_start = time.time()
        gray = cv2.blur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (11, 11))
        binarized = self.Ske.binarize(gray)

        histo, image_result = self.detect_slices(binarized, base_name, do_debug)
        cv2.imwrite(self.folder_out + base_name + '.jpg', image_result)
        self.save_histo(histo,base_name)

        if do_debug:
            cv2.imwrite(self.folder_out + base_name + '1-gray.png', gray)
            cv2.imwrite(self.folder_out + base_name + '2-gray_bin.png', binarized)

        print('%s - %1.2f sec' % (base_name, (time.time() - time_start)))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_histo(self,histo,base_name):

        with open(self.folder_out+base_name+'.xml', 'w', encoding='utf-8') as f:
            json.dump(histo, f, ensure_ascii=False, indent=4)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_folder_json(self, folder_in,folder_out):

        filenames = tools_IO.get_filenames(folder_in, '*.xml')

        S = []

        for filename_in in filenames:
            with open(folder_in+filename_in, 'r', encoding='utf-8') as f:data = json.load(f)
            s = []
            for rg in reversed(self.range_length):
                if str(rg) in data:
                    value = data[str(rg)]
                else:
                    value =numpy.nan
                s.append(value)
            S.append(s)

        S = numpy.array(S,dtype=numpy.float32)

        for i in range(1,S.shape[1]):
            S[:,i]+=S[:,i-1]

        for i in range(0, S.shape[0]):
            a = S[i,-1]
            for j in range(0, S.shape[1]):
                S[i,j]=float(S[i,j]/(a/100))

        labels = [str(w) for w in reversed(self.range_length)]
        tools_plot.plot_series(S, labels = labels, filename_out=folder_out+'s.png')

        return
# ----------------------------------------------------------------------------------------------------------------------