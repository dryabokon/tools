import pandas as pd
import cv2
import os
import numpy
import pybgs as bgs
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Detector_BG_Sub2:
    def __init__(self,folder_out,config=None):
        self.confidence_th = config.confidence_th
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        # self.algos = [
        #     bgs.FrameDifference(), bgs.StaticFrameDifference(), bgs.WeightedMovingMean(),
        #     bgs.WeightedMovingVariance(), bgs.AdaptiveBackgroundLearning(),
        #     bgs.AdaptiveSelectiveBackgroundLearning(), bgs.MixtureOfGaussianV2(),
        #     bgs.PixelBasedAdaptiveSegmenter(), bgs.SigmaDelta(), bgs.SuBSENSE(), bgs.LOBSTER(),
        #     bgs.PAWCS(), bgs.TwoPoints(), bgs.ViBe(), bgs.CodeBook(),
        #     bgs.FuzzySugenoIntegral(), bgs.FuzzyChoquetIntegral(), bgs.LBSimpleGaussian(),
        #     bgs.LBFuzzyGaussian(), bgs.LBMixtureOfGaussians(), bgs.LBAdaptiveSOM(),
        #     bgs.LBFuzzyAdaptiveSOM(), bgs.VuMeter(), bgs.KDE(), bgs.IndependentMultimodal()
        # ]

        self.image_mask = None
        self.algorithm = bgs.ViBe()
        self.cnt = 0
        self.config = config

        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update_confidence_th(self,confidence_th):
        self.confidence_th = confidence_th
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update_config(self, config):
        self.config = config
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def mask_to_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours]
        return contours
    # ----------------------------------------------------------------------------------------------------------------------
    def contours_to_rects(self, contours, min_size):
        rects = numpy.array([cv2.boundingRect(contour) for contour in contours])
        if rects.shape[0] > 0:
            rects = rects[rects[:, 2] >= min_size]
            rects = rects[rects[:, 3] >= min_size]
            rects[:, 2] = rects[:, 0] + rects[:, 2]
            rects[:, 3] = rects[:, 1] + rects[:, 3]
        return rects
    # ----------------------------------------------------------------------------------------------------------------------
    def morph(self, image, kernel_h=3, kernel_w=3, n_dilate=1, n_erode=1):
        kernel = numpy.ones((kernel_h, kernel_w), numpy.uint8)
        result = image
        result = cv2.dilate(cv2.erode(result, kernel, iterations=n_erode), kernel, iterations=n_dilate)
        return result
    # ----------------------------------------------------------------------------------------------------------------------
    def morph_v2(self, image, kernel_h=3, kernel_w=3, n_dilate=1, n_erode=1):
        kernel = numpy.ones((kernel_h, kernel_w), numpy.uint8)
        result = image
        result = cv2.erode(cv2.dilate(result, kernel, iterations=n_dilate), kernel, iterations=n_erode)
        return result
    # ----------------------------------------------------------------------------------------------------------------------
    def morph_v3(self, image, kernel_h=3, kernel_w=3, th=75):
        imb = cv2.blur(image,(kernel_h, kernel_w))
        imb[imb>255*th/100.0] = 255
        imb[imb<=255*th/100.0] = 0
        return imb
    # ----------------------------------------------------------------------------------------------------------------------
    def remove_false_positives(self, fgmask):
        if self.image_mask is not None:
            if self.image_mask.shape[:2] == fgmask.shape[:2]:
                fgmask = cv2.bitwise_and(fgmask, cv2.bitwise_not(self.image_mask))

        return fgmask
    # ----------------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.algorithm = bgs.ViBe()
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def get_detections_as_mask(self,image):
        fgmask = self.algorithm.apply(image)
        fgmask = self.remove_false_positives(fgmask)
        if self.config.median_filter_size > 0:
            fgmask = self.morph_v3(fgmask, kernel_h=self.config.median_filter_size, kernel_w=self.config.median_filter_size,th=self.config.median_filter_th)

        fgmask = self.morph(fgmask, kernel_h=3, kernel_w=3, n_dilate=self.config.n_dilate, n_erode=self.config.n_erode)

        return fgmask
# ----------------------------------------------------------------------------------------------------------------------
    def get_detections(self,image):
        fgmask = self.get_detections_as_mask(image)
        contours = self.mask_to_contours(fgmask)
        rects = self.contours_to_rects(contours, min_size=self.config.min_object_size)
        df_det_frame = pd.DataFrame(rects, columns=['x1', 'y1', 'x2', 'y2']) if len(rects) > 0 else pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
        df_det_frame['conf'] = 1.0
        return df_det_frame
    # ----------------------------------------------------------------------------------------------------------------------