import pandas as pd
import cv2
import os
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Detector_Gaussian_Mixture:
    def __init__(self,folder_out,config=None):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)
        n_upper = 50
        n = 0
        self.ksize = 29
        self.kernel = numpy.ones((self.ksize, self.ksize))
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update_config(self, config):
        self.config = config
        return
    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    def get_detections(self,origin_frame):
        frame = cv2.blur(origin_frame, (self.ksize, self.ksize))

        cluster_img, gp, gi = rge.fit_predict_mask(frame, n, slice_factor=SLICE_FACTOR)
        proba_image = (gp * 255).astype('uint8')

        new_img = (cluster_img * 255 / (M - 1)).astype('uint8')
        new_img = cv2.resize(new_img, frame.shape[:-1][::-1])
        bboxes = get_boxes(cluster_img)

        rects = [[0,0,10,10]]
        df_det_frame = pd.DataFrame(rects, columns=['x1', 'y1', 'x2', 'y2']) if len(rects) > 0 else pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
        df_det_frame['conf'] = 1.0
        return df_det_frame
    # ----------------------------------------------------------------------------------------------------------------------