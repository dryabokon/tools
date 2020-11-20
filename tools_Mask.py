import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
import tools_filter
# ----------------------------------------------------------------------------------------------------------------
from numba.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
# ----------------------------------------------------------------------------------------------------------------
class Mask(object):
    def __init__(self,image,folder_out=None):

        self.folder_out = folder_out
        self.init_from_image(image)

        return
# ----------------------------------------------------------------------------------------------------------------
    def init_from_image(self,image_mask):

        self.image_mask = tools_image.desaturate_2d(image_mask)
        framed = numpy.pad(self.image_mask,(1, 1), 'constant', constant_values=(0, 0))

        image_edges   = framed^numpy.roll(framed,1, axis=0)
        image_corners = image_edges ^numpy.roll(image_edges, 1, axis=1)

        xx = image_corners.ravel().nonzero()[0]
        self.corners_xy = -1+numpy.array(numpy.unravel_index(xx, (image_corners.shape[0], image_corners.shape[1]))).T
        self.corners_xy[:, [0, 1]] = self.corners_xy[:, [1, 0]]
        self.ids = numpy.arange(0,len(self.corners_xy))
        self.signs = numpy.zeros(self.corners_xy.shape[0],dtype=numpy.int32)
        self.vert_n = numpy.zeros(self.corners_xy.shape[0], dtype=numpy.uint8)
        self.horz_n = [i + 1-(i%2)*2 for i in range(len(self.ids))]

        for c in range(image_mask.shape[1]):
            idx = numpy.where(self.corners_xy[:,0]==c)[0]
            if len(idx)>0:
                ids = self.ids[idx]
                self.vert_n[idx] = numpy.array([(ids[i+1],ids[i]) for i in range(0,len(ids),2)]).flatten()

        stop = False

        while stop==0:
            stop=1

            for k in range(len(self.signs)):
                if self.signs[k]!=0:continue
                x,y= self.corners_xy[k,0],self.corners_xy[k,1]
                c1 = 1*(self.image_mask[y  ,x]>0)
                c2 = 1*(self.image_mask[y-1,x]>0)
                c3 = 1*(self.image_mask[y, x-1]>0)
                c4 = 1*(self.image_mask[y-1, x - 1]>0)
                if (c1 + c2 + c3 + c4 == 3):
                    sgn = -1
                else:
                    sgn = 1
                if (c1 + c2 + c3 + c4 == 2):

                    if (c1 == c4 and c4 == 1):
                        sgn = +1
                    else:
                        sgn = -1

                self.signs[k] = sgn
                k0=-1

                while (k0!= k):
                    if (k0 == -1):
                        k0 = k
                    k0 = self.vert_n[k0]
                    sgn *= (-1)
                    self.signs[k0] = sgn
                    if (k0 % 2 == 0):
                        k0 = k0 + 1
                    else:
                        k0 = k0 - 1
                    sgn *= (-1)
                    self.signs[k0] = sgn
                stop = 0

        ll=numpy.where(image_mask < 128)
        self.NN = len(ll[0])

        return
# ----------------------------------------------------------------------------------------------------------------
    def save_debug(self):

        image_temp = cv2.resize(self.image_mask,(2*self.image_mask.shape[1],2*self.image_mask.shape[0]),interpolation=cv2.INTER_NEAREST)
        image_temp = numpy.pad(image_temp,(1, 1), 'constant', constant_values=(0, 0))
        image_temp = tools_image.desaturate(image_temp)

        image_temp = tools_draw_numpy.draw_points(image_temp,1+ 2*self.corners_xy[self.signs>0 ],color=(0,32,255),w=0)
        image_temp = tools_draw_numpy.draw_points(image_temp,1+ 2*self.corners_xy[self.signs<=0],color=(255,32,0), w=0)
        cv2.imwrite(self.folder_out + 'debug.png', image_temp)

        return
# ----------------------------------------------------------------------------------------------------------------
    def convolve(self,image2d):
        image_integral = tools_filter.integral_2d(image2d, pad=self.image_mask.shape[0])
        image_result = numpy.zeros_like(image2d)

        mxL = (self.image_mask.shape[1] - 1) // 2
        mxR =  self.image_mask.shape[1] - 1 - mxL
        myL = (self.image_mask.shape[0] - 1) // 2
        myR =  self.image_mask.shape[0] - 1 - myL

        Norm = ((mxL + mxR) * (myL + myR))

        for j in range(myL,image2d.shape[0]-myR):
            for i in range(mxL,image2d.shape[1]-mxR):
                s=0
                for xy,sign in zip(self.corners_xy,self.signs):
                    s+=image_integral[j+xy[1],i+xy[0]]*sign

                s = 2 * s + self.NN * 255 -(+image_integral[j + myR][i + mxR] - image_integral[j + myR][i - mxL] - image_integral[j - myL][i + mxR] + image_integral[j - myL][i - mxL])
                image_result[j,i] = s/Norm

        return image_result
# ----------------------------------------------------------------------------------------------------------------