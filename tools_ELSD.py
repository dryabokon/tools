import os
import cv2
import numpy
import subprocess
import uuid
# ----------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
# ----------------------------------------------------------------------------------------------------------------
class ELSD(object):
    def __init__(self,folder_out):
        self.name = "ELSD"
        self.bin_name  = './../_weights/ELSD.exe'
        self.folder_out = folder_out
        return
# ----------------------------------------------------------------------------------------------------------------
    def extract_ellipses(self, image, min_size=50):

        temp_pgm = str(uuid.uuid4())
        temp_eli = str(uuid.uuid4())
        temp_poly = str(uuid.uuid4())
        temp_svg  = str(uuid.uuid4())

        cv2.imwrite(self.folder_out + temp_pgm + '.pgm',tools_image.desaturate_2d(image))
        #command = [self.bin_name_ELSD, self.folder_out + temp_pgm + '.pgm', self.folder_out + temp_eli + '.txt',self.folder_out + temp_poly + '.txt']
        command = [self.bin_name, self.folder_out + temp_pgm + '.pgm', self.folder_out + temp_eli + '.txt', self.folder_out + temp_poly + '.txt', self.folder_out + temp_svg + '.svg']
        subprocess.call(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        if os.path.isfile(self.folder_out + temp_eli + '.txt'):
            ellipses = tools_IO.load_mat(self.folder_out + temp_eli + '.txt', dtype=numpy.float, delim=' ')
            if len(ellipses.shape)==1:
                ellipses = numpy.array([ellipses])
            ellipses = ellipses[:, 1:]
            ellipses = [e for e in ellipses if numpy.linalg.norm((e[0]-e[2],e[1]-e[3]))>=min_size]
        else:
            ellipses = []

        tools_IO.remove_file(self.folder_out + temp_pgm + '.pgm')
        tools_IO.remove_file(self.folder_out + temp_eli + '.txt')
        tools_IO.remove_file(self.folder_out + temp_poly + '.txt')
        #tools_IO.remove_file(self.folder_out + temp_svg + '.svg')

        return ellipses
# ----------------------------------------------------------------------------------------------------------------

