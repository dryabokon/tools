import os
import fnmatch
from os import listdir
import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import imageio
# ---------------------------------------------------------------------------------------------------------------------
def folder_to_animated_gif_ffmpeg(path_input, path_out, filename_out, mask='.png', framerate=10,resize_H=64, resize_W=64):

    fileslist = fnmatch.filter(listdir(path_input), '*'+mask)

    prefix = fileslist[0].split('0')[0]
    command_make_palette = 'ffmpeg -i %s%s%s -vf "palettegen" -y palette.png'% (prefix,'%04d',mask)
    command_make_gif =  'ffmpeg -framerate %d -i %s%s%s -lavfi "scale=%d:%d:flags=bitexact" -i palette.png %s -y' % (framerate,prefix,'%04d',mask,resize_W,resize_H,filename_out)

    cur_dir = os.getcwd()
    os.chdir(path_input)
    os.system(command_make_palette)
    os.system(command_make_gif)
    os.system('del palette.png')
    os.chdir(cur_dir)
    os.rename('%s%s'%(path_input,filename_out), '%s%s'%(path_out,filename_out))

    return
# ---------------------------------------------------------------------------------------------------------------------
def folders_to_animated_gif_ffmpeg(path_input,path_out, mask='.png', framerate=10,resize_H=64, resize_W=64):
    tools_IO.remove_files(path_out, create=True)

    folderlist = tools_IO.get_sub_folder_from_folder(path_input)
    for subfolder in folderlist:
        filename_out = subfolder + '_animated.gif'
        folder_to_animated_gif_ffmpeg(path_input = path_input + subfolder+'/',path_out = path_out, filename_out = filename_out, mask=mask, framerate=framerate, resize_H=resize_H, resize_W=resize_W)

    return
# ---------------------------------------------------------------------------------------------------------------------
def folder_to_animated_gif_imageio(path_input, filename_out, mask='*.png', framerate=10,resize_H=64, resize_W=64):
    tools_IO.remove_file(filename_out)
    images, labels, filenames = tools_IO.load_aligned_images_from_folder(path_input, '-', mask=mask,resize_W=resize_W,resize_H=resize_H)

    for i in range(0,images.shape[0]):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

    imageio.mimsave(filename_out, images, 'GIF', duration=framerate)
    return
# ---------------------------------------------------------------------------------------------------------------------