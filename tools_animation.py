import os
import fnmatch
from os import listdir
import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
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
        images[i] = tools_image.desaturate(images[i],level=0.7)

    imageio.mimsave(filename_out, images, 'GIF', duration=1/framerate)
    return
# ---------------------------------------------------------------------------------------------------------------------
def folder_to_video(path_input,filename_out,mask='*.jpg',resize_W=320,resize_H=240):
    fileslist = fnmatch.filter(listdir(path_input), mask)
    fileslist.sort()

    image = cv2.imread(os.path.join(path_input, fileslist[0]))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename_out,fourcc, 20.0, (resize_W,resize_H))


    for filename in fileslist:
        image = cv2.imread(os.path.join(path_input, filename))
        image = cv2.resize(image,(resize_W,resize_H),interpolation=cv2.INTER_CUBIC)
        out.write(image)

    out.release()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------------------------
def crop_images_in_folder(path_input,path_output,top, left, bottom, right,mask='*.jpg'):
    tools_IO.remove_files(path_output,create=True)

    fileslist = fnmatch.filter(listdir(path_input), mask)

    for filename in fileslist:
        image = cv2.imread(path_input+filename)
        if image is None: continue

        image = tools_image.crop_image(image,top, left, bottom, right)
        cv2.imwrite(path_output+filename,image)

    return
# ---------------------------------------------------------------------------------------------------------------------