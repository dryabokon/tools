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
    command_make_palette = 'ffmpeg -i %s%s%s -vf "palettegen" -y palette.png'% (prefix,'%06d',mask)
    command_make_gif =  'ffmpeg -framerate %d -i %s%s%s -lavfi "scale=%d:%d:dither=none" -i palette.png %s -y' % (framerate,prefix,'%06d',mask,resize_W,resize_H,filename_out)

    cur_dir = os.getcwd()
    os.chdir(path_input)
    os.system(command_make_palette)
    os.system(command_make_gif)
    os.system('del palette.png')
    os.chdir(cur_dir)
    tools_IO.remove_file('%s%s' % (path_out, filename_out))
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
def folder_to_animated_gif_imageio(path_input, filename_out, mask='*.png', framerate=10,resize_H=64, resize_W=64,do_reverce=False):
    tools_IO.remove_file(filename_out)
    images, labels, filenames = tools_IO.load_aligned_images_from_folder(path_input, '-', mask=mask,resize_W=resize_W,resize_H=resize_H)

    for i in range(0,images.shape[0]):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        #images[i] = tools_image.desaturate(images[i],level=0.7)

    if not do_reverce:
        imageio.mimsave(filename_out, images, 'GIF', duration=1/framerate)
    else:
        images_all = []
        for image in images:images_all.append(image)
        for image in reversed(images): images_all.append(image)

        images_all = numpy.array(images_all,dtype=images.dtype)

        imageio.mimsave(filename_out, images_all, 'GIF', duration=1 / framerate)


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
def merge_images_in_folders(path_input1,path_input2,path_output,mask='*.jpg'):
    tools_IO.remove_files(path_output,create=True)

    fileslist1 = fnmatch.filter(listdir(path_input1), mask)
    fileslist2 = fnmatch.filter(listdir(path_input2), mask)

    fileslist1.sort()
    fileslist2.sort()

    for filename1,filename2 in zip(fileslist1,fileslist2):
        image1 = cv2.imread(path_input1 + filename1)
        image2 = cv2.imread(path_input2 + filename2)
        if image1 is None or image2 is None: continue

        shape= image1.shape
        image = numpy.zeros((shape[0],shape[1]*2,shape[2]),dtype=numpy.uint8)
        image[:,:shape[1]] = image1
        image[:,shape[1]:] = image2
        cv2.imwrite(path_output+filename1,image)

    return
# ---------------------------------------------------------------------------------------------------------------------