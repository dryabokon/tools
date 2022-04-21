import os
import fnmatch
from os import listdir
import cv2
import numpy
from PIL import Image
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
def folder_to_video_ffmpeg(folder_in, filename_out, mask='*.png', framerate=24):

    fileslist = tools_IO.get_filenames(folder_in,mask)
    prefix = fileslist[0].split('0')[0]
    command_make_video =  'ffmpeg -framerate %d -i %s%s%s %s -vcodec libx264 -y ' % (framerate, prefix,'%05d',mask[1:],filename_out)

    cur_dir = os.getcwd()
    os.chdir(folder_in)
    print(command_make_video+'\n\n')
    os.system(command_make_video)
    os.chdir(cur_dir)

    return
# ---------------------------------------------------------------------------------------------------------------------
def animated_gif_uncover(filename_out,image,N,framerate=10,stop_ms=None):
    col_bg = 255
    images = []
    for i in range(N):
        tmp_image = image.copy()
        pos = int(image.shape[1]*i/N)
        tmp_image[:, pos:] = col_bg
        images += [tmp_image[:,:,[2,1,0]]]

    if stop_ms is not None:
        images+=[images[-1]]*int(stop_ms*framerate/1000)

    images = numpy.array(images,dtype=numpy.uint8)
    imageio.mimsave(filename_out, images, 'GIF', duration=1/framerate)

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
def folder_to_animated_gif_imageio(path_input, filename_out, mask='*.png', framerate=10,stop_ms=None,resize_H=None, resize_W=None,stride=1,do_reverce=False):
    tools_IO.remove_file(filename_out)

    images = []
    filenames = tools_IO.get_filenames(path_input,mask)

    for b in numpy.arange(0,len(filenames),stride):
        filename_in = filenames[b]
        image = cv2.imread(path_input+filename_in)
        if resize_H is not None and resize_W is not None:
            image = tools_image.do_resize(image,(resize_W,resize_H))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    if stop_ms is not None:
        images+=[image]*int(stop_ms*framerate/1000)

    images = numpy.array(images,dtype=numpy.uint8)

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
def folder_to_video(folder_in, filename_out, mask='*.jpg', framerate=24, resize_W=None, resize_H=None, do_reverce=False,silent=False):
    fileslist = tools_IO.get_filenames(folder_in,mask)
    fileslist.sort()

    if resize_W is None or resize_H is None:
        image = cv2.imread(os.path.join(folder_in, fileslist[0]))
        resize_H, resize_W = image.shape[:2]

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    #fourcc = cv2.VideoWriter_fourcc(*'MSVC')

    out = cv2.VideoWriter(filename_out,fourcc, framerate, (resize_W,resize_H))

    for b,filename in enumerate(fileslist):
        image = cv2.imread(os.path.join(folder_in, filename))
        if resize_W is not None and resize_H is not None:
            image = cv2.resize(image,(resize_W,resize_H),interpolation=cv2.INTER_CUBIC)
        out.write(image)

    if do_reverce:
        for filename in reversed(fileslist):
            image = cv2.imread(os.path.join(folder_in, filename))
            if resize_W is not None and resize_H is not None:
                image = cv2.resize(image, (resize_W, resize_H), interpolation=cv2.INTER_CUBIC)
            out.write(image)

    out.release()
    #cv2.destroyAllWindows()
    return
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
def merge_images_in_folders(path_input1,path_input2,path_output,mask='*.png,*.jpg',rotate_first=False):
    tools_IO.remove_files(path_output,create=True)

    fileslist1 = tools_IO.get_filenames(path_input1,mask)
    fileslist2 = tools_IO.get_filenames(path_input2,mask)

    fileslist1.sort()
    fileslist2.sort()

    for filename1,filename2 in zip(fileslist1,fileslist2):
        image1 = cv2.imread(path_input1 + filename1)
        image2 = cv2.imread(path_input2 + filename2)
        if image1 is None or image2 is None: continue

        if rotate_first:
            image1 = numpy.transpose(image1,[1,0,2])

        shape1 = image1.shape
        shape2 = image2.shape

        image2_resized = cv2.resize(image2, (int(shape1[0] * shape2[1] / shape2[0]),shape1[0]))
        image = numpy.zeros((shape1[0], shape1[1] + image2_resized.shape[1], shape1[2]), dtype=numpy.uint8)

        image[:,:shape1[1]] = image1
        image[:,shape1[1]:] = image2_resized
        cv2.imwrite(path_output+filename1,image)

    return
# ---------------------------------------------------------------------------------------------------------------------
def generate_zoom_in(filename_in,folder_out,duration=100,scale=1.05):

    image = cv2.imread(filename_in)
    H,W = image.shape[:2]

    xmax = image.shape[1]*(scale-1)
    ymax = image.shape[0]*(scale-1)

    for t in range(duration):
        x = int(xmax*t/duration)
        y = int(ymax*t/duration)
        res = image[y:H-y,x:W-x]
        res = cv2.resize(res,(W,H))
        cv2.imwrite(folder_out+'res%04d.png'%t,res)

    return
# ---------------------------------------------------------------------------------------------------------------------
def merge_all_folders(list_of_folders,folder_out,target_W,target_H,filename_watermark=None):

    if filename_watermark is not None:
        image_watermark = cv2.imread(filename_watermark)
        w = int(target_W / 5)
        h = int(w*image_watermark.shape[0]/image_watermark.shape[1])
        image_watermark = cv2.resize(image_watermark,(w,h))
    else:
        image_watermark = None

    tools_IO.remove_files(folder_out,create=True)

    pad = 0
    bg_color = (255, 255, 255)
    empty = numpy.full((target_H,target_W,3),numpy.array(bg_color,dtype=numpy.uint8),dtype=numpy.uint8)

    cnt = 0
    for folder_in in list_of_folders:

        for filename_in in tools_IO.get_filenames(folder_in, '*.jpg,*.png'):
            base_name, ext = filename_in.split('/')[-1].split('.')[0], filename_in.split('/')[-1].split('.')[1]
            image = cv2.imread(folder_in+filename_in)
            image = tools_image.smart_resize(image, target_H-2*pad, target_W-2*pad,bg_color=bg_color)
            result = tools_image.put_image(empty,image,pad,pad)
            if filename_watermark is not None:
                result = tools_image.put_image(result,image_watermark,0,result.shape[1]-image_watermark.shape[1])

            cv2.imwrite(folder_out+'res_%06d_'%cnt + base_name+'.png',result)
            cnt+=1
            print(base_name)

    return
# ---------------------------------------------------------------------------------------------------------------------
def merge_videos_ffmpeg(folder_in,mask):
    fileslist = numpy.array(tools_IO.get_filenames(folder_in,mask)).reshape((-1,1))

    A = numpy.hstack((numpy.full((fileslist.shape[0],1),'file'),fileslist))

    filename_temp = 'tmplist.txt'
    tools_IO.save_mat(A, folder_in+filename_temp,fmt='%s',delim=' ')
    command_make_video = 'ffmpeg -f concat -safe 0 -i %s -map 0:v:0 -pix_fmt yuv420p res.mp4'%filename_temp

    cur_dir = os.getcwd()
    os.chdir(folder_in)
    print(command_make_video+'\n\n')
    os.system(command_make_video)
    os.remove(filename_temp)

    os.chdir(cur_dir)

    return
# ---------------------------------------------------------------------------------------------------------------------
def to_seconds(offset):
    splits = offset.split(':')
    result = 0
    mul = 1
    for split in splits[::-1]:
        result+=int(split)*mul
        mul*=60
    return result
# ---------------------------------------------------------------------------------------------------------------------
def extract_audio(folder_in,filename_video,filename_result):

    cur_dir = os.getcwd()
    os.chdir(folder_in)

    command = 'ffmpeg ' \
    '-i {filename_video} ' \
    '-filter_complex "apad" ' \
    '{filename_result} -y'.format(filename_video=filename_video,filename_result=filename_result)
    print(command + '\n\n')
    os.system(command)

    os.chdir(cur_dir)
    return
# ---------------------------------------------------------------------------------------------------------------------
