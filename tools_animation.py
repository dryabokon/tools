import re
import os
import fnmatch
from os import listdir
import cv2
import numpy
#from PIL import Image
import uuid
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
def folder_to_video_ffmpeg(folder_in, filename_out, mask='*.jpg', framerate=24):
    filename_out = filename_out.split('/')[-1]

    fileslist = tools_IO.get_filenames(folder_in,mask)
    prefix = fileslist[0].split('0')[0]
    start_number = int(re.sub(r"[^0-9]", "", fileslist[0]))
    pattern = prefix+'%06d'+mask[1:]

    command_make_video = 'ffmpeg -start_number %d -i %s -vcodec libx264 -framerate %d %s -y'%(start_number,pattern,framerate,filename_out)
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
def prepare_images(path_input, mask='*.png', framerate=10,stop_ms=0,duration_ms=None,resize_H=None, resize_W=None,stride=1,do_reverce=False):
    images = []
    filenames = tools_IO.get_filenames(path_input, mask)
    if duration_ms is None:
        for b in numpy.arange(0, len(filenames), stride):
            image = cv2.imread(path_input + filenames[b])
            if resize_H is not None and resize_W is not None:
                #image = tools_image.do_resize(image, (resize_W, resize_H))
                image = cv2.resize(image, (resize_W, resize_H))
            image = tools_image.desaturate(image, 0.2)
            images.append(image)
    else:
        II = numpy.linspace(0, len(filenames),int((duration_ms - stop_ms) * framerate / 1000 / (2 if do_reverce else 1)),endpoint=True).astype(numpy.int)


        II[II == len(filenames)] = len(filenames) - 1

        images_orig = []
        for b in numpy.arange(0, len(filenames), stride):
            image = cv2.imread(path_input + filenames[b])
            if resize_H is not None and resize_W is not None:
                image = tools_image.do_resize(image, (resize_W, resize_H))
            images_orig += [image]

        for i in II:
            images += [images_orig[i]]

    if do_reverce:
        images = images + images[::-1]

    if stop_ms > 0:
        images += [images[-1]] * int(stop_ms * framerate / 1000)
    return images
# ---------------------------------------------------------------------------------------------------------------------
def folder_to_animated_gif_imageio(path_input, filename_out, mask='*.png,*.jpg', framerate=10,stop_ms=0,duration_ms=None,resize_H=None, resize_W=None,stride=1,do_reverce=False):
    tools_IO.remove_file(filename_out)

    images = prepare_images(path_input, mask, framerate, stop_ms, duration_ms, resize_H, resize_W, stride, do_reverce)

    if '.gif' in filename_out:
        images = numpy.array(images, dtype=numpy.uint8)[:, :, :, [2, 1, 0]]
        imageio.mimsave(filename_out, images, 'GIF', duration=(1/framerate if duration_ms is None else duration_ms/len(images)/1000))

    else:
        tools_IO.remove_files(filename_out,create=True)
        for i,image in enumerate(images):
            cv2.imwrite(filename_out + '%04d.png' % i, image)


    return
# ---------------------------------------------------------------------------------------------------------------------
def folder_to_video(folder_in, filename_out, mask='*.jpg', framerate=24,stop_ms=0,duration_ms=None, resize_W=None, resize_H=None, stride=1,do_reverce=False):

    images = prepare_images(folder_in, mask, framerate, stop_ms, duration_ms, resize_H, resize_W, stride, do_reverce)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    resize_H, resize_W = images[0].shape[:2]
    out = cv2.VideoWriter(filename_out,fourcc, framerate, (resize_W,resize_H))


    for image in images:
        out.write(image)
    out.release()

    return
# ---------------------------------------------------------------------------------------------------------------------
def re_encode(filaneme_in,filename_out):
    if os.path.isfile(filename_out):
        return

    split_in = filaneme_in.split('/')
    folder_in = '/'.join(split_in[:-1])+'/'
    split_out = filename_out.split('/')

    filename_tmp1 = uuid.uuid4().hex + '.mp4'
    filename_tmp2 = uuid.uuid4().hex + '.mp4'

    cur_dir = os.getcwd()
    os.chdir(folder_in)

    os.system('ffmpeg -i %s -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -filter:v fps=fps=30 -c:a copy %s -y' % (split_in[-1], filename_tmp1))
    os.system('ffmpeg -i %s -vf scale=1280:720 %s -y' % (filename_tmp1, filename_tmp2))
    os.system('ffmpeg -i %s -ar 44100 %s -y' % (filename_tmp2, split_out[-1]))

    tools_IO.remove_file(filename_tmp1)
    tools_IO.remove_file(filename_tmp2)
    os.chdir(cur_dir)

    return
# ---------------------------------------------------------------------------------------------------------------------
def re_encode_folder(folder_in,folder_out):
    #tools_IO.remove_files(folder_out,'*.mp4')
    filename_tmp = uuid.uuid4().hex + '.mp4'
    for filaneme_in in tools_IO.get_filenames(folder_in, '*.mp4'):
        re_encode(folder_in+filaneme_in, filename_tmp)
        tools_IO.copyfile(folder_in+filename_tmp,folder_out + filaneme_in)
        tools_IO.remove_file(folder_in+filename_tmp)


    return
# ---------------------------------------------------------------------------------------------------------------------
def folders_to_video(folder_in,filename_out):
    subfolders = tools_IO.get_sub_folder_from_folder(folder_in)
    for subfolder in subfolders:
        folder_to_video(folder_in+subfolder+'/', folder_in+subfolder+'.mp4', mask='*.jpg,*.png', framerate=24)#,resize_W=1920, resize_H=1080
    merge_videos_ffmpeg(folder_in, '*.mp4',filename_out=filename_out)

    for subfolder in subfolders:
        tools_IO.remove_file(folder_in+subfolder+'.mp4')
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
def compose_scene(folder_in,folder_out,target_W,target_H,pad_left=0,pad_top=0,pad_right=0,pad_bottom=0,img_header=None):

    col_bg = (255,255,255)
    #col_bg = (43,43,43)
    #col_bg = (192, 192, 192)
    tools_IO.remove_files(folder_out,create=True)

    target_frame_height,target_frame_width = None,None
    if pad_right is not None:
        target_frame_width = int(target_W - pad_right - pad_left)
    if pad_bottom is not None:
        target_frame_height = int(target_H - pad_top - pad_bottom)

    for filename in tools_IO.get_filenames(folder_in,'*.png,*.jpg'):
        im_frame = cv2.imread(folder_in+filename)
        #im_frame = tools_image.do_rescale(im_frame,1.5)

        image = numpy.full((target_H, target_W, 3), col_bg, dtype=numpy.uint8)

        if target_frame_height is not None or target_frame_width is not None:
            im_frame = tools_image.smart_resize(im_frame, target_frame_height, target_frame_width, bg_color=col_bg)


        image = tools_image.put_image(image,im_frame,pad_top,pad_left)
        if img_header is not None:
            image = tools_image.put_image(image, img_header, 0, 0)

        cv2.imwrite(folder_out+filename,image)


    return
# ---------------------------------------------------------------------------------------------------------------------
def fly_effetct(folder_in,folder_out,left,top,right,bottom,n_frames,effect='in'):
    col_bg = (255, 255, 255)
    tools_IO.remove_files(folder_out, create=True)
    filenames = tools_IO.get_filenames(folder_in, '*.png,*.jpg')

    image0 = cv2.imread(folder_in + filenames[0 if effect == 'in' else -1])
    im_fg = image0[top:bottom, left:right].copy()
    image0[top:bottom, left:right]=col_bg

    for frame in range(n_frames):
        delta = (right - left) * frame / (n_frames - 1)
        if effect=='in':
            col = int(right-delta)
        else:
            col = int(left - delta)
        cv2.imwrite(folder_out + '%05d.png'%frame, tools_image.put_image(image0,im_fg,top,col))


    return
# ---------------------------------------------------------------------------------------------------------------------
def merge_images_in_folders(path_input1,path_input2,path_output,mask='*.png,*.jpg',mode='V'):
    tools_IO.remove_files(path_output,list_of_masks=mask,create=True)

    fileslist1 = tools_IO.get_filenames(path_input1,mask)
    fileslist2 = tools_IO.get_filenames(path_input2,mask)

    for i,filename1 in enumerate(fileslist1):
        image1 = cv2.imread(path_input1 + filename1)
        image2 = cv2.imread(path_input2 + fileslist2[i%len(fileslist2)])

        cv2.imwrite(path_output+filename1,numpy.concatenate([image1,image2],axis=0 if mode=='V' else 1))

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
def merge_videos_ffmpeg(folder_in,mask,filename_out='_result.mp4'):
    fileslist = tools_IO.get_filenames(folder_in, mask)
    fileslist = [f for f in fileslist if f!=filename_out]
    fileslist = numpy.array(fileslist).reshape((-1,1))

    A = numpy.hstack((numpy.full((fileslist.shape[0],1),'file'),fileslist))

    filename_temp = 'tmplist.txt'
    tools_IO.save_mat(A, folder_in+filename_temp,fmt='%s',delim=' ')
    #command_make_video   = 'ffmpeg -f concat -safe 0 -i %s -map 0:v:0 %s -y'%(filename_temp,filename_out)
    #command_make_video  = 'ffmpeg -f concat -safe 0 -i %s -map 0:1 -map 0:0 %s -y'%(filename_temp,filename_out)
    command_make_video = 'ffmpeg -f concat -safe 0 -i %s -c copy %s -y'%(filename_temp,filename_out)



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
