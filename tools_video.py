import os
import cv2
import time
import numpy
import pandas as pd
from pytube import YouTube
from PIL import Image
from scenedetect import detect, ContentDetector
from os import listdir
import fnmatch
# ----------------------------------------------------------------------------------------------------------------------
def do_rescale(image,scale,anti_aliasing=True,multichannel=False):
    pImage = Image.fromarray(image)
    resized = pImage.resize((int(image.shape[1]*scale),int(image.shape[0]*scale)),resample=Image.BICUBIC)
    result = numpy.array(resized)
    return result
# --------------------------------------------------------------------------------------------------------------------
def rescale_overwrite_images(folder_in_out,mask='*.jpg',target_width=320):

    local_filenames= fnmatch.filter(listdir(folder_in_out), mask)

    for local_filename in local_filenames:
        image = cv2.imread(folder_in_out+local_filename)
        scale = target_width/image.shape[1]
        image = do_rescale(image, scale)
        cv2.imwrite(folder_in_out+local_filename,image)

    return
# --------------------------------------------------------------------------------------------------------------------
def capture_video(source, filename_out,fps=20):
    cap = cv2.VideoCapture(source)
    success, frame = cap.read()
    out_shape = (frame.shape[1], frame.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')#fourcc = cv2.VideoWriter_fourcc(*'XVID')


    out = cv2.VideoWriter(filename_out,fourcc,fps, out_shape)

    cnt, start_time, = 0, time.time()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            #frame = cv2.flip(frame,1)
            out.write(frame)
            cv2.imshow('frame',frame)
            cnt+=1

            key = cv2.waitKey(1)
            if key & 0xFF == 27:break
            if key & 0xFF == ord('q'):break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return
# ----------------------------------------------------------------------------------------------------------------------
def reconvert_video(filename_in,filename_out):

    vidcap = cv2.VideoCapture(filename_in)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success and count==0:
            out = cv2.VideoWriter(filename_out, fourcc, 20.0, (image.shape[1], image.shape[0]))
        if success:
            out.write(image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
    return
# ----------------------------------------------------------------------------------------------------------------------
def extract_frames(filename_in,folder_out,prefix='',start_time_sec=0,end_time_sec=None,stride=1,scale=1):

    if not os.path.exists(folder_out):
            os.mkdir(folder_out)

    #tools_IO.remove_files(folder_out,create=True)
    vidcap = cv2.VideoCapture(filename_in)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    end_time = vidcap.get(cv2.CAP_PROP_POS_MSEC)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec*1000)

    success, image = vidcap.read()
    if success and scale!=1:image = do_rescale(image,scale)

    count = 1

    while success:

        cv2.imwrite(folder_out+prefix+'%05d.jpg' % count, image)
        success, image = vidcap.read()
        if success and scale != 1: image = do_rescale(image, scale)
        if end_time_sec is not None and end_time_sec<1000000:
            current_time = vidcap.get(cv2.CAP_PROP_POS_MSEC)
            if current_time > 1000*end_time_sec: success = False
        count += 1

    return
# ----------------------------------------------------------------------------------------------------------------------
def extract_frames_v2(filename_in,folder_out,prefix='',start_frame=0, end_frame=None,step=1,scale=1):

    if not os.path.exists(folder_out):
            os.mkdir(folder_out)

    if ('jpg' in filename_in) or ('png' in filename_in):
        cv2.imwrite(folder_out + prefix + '%05d.jpg' % 0, cv2.imread(filename_in))
        return

    vidcap = cv2.VideoCapture(filename_in)
    success, image = vidcap.read()
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames==0:
        print('No frames found in %s'%filename_in)
        return

    cnt = start_frame

    while success:

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, cnt)
        success, image = vidcap.read()
        if success and scale != 1:
            image = do_rescale(image, scale)
        if not success:continue
        cv2.imwrite(folder_out + prefix + '%05d.jpg'%cnt, image)
        success, image = vidcap.read()
        cnt+=step
        if end_frame is not None and cnt >= end_frame:
            success = False

    return
# ----------------------------------------------------------------------------------------------------------------------
def extract_frames_v3(filename_in,folder_out,frame_IDs,prefix='',scale=1):
    if not os.path.exists(folder_out):
            os.mkdir(folder_out)

    vidcap = cv2.VideoCapture(filename_in)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames==0:
        print('No frames found in %s'%filename_in)
        return

    for cnt in frame_IDs:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, cnt)
        success, image = vidcap.read()
        if not success: continue
        if scale != 1:
            image = do_rescale(image, scale)

        cv2.imwrite(folder_out +prefix+'%06d.jpg' % cnt, image)

    return
# ----------------------------------------------------------------------------------------------------------------------
def grab_youtube_video(URL,out_path, out_filename,resolution='720p'):

    try:
        yt = YouTube(URL)
        stream_filtered = yt.streams.filter(file_extension="mp4").get_by_resolution(resolution)
        stream_filtered.download(out_path, out_filename)
        df_log = pd.DataFrame({'URL': [URL], 'title': [yt.title], 'description': [yt.description], 'author': [yt.author]})
    except:
        df_log = pd.DataFrame({'URL':[],'title':[],'description':[],'author':[]})

    return df_log
# ----------------------------------------------------------------------------------------------------------------------
