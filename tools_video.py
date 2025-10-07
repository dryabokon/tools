import os
import cv2
import time
import numpy
import pandas as pd
from PIL import Image
from os import listdir
import fnmatch
from tqdm import tqdm
#from pytube import YouTube
#import yt_dlp
#import av
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
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
    cap = cv2.VideoCapture(source,cv2.CAP_DSHOW)

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
def extract_frames(filename_in,folder_out,prefix='',start_time_sec=0,end_time_sec=None,stride=1,scale=1,rect=None):

    if not os.path.exists(folder_out):
            os.mkdir(folder_out)

    vidcap = cv2.VideoCapture(filename_in)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec*1000)

    success, image = vidcap.read()
    if success and scale!=1:image = do_rescale(image,scale)
    if rect is not None and image is not None:image = image[rect[1]:rect[3], rect[0]:rect[2]]
    count = 0

    with tqdm(total=total_frames) as pbar:
        while success:
            pbar.update(1)
            cv2.imwrite(folder_out+prefix+'%06d.jpg' % count, image)
            try:
                success, image = vidcap.read()
            except:
                pass

            if success and scale != 1:image = do_rescale(image, scale)
            if rect is not None and image is not None:image = image[rect[1]:rect[3],rect[0]:rect[2]]
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
        cv2.imwrite(folder_out + prefix + '%05d.png'%cnt, image)
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
def extract_frames_ffmpeg(filename_in,folder_out,prefix='',start_time_sec=0,end_time_sec=None,stride=1,scale=1):

    if not os.path.exists(folder_out):
            os.mkdir(folder_out)
    tools_IO.remove_files(folder_out,create=True)

    hwaccel = {'hwaccel': 'cuda'}

    container = av.open(filename_in,options=hwaccel)
    cnt = 1

    for frame in tqdm(container.decode(container.streams.video[0]),total=container.streams.video[0].frames):
        frame.to_image().save(folder_out + prefix + '%06d.jpg' % cnt)
        cnt+=1

    return
# ----------------------------------------------------------------------------------------------------------------------
def grab_youtube_video(URL,out_path, out_filename):

    try:
        yt = YouTube(URL)
        stream_filtered = yt.streams.filter(file_extension="mp4")
        resolution = sorted([s.resolution for s in stream_filtered.fmt_streams if s.resolution is not None and s.resolution[-1] == 'p'])[-1]
        stream_filtered = stream_filtered.get_by_resolution(resolution)
        stream_filtered.download(out_path, out_filename)
        df_log = pd.DataFrame({'URL': [URL], 'title': [yt.title], 'description': [yt.description], 'author': [yt.author]})
    except:
        df_log = pd.DataFrame({'URL':[],'title':[],'description':[],'author':[]})

    return df_log
# ----------------------------------------------------------------------------------------------------------------------
# def grab_youtube_stream0(source,folder_out,total_frames = 1000):
#     tools_IO.remove_files(folder_out,'*.jpg')
#
#     with yt_dlp.YoutubeDL({'format': 'bestvideo[ext=mp4]', 'noplaylist': True, 'quiet': True, 'simulate': True}) as ydl:
#         cap = cv2.VideoCapture(ydl.extract_info(source, download=False)['url'])
#
#
#     for i in tqdm(range(total_frames), total=total_frames):
#         ret, image = cap.read()
#         cv2.imwrite(folder_out + 'frame_%06d.jpg' % i, image)
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     import tools_animation
#     tools_animation.folder_to_video(folder_out, folder_out+source.split('?v=')[-1]+'.mp4')
#
#     return
# ----------------------------------------------------------------------------------------------------------------------
def grab_youtube_stream(source,filename_out,total_frames = 1000):
    # youtube_dl_options = {
    #     "format": "best[height=720]",  # This will select the specific resolution typed here
    #     "outtmpl": "%(title)s-%(id)s.%(ext)s",
    #     "restrictfilenames": True,
    #     "nooverwrites": True,
    #     "writedescription": True,
    #     "writeinfojson": True,
    #     "writeannotations": True,
    #     "writethumbnail": True,
    #     "writeautomaticsub": True
    # }

    youtube_dl_options = {'format': 'bestvideo[ext=mp4]', 'noplaylist': True, 'quiet': True, 'simulate': True}

    with yt_dlp.YoutubeDL(youtube_dl_options) as ydl:
        cap = cv2.VideoCapture(ydl.extract_info(source, download=False)['url'])

    ret, image = cap.read()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    resize_H, resize_W = image.shape[:2]
    framerate = 24
    out = cv2.VideoWriter(filename_out, fourcc, framerate, (resize_W, resize_H))

    for i in tqdm(range(total_frames), total=total_frames):
        ret, image = cap.read()
        out.write(image)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    return
# ----------------------------------------------------------------------------------------------------------------------