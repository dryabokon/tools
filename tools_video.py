import cv2
import time
import tools_IO
from pytube import YouTube
#--------------------------------------------------------------------------------------------------------------------------
def capture_image_to_disk(out_filename):

    cap = cv2.VideoCapture(0)

    while (True):

        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(out_filename,frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return
# ----------------------------------------------------------------------------------------------------------------------
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
def extract_frames(filename_in,folder_out,prefix='',start_time_sec=0,end_time_sec=1000000):

    tools_IO.remove_files(folder_out,create=True)

    vidcap = cv2.VideoCapture(filename_in)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    #end_time = vidcap.get(cv2.CAP_PROP_POS_MSEC)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec*1000)

    success, image = vidcap.read()
    count = 1
    while success:
        #image = numpy.transpose(image,(1, 0, 2))
        cv2.imwrite(folder_out+prefix+'%05d.jpg' % count, image)
        success, image = vidcap.read()
        current_time = vidcap.get(cv2.CAP_PROP_POS_MSEC)

        if current_time > 1000*end_time_sec: success = False
        count += 1

    return
# ----------------------------------------------------------------------------------------------------------------------
def grab_youtube_video(URL,out_path, out_filename):


    yt = YouTube(URL)
    streams = yt.streams
    stream = streams[1]

    stream.download(out_path,out_filename)
    return
# ----------------------------------------------------------------------------------------------------------------------





















