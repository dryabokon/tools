import cv2
import time
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
def extract_frames(filename_in,folder_out):
    vidcap = cv2.VideoCapture(filename_in)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(folder_out+'frame%06d.jpg' % count, image)
        success, image = vidcap.read()
        count += 1

    return
# ----------------------------------------------------------------------------------------------------------------------