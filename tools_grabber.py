import yaml
from yaml.loader import SafeLoader
import requests
import numpy
import subprocess
import cv2
import time
from threading import Lock
import threading
# ---------------------------------------------------------------------------------------------------------------------
import tools_heartbeat
import tools_draw_numpy
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
class Grabber:
    def __init__(self, source,looped=True):
        self.looped = looped
        self.source = source
        self.should_be_closed = False
        self.HB = tools_heartbeat.tools_HB()

        self.mode = None
        self.max_frame_id = None
        self.is_initiated = False
        self.current_frame = None
        self.frame_buffer = []
        self.frame_ids = []
        self.last_frame_given = None
        self._lock = Lock()
        threading.Thread(target=self.capture_frames, daemon=True).start()

        while not self.is_initiated:
            time.sleep(0.01)

        print('Grabber Initiated:', self.source)
        return 
    # ---------------------------------------------------------------------------------------------------------------------
    def get_cookies(self):

        with open('./cred.yaml') as file:
            cred_config = yaml.load(file, Loader=SafeLoader)
            usernames = [u for u in cred_config['credentials']['usernames']]
            passwords = [str(cred_config['credentials']['usernames'][u]['password']) for u in usernames]

        if self.source.startswith('http'):
            host = self.source.split('/')[2].split(':')[0]
            login_url = f"https://{host}:7001/rest/v1/login/sessions"
            username = usernames[-1]
            password = passwords[-1]
            session = requests.Session()
            response = session.post(login_url,json={"username": username, "password": password},verify=False)
            response.raise_for_status()
            login_data = response.json()

            return login_data

        return None
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_frames_ffmpeg(self,width,height):

        cookies = f"x-runtime-guid={self.get_cookies()['token']};"
        cmd = ["ffmpeg", "-headers", f"Cookie: {cookies}\r\n", "-user_agent", "Mozilla/5.0", "-tls_verify", "0", "-i",self.source, "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

        while not self.should_be_closed:
            time.sleep(0.01)
            self.HB.do_heartbeat()
            raw_frame = proc.stdout.read(width * height * 3)
            if not raw_frame:
                continue

            frame = numpy.frombuffer(raw_frame, numpy.uint8).reshape((height, width, 3))

            with self._lock:
                self.frame_buffer.append(frame)
                #self.frame_buffer = self.frame_buffer[-self.frame_buffer_size:]

        #self.frame_buffer = []
        proc.terminate()
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def print_debug_info(self,image, font_size=18):
        clr_bg = (192, 192, 192) if image[:200, :200].mean() > 192 else (64, 64, 64)
        color_fg = (32, 32, 32) if image[:200, :200].mean() > 192 else (192, 192, 192)
        space = font_size + 20
        label = '%06d / %06d | %.1f sec | %.1f fps'%(self.last_frame_given,len(self.frame_buffer),self.HB.get_delta_time(), self.HB.get_fps())
        image = tools_draw_numpy.draw_text_fast(image, label, (0, space * 2), color_fg=color_fg, clr_bg=clr_bg,font_size=font_size)
        return image
    # ---------------------------------------------------------------------------------------------------------------------
    def is_endless_stream(self,source):
        if isinstance(source,int):return True
        else:
            if source.startswith('http') or source.startswith('rtsp'):return True
            if source.startswith('nvarguscamerasrc') or source.startswith('v4l2src') or source.isdigit():return True
        return False
    # ---------------------------------------------------------------------------------------------------------------------
    def is_mp4_video(self,source):
        if isinstance(source, int): return False
        if source.endswith('.mp4') or source.endswith('.avi') or source.endswith('.mov'):return True
        return False
    # ---------------------------------------------------------------------------------------------------------------------
    def open_capture(self,source):
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            #cv2.CAP_DSHOW
            return cv2.VideoCapture(int(source))
        if isinstance(source, str) and (source.startswith('nvarguscamerasrc') or source.startswith('v4l2src') or source.startswith('rtsp://') or source.startswith('rtsps://')):
            if 'appsink' not in source:
                source = source.strip() + ' ! appsink drop=1 sync=false'
            return cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        return cv2.VideoCapture(source)
    # ---------------------------------------------------------------------------------------------------------------------
    def get_max_frame_id(self):
        return self.max_frame_id
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_empty(self):
        self.mode = 'empty'
        self.max_frame_id = 1
        self.last_frame_given = 0
        with self._lock:
            self.frame_buffer = [tools_draw_numpy.random_noise(480, 640, (128, 128, 164))]
            self.frame_ids = [0]
            self.is_initiated = True

        self.exhausted = True

        while not self.should_be_closed:
            time.sleep(0.01)
            self.HB.do_heartbeat()
            with self._lock:
                self.frame_buffer[0] = tools_draw_numpy.random_noise(480, 640, (128, 128, 164))
                self.frame_ids[0] = 0

        return

    # ---------------------------------------------------------------------------------------------------------------------
    def capture_filenames(self):
        self.mode = 'filenames'
        filenames = tools_IO.get_filenames(self.source, '*.jpg,*.png')
        self.max_frame_id = len(filenames)
        self.last_frame_given = -1
        self.exhausted = False
        for i, filename in enumerate(filenames):
            self.HB.do_heartbeat()
            frame = cv2.imread(self.source+filename)
            with self._lock:
                self.frame_buffer.append(frame)
                self.frame_ids.append(i)
            self.is_initiated = True

        self.exhausted = True

        return
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_finite(self):
        self.mode = 'finite'
        self.cap = self.open_capture(self.source)
        self.max_frame_id = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.exhausted = False
        self.last_frame_given = -1
        for i in range(self.max_frame_id):
            time.sleep(0.01)
            self.HB.do_heartbeat()
            ret, frame = self.cap.read()
            with self._lock:
                self.frame_buffer.append(frame)
                self.frame_ids.append(self.HB.get_frame_id())
            self.is_initiated = True

        self.exhausted = True
        self.cap.release()
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_endless(self):
        self.mode = 'endless'
        self.cap = self.open_capture(self.source)
        self.max_frame_id = numpy.inf
        self.last_frame_given = 0
        while not self.should_be_closed:
            time.sleep(0.01)
            self.HB.do_heartbeat()
            ret, frame = self.cap.read()
            if str(self.source) in ['0', '1']:
                frame = cv2.flip(frame, 1)

            with self._lock:
                self.current_frame = frame

            self.is_initiated = True

        if self.cap:
            self.cap.release()

        return
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_frames(self):
        if   self.source is None or self.source == ''   :self.capture_empty()
        elif self.is_endless_stream(self.source)        :self.capture_endless()
        elif self.is_mp4_video(self.source)             :self.capture_finite()
        else                                            :self.capture_filenames()
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def get_frame_empty(self,frame_id=None):
        return self.frame_buffer[-1]
    # ---------------------------------------------------------------------------------------------------------------------
    def get_frame_finite(self,frame_id=None):
        if frame_id is not None:
            with self._lock:
                if frame_id in self.frame_ids:
                    index = self.frame_ids.index(frame_id)
                    result = self.frame_buffer[index].copy()
                else:
                    if self.looped:
                        result = self.frame_buffer[int(frame_id) % len(self.frame_buffer)].copy()
                    else:
                        result = None
        else:
            with self._lock:
                if self.last_frame_given + 1  < len(self.frame_buffer):
                    self.last_frame_given += 1
                    result = self.frame_buffer[self.last_frame_given].copy()
                else:
                    if self.exhausted:
                        if self.looped:
                            self.last_frame_given = 0
                            if len(self.frame_buffer) >= 2:
                                print('Restarting ..')
                        else:
                            self.last_frame_given = len(self.frame_buffer) - 1
                    else:
                        self.last_frame_given = len(self.frame_buffer) - 1

                    result = self.frame_buffer[self.last_frame_given].copy()

        return result
    # ---------------------------------------------------------------------------------------------------------------------
    def get_frame_endless(self,frame_id=None):
        if frame_id is None:
            with self._lock:
                self.last_frame_given+=1
                self.frame_ids.append(self.last_frame_given)
                self.frame_buffer.append(self.current_frame.copy())
                self.frame_buffer = self.frame_buffer[-1000:]
                self.frame_ids = self.frame_ids[-1000:]
                return self.current_frame.copy()
        else:
            with self._lock:
                if frame_id in self.frame_ids:
                    index = self.frame_ids.index(frame_id)
                    result = self.frame_buffer[index].copy()
                else:
                    H,W = self.current_frame.shape[:2]
                    result = tools_draw_numpy.random_noise(H, W, (70, 80, 64))

        return result
    # ---------------------------------------------------------------------------------------------------------------------
    def get_frame(self,frame_id=None):
        if   self.mode == 'empty':return self.get_frame_empty(frame_id)
        elif self.mode== 'filenames':return self.get_frame_finite(frame_id)
        elif self.mode == 'finite': return self.get_frame_finite(frame_id)
        elif self.mode == 'endless':return self.get_frame_endless(frame_id)
        return
    # ---------------------------------------------------------------------------------------------------------------------