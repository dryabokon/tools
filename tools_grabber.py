import os
import yaml
import socket
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
    def __init__(self, source,looped=True,width=None,height=None,cred_config_yaml=None):
        self.name = 'Grabber cv2'
        self.looped = looped
        self.source = source
        self.width = width
        self.height = height
        self.cred_config_yaml = cred_config_yaml
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

        with open(self.cred_config_yaml) as file:
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
    # ----------------------------------------------------------------------------------------------------------------------
    def print_debug_info(self,image, font_size=18):
        clr_bg = (192, 192, 192) if image[:200, :200].mean() > 192 else (64, 64, 64)
        color_fg = (32, 32, 32) if image[:200, :200].mean() > 192 else (192, 192, 192)
        space = font_size + 20
        label = '%06d / %06d | %.1f sec | %.1f fps'%(self.last_frame_given,len(self.frame_buffer),self.HB.get_delta_time(), self.HB.get_fps())
        image = tools_draw_numpy.draw_text_fast(image, label, (0, space * 2), color_fg=color_fg, clr_bg=clr_bg,font_size=font_size)
        return image

    # ---------------------------------------------------------------------------------------------------------------------
    def udp_ok(self,ip="0.0.0.0",port=None):
        if port is None:
            return False
        
        timeout = 0.5
        bufsize = 2048
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind((ip, port))
            s.setblocking(False)
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    data, addr = s.recvfrom(bufsize)
                    return True
                except BlockingIOError:
                    time.sleep(0.01)
            return False
        finally:
            s.close()
    # ---------------------------------------------------------------------------------------------------------------------
    def is_endless_stream_argus(self, source):
        if isinstance(source, int): return False
        if (source.startswith('nvarguscamerasrc') or source.startswith('v4l2src')):return True
        return False
    # ---------------------------------------------------------------------------------------------------------------------
    def is_endless_stream_ffmpeg(self,source):
        if isinstance(source, int): return False
        if source.startswith('udp'):return True
        return False
    # ---------------------------------------------------------------------------------------------------------------------
    def is_endless_stream_youtube(self,source):
        if isinstance(source, int): return False
        if source.startswith('https://www.youtube.com') or source.startswith('https://youtu.be'):return True
        return
        # ---------------------------------------------------------------------------------------------------------------------
    def is_endless_stream_webcam(self,source):
        if isinstance(source, str) and self.source in ['0','1','2','3']: return True
        if isinstance(source, int): return True
        return False
        # ---------------------------------------------------------------------------------------------------------------------
    def is_mp4_video(self,source):
        if isinstance(source, int): return False
        if source.endswith('.mp4') or source.endswith('.avi') or source.endswith('.mov'):return True
        return False
    # ---------------------------------------------------------------------------------------------------------------------
    def get_max_frame_id(self):
        return self.max_frame_id
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_empty(self):
        print('capture_empty')
        
        self.mode = 'empty'
        self.max_frame_id = 1
        self.last_frame_given = 0
        color = (98, 94, 76)
        with self._lock:
            self.frame_buffer = [tools_draw_numpy.random_noise(480, 640, color)]
            self.frame_ids = [0]
            self.is_initiated = True

        self.exhausted = True

        while not self.should_be_closed:
            time.sleep(0.01)
            self.HB.do_heartbeat()
            with self._lock:
                self.frame_buffer[0] = tools_draw_numpy.random_noise(480, 640, color)
                self.frame_ids[0] = 0

        return

    # ---------------------------------------------------------------------------------------------------------------------
    def capture_filenames(self):
        print('capture_filenames')
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
        print('capture_finite')
        self.mode = 'finite'
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f'Error: opening {self.source}')
            self.capture_empty()
            return

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
    def capture_endless_argus(self):
        print('capture_endless_argus')

        self.mode = 'endless'
        source = self.source
        if 'appsink' not in source:
            source = source.strip() + ' ! appsink drop=1 sync=false'
        self.cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        self.max_frame_id = numpy.inf
        self.last_frame_given = 0

        failed = 0
        while not self.should_be_closed:
            time.sleep(0.01)
            self.HB.do_heartbeat()
            ret, frame = self.cap.read()

            if frame is None:
                failed += 1
            else:
                failed = 0
                self.is_initiated = True
                with self._lock:
                    self.current_frame = frame

            if failed > 20:
                print(f'Error: failed to read from {self.source}, stopping capture.')
                if self.cap:
                    self.cap.release()
                self.capture_empty()
                return



        if self.cap:
            self.cap.release()

        return
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_endless_ffmpeg(self):
        print('capture_endless_ffmpeg')
        self.mode = 'endless'

        port = int(self.source.rsplit(':', 1)[-1]) if ':' in self.source else None
        ip = '127.0.0.1'
        
        self.cap = None
        if self.udp_ok(ip, port):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            
        if self.cap is None or (not self.cap.isOpened()):
            print(f'Error: opening {self.source}')
            self.capture_empty()
            return

        self.max_frame_id = numpy.inf
        self.last_frame_given = 0
        failed = 0
        while not self.should_be_closed:
            time.sleep(0.01)
            self.HB.do_heartbeat()
            ret, frame = self.cap.read()

            if not ret:
                failed+=1
            else:
                failed = 0
                with self._lock:
                    self.current_frame = frame

            if failed>20:
                print(f'Error: failed to read from {self.source}, stopping capture.')
                if self.cap:
                    self.cap.release()
                self.capture_empty()
                return

            self.is_initiated = True

        if self.cap:
            self.cap.release()
        return

    # ---------------------------------------------------------------------------------------------------------------------
    def capture_endless_webcam(self):
        print('capture_endless_webcam')
        self.mode = 'endless'

        if os.name in ['nt']:
            self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(int(self.source))

        if self.cap is None or (not self.cap.isOpened()):
            print(f'Error: opening {self.source}')
            self.capture_empty()
            return

        self.max_frame_id = numpy.inf
        self.last_frame_given = 0
        failed = 0
        while not self.should_be_closed:
            time.sleep(0.01)
            self.HB.do_heartbeat()
            ret, frame = self.cap.read()
            if not ret:
                failed += 1
            else:
                failed = 0
                with self._lock:
                    self.current_frame = frame

            if failed > 20:
                print(f'Error: failed to read from {self.source}, stopping capture.')
                if self.cap:
                    self.cap.release()
                self.capture_empty()
                return

            self.is_initiated = True

        if self.cap:
            self.cap.release()
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_endless_grab(self,width,height):
        print('capture_endless_grab')
        self.mode = 'endless'
        cookies = f"x-runtime-guid={self.get_cookies()['token']};"
        cmd = ["ffmpeg", "-headers", f"Cookie: {cookies}\r\n", "-user_agent", "Mozilla/5.0", "-tls_verify", "0", "-i",self.source, "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)

        self.max_frame_id = numpy.inf
        self.last_frame_given = 0
        while not self.should_be_closed:
            time.sleep(0.01)
            self.HB.do_heartbeat()
            frame = proc.stdout.read(width * height * 3)
            if not frame:continue
            frame = numpy.frombuffer(frame, numpy.uint8).reshape((height, width, 3))
            with self._lock:
                self.current_frame = frame

            self.is_initiated = True

        proc.terminate()

        return
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_endless_YT(self):
        print('capture_endless_YT')
        from yt_dlp import YoutubeDL
        self.mode = 'endless'

        target_h = 720  # 360 / 480 / 720 / 1080

        ydl_opts = {"noplaylist": True,
                    "quiet": True,
                    "format": (f"best[ext=mp4][height<={target_h}][acodec!=none]"
                        f"[vcodec!*=av01][vcodec!*=vp9]"
                        f"/bestvideo[height<={target_h}][vcodec!*=av01][vcodec!*=vp9]"
                        f"/best[height<={target_h}]"),
                    "cookiefile": r"cookies.txt",
                    "extractor_args":{"youtube": {"player_client": ["android", "web"]}}
                    }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.source, download=False)
            stream_url = info.get("url")
            self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

            if self.cap is None or (not self.cap.isOpened()):
                print(f'Error: opening {self.source}')
                self.capture_empty()
                return

        self.max_frame_id = numpy.inf
        self.last_frame_given = 0
        failed = 0
        while not self.should_be_closed:
            time.sleep(1.0/30.0)
            self.HB.do_heartbeat()
            ret, frame = self.cap.read()

            if not ret:
                failed+=1
            else:
                failed = 0
                with self._lock:
                    self.current_frame = frame

            if failed>20:
                print(f'Error: failed to read from {self.source}, stopping capture.')
                if self.cap:
                    self.cap.release()
                self.capture_empty()
                return

            self.is_initiated = True

        if self.cap:
            self.cap.release()

        return
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_frames(self):
        if   self.source is None or self.source == ''   :self.capture_empty()
        elif self.is_endless_stream_webcam(self.source) :self.capture_endless_webcam()
        elif self.is_endless_stream_argus(self.source)  :self.capture_endless_argus()
        elif self.is_endless_stream_ffmpeg(self.source) :self.capture_endless_ffmpeg()
        elif self.is_endless_stream_youtube(self.source):self.capture_endless_YT()
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