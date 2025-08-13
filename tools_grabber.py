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
# ---------------------------------------------------------------------------------------------------------------------
class Grabber:
    def __init__(self, source,looped=True):
        self.looped = looped
        self.source = source
        self.should_be_closed = False
        self.HB = tools_heartbeat.tools_HB()

        self.frame_buffer_size = 500
        self.frame_buffer = []
        self.frame_ids = []
        self.last_frame_given = -1
        self._lock = Lock()
        print('Initializing Grabber with source:', self.source)
        threading.Thread(target=self.capture_frames, daemon=True).start()

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
    def capture_empty(self):
        self.exhausted = True
        with self._lock:
            self.frame_buffer.append(tools_draw_numpy.random_noise(256, 256, (128, 128, 200)))
            self.frame_ids.append(0)

        while not self.should_be_closed:
            time.sleep(0.01)
            self.HB.do_heartbeat()
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def capture_frames(self):
        if self.source is None or self.source == '':
            print('Empty source, capturing empty frames ..')
            self.capture_empty()
        else:
            self.exhausted = False
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print('Error: Could not open video source:', self.source)
                self.cap.release()
                self.capture_empty()
                return

            while not self.should_be_closed:
                time.sleep(0.01)
                if not self.exhausted:
                    self.HB.do_heartbeat()
                    ret, frame = self.cap.read()

                    if not ret:
                        self.exhausted = True
                    else:
                        with self._lock:
                            self.frame_buffer.append(frame)
                            self.frame_ids.append(self.HB.get_frame_id())

            if self.cap:
                self.cap.release()
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def get_frame(self,frame_id=None):
        if frame_id is None:
            with self._lock:
                if self.last_frame_given + 1 < len(self.frame_buffer):
                    self.last_frame_given += 1
                else:
                    if self.exhausted:
                        if self.looped:
                            self.last_frame_given = 0
                            print('Restarting ..')
                        else:
                            self.last_frame_given = len(self.frame_buffer) - 1
                    else:
                        self.last_frame_given = len(self.frame_buffer) - 1

                return self.frame_buffer[self.last_frame_given].copy()

        else:
            with self._lock:
                if frame_id in self.frame_ids:
                    index = self.frame_ids.index(frame_id)
                    return self.frame_buffer[index].copy()
                else:
                    if self.looped:
                        return self.frame_buffer[int(frame_id) % len(self.frame_buffer)].copy()
                    else:
                        return None
    # ---------------------------------------------------------------------------------------------------------------------