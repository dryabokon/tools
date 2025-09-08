import cv2
import pandas as pd
import json
import requests
import base64
# ----------------------------------------------------------------------------------------------------------------------
class DetectorFireFly:
    def __init__(self,ip_address,port):
        self.ip_address = ip_address
        self.port = port
        self.API = f"http://{self.ip_address}:{self.port}"
        return
# ----------------------------------------------------------------------------------------------------------------------
    def set_video_source(self,use_mipi):
        response = requests.post(self.API + '/set_video_source', json={"source": "camera 0" if use_mipi else "camera 3"},headers={'Content-Type': 'application/json'})
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def start_processing(self):
        response = requests.post(self.API + '/start_processing', headers={'Content-Type': 'application/json'})
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def stop_processing(self):
        response = requests.post(self.API + '/stop_processing', headers={'Content-Type': 'application/json'})
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def warmup(self,img):
        cnt = 50
        ready = False

        while not ready:
            response = requests.post(self.API + '/get_detections',data=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'))
            dct_res = json.loads(response.content.decode())
            df = pd.DataFrame(dct_res.get('detections', []))
            if df.empty:
                cnt -= 1
            else:
                ready = True

        return ready
    # ----------------------------------------------------------------------------------------------------------------------
    def get_detections(self,filename_in,do_debug=False):
        df_pred = pd.DataFrame({'class_ids':[],'class_name':[],'track_id': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'conf': []})

        image = cv2.imread(filename_in) if isinstance(filename_in,str) else filename_in
        response = requests.post(self.API + '/get_detections',data=base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8'))
        dct_res = json.loads(response.content.decode())
        if dct_res:
            df = pd.DataFrame(dct_res.get('detections', []))
            if not df_pred.empty:
                i=0

            # encoded_bytes = dct_res.get('image', None)
            # if encoded_bytes is not None:
            #     frame = tools_image.decode_base64(encoded_bytes.encode())

        #df_pred = pd.DataFrame(numpy.concatenate((class_ids.reshape((-1, 1)), rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['class_ids', 'x1', 'y1', 'x2', 'y2', 'conf'])
        #df_pred = df_pred.astype({'class_ids': int, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'conf': float})
        #df_pred['class_name'] = class_names
        return df_pred
    # ----------------------------------------------------------------------------------------------------------------------
