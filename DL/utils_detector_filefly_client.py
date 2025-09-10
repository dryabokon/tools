import numpy
import cv2
import pandas as pd
import json
import requests
import base64
# ----------------------------------------------------------------------------------------------------------------------
class DetectorFireFly:
    def __init__(self,ip_address,port,source=None):
        self.ip_address = ip_address
        self.port = port
        self.source = source
        self.API = f"http://{self.ip_address}:{self.port}"
        self.stop_processing()
        self.set_video_source()
        self.start_processing()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def set_video_source(self):
        if self.source is not None:
            str_source = "camera " + self.source if self.source in ['0','1','2','3','4'] else self.source
            response = requests.post(self.API + '/set_video_source', json={"source": str_source},headers={'Content-Type': 'application/json'})
            print('set_video_source')
            print(str_source)
            print(response.content.decode())

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def start_processing(self):
        response = requests.post(self.API + '/start_processing', headers={'Content-Type': 'application/json'})
        print('start_processing')
        print(response.content.decode())
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def stop_processing(self):
        response = requests.post(self.API + '/stop_processing', headers={'Content-Type': 'application/json'})
        print('stop_processing')
        print(response.content.decode())
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
        df_pred = pd.DataFrame({'class_ids':[],'class_name':[],'x1': [], 'y1': [], 'x2': [], 'y2': [], 'conf': []})

        image = cv2.imread(filename_in) if isinstance(filename_in,str) else filename_in
        response = requests.post(self.API + '/get_detections',data=base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8'))
        dct_res = json.loads(response.content.decode())
        if dct_res:
            df_pred_firefly = pd.DataFrame(dct_res.get('detections', []))
            if not df_pred_firefly.empty:
                df_pred = pd.DataFrame({
                    'class_ids': df_pred_firefly['class'].values,
                    'class_name':['xxx']*df_pred_firefly.shape[0],
                    'x1': (df_pred_firefly['x'] ).values.astype(int),
                    'y1': (df_pred_firefly['y'] ).values.astype(int),
                    'x2': (df_pred_firefly['x'] + df_pred_firefly['width'] ).values.astype(int),
                    'y2': (df_pred_firefly['y'] + df_pred_firefly['height']).values.astype(int),
                    'conf': df_pred_firefly['confidence'].values
                })
        return df_pred
    # ----------------------------------------------------------------------------------------------------------------------
    def get_frame_and_pred(self):
        df_pred = pd.DataFrame({'class_ids':[],'class_name':[],'x1': [], 'y1': [], 'x2': [], 'y2': [], 'conf': []})
        image = None
        response = requests.get(self.API + '/get_detections_native')
        dct_res = json.loads(response.content.decode())
        if dct_res:
            df_pred_firefly = pd.DataFrame(dct_res.get('detections', []))
            if not df_pred_firefly.empty:
                df_pred = pd.DataFrame({
                    'class_ids': df_pred_firefly['class'].values,
                    'class_name': ['xxx'] * df_pred_firefly.shape[0],
                    'x1': (df_pred_firefly['x']).values.astype(int),
                    'y1': (df_pred_firefly['y']).values.astype(int),
                    'x2': (df_pred_firefly['x'] + df_pred_firefly['width']).values.astype(int),
                    'y2': (df_pred_firefly['y'] + df_pred_firefly['height']).values.astype(int),
                    'conf': df_pred_firefly['confidence'].values
                })
            'class_ids', 'class_name'
            encoded_bytes = dct_res.get('image', None)
            if encoded_bytes is not None:
                decoded_bytes = base64.b64decode(encoded_bytes)
                image = cv2.imdecode(numpy.frombuffer(decoded_bytes, numpy.uint8), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, df_pred
    # ----------------------------------------------------------------------------------------------------------------------