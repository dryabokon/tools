import json
import cv2
import numpy
import requests
import base64
# ----------------------------------------------------------------------------------------------------------------------
class DetectorFireFly:
    def __init__(self,ip_address,port):
        self.ip_address = ip_address
        self.port = port
        self.url = f'http://{self.ip_address}:{self.port}/get_detections'
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_detections(self,filename_in,do_debug=False):
        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in
        image_base64 = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
        response = requests.post(self.url, headers={'Content-Type': 'application/json'}, data=json.dumps({'data': image_base64}))
        rects = numpy.array(response.json()['detections'])
        return rects
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    D = DetectorFireFly('127.0.0.1',8508)
    image = numpy.full((480, 640, 3), 64, numpy.uint8)
    image[100:170, 100:150] = [0, 0, 255]
    image[200:220, 200:230] = [0, 0, 255]
    res = D.get_detections(image)
    print(res)
