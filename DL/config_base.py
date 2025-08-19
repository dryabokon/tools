import json
import yaml
#----------------------------------------------------------------------------------------------------------------------
class cnfg_base(object):
    def __init__(self, filename_in=None):
        if filename_in is not None:
            C = self
            with open(filename_in, 'r') as file:
                config = yaml.safe_load(file)
                for key, value in config.items(): setattr(C, key, None if str(value) == 'None' else value)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def save(self, filename_out):
        with open(filename_out, 'w') as f:
            json.dump({k: getattr(self, k) for k in dir(self) if not (k.startswith("__") or callable(getattr(self, k)))}, f, indent=4)
    # ----------------------------------------------------------------------------------------------------------------------
    exp_name = 'default'
    source = None

    resize_ratio = None
    start = 0
    limit = None
    gt = None
    iou_th = 0.25
    min_object_size = 10
    median_filter_size = 5
    median_filter_th = 50
    n_erode = 0
    n_dilate = 4

    do_detection = False
    detection_model = 'yolov8n.pt'
    confidence_th = None

    do_tracking = False
    #tracking_model = 'BOXMOT'
    tracking_model = 'DEEPSORT'
    track_lifetime = 2

    do_classification = False
    classification_model = 'yolo'
    classification_confidence_th = 0.5

    do_profiling = False

    host_mlflow = None
    port_mlflow = None
    remote_storage_folder = None