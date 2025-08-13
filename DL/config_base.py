import json
class cnfg_base(object):

    def save(self, filename_out):
        with open(filename_out, 'w') as f:
            json.dump({k: getattr(self, k) for k in dir(self) if not (k.startswith("__") or callable(getattr(self, k)))}, f, indent=4)

    exp_name = 'default'
    source = None

    resize_W = None
    resize_H = None
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
    detector = 'yolo'
    model_detect = None

    do_tracking = False
    tracker = 'BOXMOT'
    model_track = None
    track_lifetime = 2

    do_classification = False
    classifier = 'yolo'
    model_classify = None
    classification_confidence_th = 0.5

    do_profiling = False

    host_mlflow = None
    port_mlflow = None
    remote_storage_folder = None