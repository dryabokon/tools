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
    do_detection = False
    detection_model = 'yolov8n.pt'
    detection_model_desc = 'default model'
    confidence_th = None

