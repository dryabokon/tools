import time
# ----------------------------------------------------------------------------------------------------------------------
class tools_HB(object):
    def __init__(self):
        self.heartbeat0 = time.time()
        self.frame_ids = [0]
        self.heartbeat_times = [self.heartbeat0]
        self.fps = 0.0
        return
# ----------------------------------------------------------------------------------------------------------------------
    def do_heartbeat(self):
        self.frame_ids.append(self.frame_ids[-1]+1)
        self.heartbeat_times.append(time.time())
        self.fps = (self.frame_ids[-1] - self.frame_ids[0]) / (self.heartbeat_times[-1] - self.heartbeat_times[0] + 1e-4)
        if len(self.frame_ids) > 20:
            self.frame_ids.pop(0)
            self.heartbeat_times.pop(0)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_frame_id(self):return self.frame_ids[-1]
    def get_fps(self):return self.fps
    def get_delta_time(self):
        delta = time.time()-self.heartbeat0
        return delta
# ----------------------------------------------------------------------------------------------------------------------
