import os
import numpy
import time
#----------------------------------------------------------------------------------------------------------------------
class Time_Profiler():
    
    def __init__(self):
        self.current_event = None
        self.dict_event_time = {}
        self.dict_event_cnt  = {}
        self.cnt=0
        return
# ----------------------------------------------------------------------------------------------------------------------
    def tic(self, event,reset=False):

        if reset:
            if event in self.dict_event_time:
                del self.dict_event_time[event]
            if event in self.dict_event_cnt:
                del self.dict_event_cnt[event]


        if event not in self.dict_event_time:
            self.dict_event_time[event] = 0
            self.dict_event_cnt[event] = 0

        if self.current_event is not None:
            self.dict_event_time[self.current_event]+= time.time() - self.current_start
            self.dict_event_cnt[event] += 1

        self.current_event = event
        self.current_start = time.time()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def stage_stats(self,filename_out):

        E = list(self.dict_event_time.keys())
        V = [self.dict_event_time[e]/self.dict_event_cnt[e] for e in E if self.dict_event_cnt[e]>0]

        idx = numpy.argsort(-numpy.array(V))

        f_handle = os.open(filename_out, os.O_RDWR | os.O_CREAT)

        for i in idx:
            value = ('%2.2f\t%s\n'%(V[i],E[i])).encode()
            os.write(f_handle, value)

        os.close(f_handle)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def print_duration(self,event):

        if event not in self.dict_event_time:
            return
        else:
            self.dict_event_time[event] = time.time() - self.current_start
            value = '%2.2f sec' % (self.dict_event_time[event])
            print(value,'-',event)
        return
# ----------------------------------------------------------------------------------------------------------------------
