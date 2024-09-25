import numpy
import psutil
# ----------------------------------------------------------------------------------------------------------------------
class IO_Profiler:
    def __init__(self, verbose=True):
        self.current_event = None
        self.current_start_read = {}
        self.current_start_write = {}

        self.dict_event_reads = {}
        self.dict_event_writes = {}
        self.dict_event_cnt = {}

        self.verbose = verbose
# ----------------------------------------------------------------------------------------------------------------------
    def tic(self, event,verbose=None):

        if event not in self.dict_event_reads:
            self.dict_event_reads[event] = 0
            self.dict_event_writes[event] = 0
            self.dict_event_cnt[event] = 0

        if self.current_event is None:
            self.current_event = event
            self.current_start_read[event]  = psutil.disk_io_counters().read_bytes
            self.current_start_write[event] = psutil.disk_io_counters().write_bytes
        else:
            delta_read = 0
            delta_write = 0
            if event in self.current_start_read:
                delta_write = psutil.disk_io_counters().write_bytes - self.current_start_write[event]
                delta_read  = psutil.disk_io_counters().read_bytes - self.current_start_read[event]

            self.dict_event_reads[self.current_event]+=delta_read
            self.dict_event_writes[self.current_event]+=delta_write
            self.dict_event_cnt[event] += 1
            self.current_event = None

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print(event,': started')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def print_IO(self,event,verbose=None):

        if event not in self.dict_event_reads:
            return
        else:
            self.dict_event_reads[event]  = psutil.disk_io_counters().read_bytes  - self.current_start_read[event]
            self.dict_event_writes[event] = psutil.disk_io_counters().write_bytes - self.current_start_write[event]

            value_read  = self.dict_event_reads[event]
            value_write = self.dict_event_writes[event]
            verbose = self.verbose if verbose is None else verbose
            if verbose:
                print(event,': R %s W %s' % (self.prettify(value_read),self.prettify(value_write)))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def stage_stats(self,filename_out):

        E = list(self.dict_event_reads.keys())
        #EE = [e for e in E if self.dict_event_cnt[e] > 0]

        with open(filename_out, 'w') as f:
            for e in E:
                f.write('%s\t%s\t%s\n'%(e,self.prettify(self.dict_event_reads[e]),self.prettify(self.dict_event_writes[e])))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def print_stats(self):
        E = list(self.dict_event_time.keys())
        V = [self.dict_event_time[e]/self.dict_event_cnt[e] for e in E if self.dict_event_cnt[e]>0]
        idx = numpy.argsort(-numpy.array(V))

        for i in idx:
            print('%2.3f\t%s'%(V[i],E[i]))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def prettify(self, value):
        if value >= 1e9:return f'{value / 1e9:.2f}G'
        elif value >= 1e6:return f'{value / 1e6:.2f}M'
        elif value >= 1e3:return f'{value / 1e3:.2f}k'
        else:return f'{value:.2f}'
# ----------------------------------------------------------------------------------------------------------------------