import time
import numpy
# ----------------------------------------------------------------------------------------------------------------------
class Time_Profiler:
    def __init__(self, verbose=True):
        self.current_event = None
        self.current_start = {}

        self.dict_event_time = {}
        self.dict_event_cnt = {}
        self.verbose = verbose
# ----------------------------------------------------------------------------------------------------------------------
    def tic(self, event,reset=False,verbose=None):

        if reset:
            if event in self.dict_event_time:
                del self.dict_event_time[event]
            if event in self.dict_event_cnt:
                del self.dict_event_cnt[event]

        if event not in self.dict_event_time:
            self.dict_event_time[event] = 0
            self.dict_event_cnt[event] = 0


        if self.current_event is None:
            self.current_event = event
            self.current_start[event] = time.time()
        else:
            delta=0
            if event in self.current_start:
                delta = time.time() - self.current_start[event]
            self.dict_event_time[self.current_event]+=delta
            self.dict_event_cnt[event] += 1
            self.current_event = None


        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print('start', '-', event)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def print_duration(self,event,verbose=None):

        if event not in self.dict_event_time:
            return
        else:
            self.dict_event_time[event] = time.time() - self.current_start[event]
            if self.dict_event_time[event]<60:
                format = '%M:%S'
            elif self.dict_event_time[event]<60*60:
                format = '%M:%S'
            else:
                format = '%H:%M:%S'

            value = time.strftime(format, time.gmtime(self.dict_event_time[event]))
            verbose = self.verbose if verbose is None else verbose
            if verbose:
                print(value,'-',event)
        return value
# ----------------------------------------------------------------------------------------------------------------------
    def get_diration_sec(self,event):
        return time.time() - self.current_start[event]
# ----------------------------------------------------------------------------------------------------------------------
    def stage_stats(self,filename_out):

        E = list(self.dict_event_time.keys())
        V = [self.dict_event_time[e]/self.dict_event_cnt[e] for e in E if self.dict_event_cnt[e]>0]
        idx = numpy.argsort(-numpy.array(V))

        with open(filename_out, 'w') as f:
            for i in idx:
                f.write('%2.4f\t%s\n'%(V[i],E[i]))

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
    def funA(self):
        time.sleep(1.2)
        return

    def funB(self):
        time.sleep(0.3)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def test(self):
        for i in range(10):
            self.tic('A')
            self.funA()
            self.tic('A')

            self.tic('B')
            self.funB()
            self.tic('B')

        self.print_stats()
# ----------------------------------------------------------------------------------------------------------------------