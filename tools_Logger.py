import os
import pandas as pd
# --------------------------------------------------------------------------------------------------------------------
import tools_time_convertor
import tools_time_profiler
# --------------------------------------------------------------------------------------------------------------------
class Logger(object):
    def __init__(self,filename_out):
        self.filename_out = filename_out
        if os.path.isfile(filename_out):
            os.remove(filename_out)
        self.T = tools_time_profiler.Time_Profiler(verbose=False)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def write(self,message):

        self.T.tic(message,reset=True)
        df = pd.DataFrame({'time': [tools_time_convertor.datetime_to_str(pd.Timestamp.now(),format='%Y-%m-%d_%H:%M:%S')], 'msg':[message]})
        if not os.path.isfile(self.filename_out):
            df.to_csv(self.filename_out, index=False,sep='\t')
        else:
            df.to_csv(self.filename_out, index=False, mode='a', header=False,sep='\t')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def write_time(self,message):
        self.write(self.T.print_duration(message))
        return
# ----------------------------------------------------------------------------------------------------------------------