import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
def str_to_datetime(str_series,format='%Y-%m-%d',errors='ignore'):
    res = pd.to_datetime(str_series, format=format, errors=errors)
    return res
# ----------------------------------------------------------------------------------------------------------------------
def datetime_to_str(dt,format='%Y-%m-%d'):
    if isinstance(dt, (pd.Series)):
        res = pd.Series(pd.DatetimeIndex(dt).strftime(format))
    else:
        res = dt.strftime(format)
    return res
# ----------------------------------------------------------------------------------------------------------------------
def add_delta(str_value,str_delta,format='%Y-%m-%d'):

    if isinstance(str_value,str):
        str_value = pd.Series([str_value])
        res = datetime_to_str(pd.to_datetime(str_value,format=format,errors='ignore').values[0] + pd.Timedelta(str_delta), format)
    else:
        res = pd.to_datetime(str_value, format=format, errors='ignore') +  pd.Timedelta(str_delta)

    return res
# ----------------------------------------------------------------------------------------------------------------------
def generate_date_range(dt_start,dt_stop,freq):

    need_postprocess_to_str = False

    if isinstance(dt_start,str):
        dt_start = str_to_datetime(dt_start)
        need_postprocess_to_str = True

    dates = []
    now = dt_stop if not isinstance(dt_stop,str) else str_to_datetime(dt_stop)
    while now>=(dt_start if not isinstance(dt_start,str) else str_to_datetime(dt_start)):
        dates.append(now)
        now= now - pd.Timedelta('1'+freq)
    dates = dates[::-1]

    if need_postprocess_to_str:
        dates = datetime_to_str(pd.Series(dates))

    return dates
# ----------------------------------------------------------------------------------------------------------------------