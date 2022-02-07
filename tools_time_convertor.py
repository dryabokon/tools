import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
def str_to_datetime(str_series,format='%Y-%m-%d',errors='ignore'):
    res = pd.to_datetime(str_series, format='%Y-%m-%d', errors='ignore')
    return res
# ----------------------------------------------------------------------------------------------------------------------
def datetime_to_str(dt,format='%Y-%m-%d'):
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