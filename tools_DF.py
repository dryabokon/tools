import pandas as pd
import numpy
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from collections import Counter
# ----------------------------------------------------------------------------------------------------------------------
def df_to_XY(df,idx_target,keep_categoirical=True,drop_na=True,numpy_style=True):
    if drop_na:
        df = df.dropna()
    columns = df.columns.to_numpy()
    col_types = numpy.array([str(t) for t in df.dtypes])
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    if keep_categoirical==False:
        are_categoirical = numpy.array([cc in ['object', 'category', 'bool'] for cc in col_types])
        are_categoirical = numpy.delete(are_categoirical, idx_target)
        idx = idx[~are_categoirical]

    Y = df.iloc[:, [idx_target]].to_numpy().flatten()
    if numpy_style:
        X = df.iloc[:,idx].to_numpy()
    else:
        X = df.iloc[:, idx]
    return X,Y
# ---------------------------------------------------------------------------------------------------------------------
def get_names(df,idx_target,keep_categoirical=True):
    columns = df.columns.to_numpy()
    col_types = numpy.array([str(t) for t in df.dtypes])
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
    if keep_categoirical==False:
        are_categoirical = numpy.array([cc in ['object', 'category', 'bool'] for cc in col_types])
        are_categoirical = numpy.delete(are_categoirical, idx_target)
        idx = idx[~are_categoirical]

    return columns[idx]
# ----------------------------------------------------------------------------------------------------------------------
def get_categoricals_hash_map(df,drop_na=True):

    res_dict= {}

    if drop_na:
        df = df.dropna()

    are_categoirical = numpy.array([cc in ['object', 'category', 'bool'] for cc in [str(t) for t in df.dtypes]])
    columns_categorical = df.columns[are_categoirical]
    for column in columns_categorical:

        idx = df[column].isna()
        S = pd.Series(df[column]).astype(str)
        S[idx] = 'N/A'
        df[column] = S

        types = numpy.array([str(type(v)) for v in df[column]])
        values = df[column].values
        idx = numpy.argsort(types)

        values = values[idx]
        types = types[idx]
        for typ in Counter(types).keys():
            values_typ = values[types==typ].copy()
            idx = numpy.argsort(values_typ)
            values[types==typ] = values_typ[idx]

        keys = numpy.unique(values)

        dct = dict(zip(keys,numpy.arange(0,len(keys))))
        res_dict[column] = dct

    return res_dict
# ----------------------------------------------------------------------------------------------------------------------
def hash_categoricals(df,drop_na=True):

    dct_hashmap = get_categoricals_hash_map(df,drop_na)

    for column,dct in zip(dct_hashmap.keys(),dct_hashmap.values()):
        df[column] = df[column].map(dct).astype('int32')

    return df
# ----------------------------------------------------------------------------------------------------------------------
def impute_na(df,strategy='constant',strategy_bool='int'):

    #strategies = ['mean', 'median', 'most_frequent', 'interpolate', 'constant']

    imp = SimpleImputer(missing_values=numpy.nan, strategy=strategy)
    for column in df.columns:
        if column=='deck':
            ii=0
        if any([isinstance(v, bool) for v in df[column]]) and strategy_bool == 'int':
            df[column] = df[column].fillna(numpy.nan).map(dict(zip([False,numpy.nan,True], [-1,0,1]))).astype('int32')
        else:
            vvv = numpy.array([v for v in df[column].values])
            if any([isinstance(v, str) for v in df[column]]):
                idx = df[column].isna()
                S = pd.Series(df[column]).astype(str)
                S[idx] = 'N/A'
                df[column]=S
            else:
                df[column] = imp.fit_transform(vvv.reshape(-1, 1))


    return df
# ----------------------------------------------------------------------------------------------------------------------
def scale(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df2 = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
    return df2
# ----------------------------------------------------------------------------------------------------------------------
def get_conf_mat(df,idx_target):

    df = df.dropna()
    df = hash_categoricals(df)
    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
    columns = columns[idx]

    M = numpy.eye((len(columns)))
    for i in range(len(columns)):
        v1 = df[columns[i]].to_numpy()
        for j in range(len(columns)):
            v2 = df[columns[j]].to_numpy()
            M[i,j]=numpy.corrcoef(v1,v2)[0,1]

    return M
# ----------------------------------------------------------------------------------------------------------------------
def get_entropy(df,idx_target,idx_c1,idx_c2):

    df = df.dropna()
    df = hash_categoricals(df)
    columns = df.columns.to_numpy()
    Y = df[columns[idx_target]].to_numpy()
    X = df[[columns[idx_c1], columns[idx_c2]]].to_numpy()
    x_tpl = [(x[0], x[1]) for x in X]

    dct_f0 = dict(zip(x_tpl, numpy.zeros(len(x_tpl))))
    dct_f1 = dict(zip(x_tpl, numpy.zeros(len(x_tpl))))

    for x,y in zip(x_tpl,Y):
        if y<=0:
            dct_f0[x]+=1
        else:
            dct_f1[x]+=1

    P0 = numpy.array([v for v in dct_f0.values()]).astype(numpy.int)
    P0 = P0.astype(numpy.float32)/Y.shape[0]

    P1 = numpy.array([v for v in dct_f1.values()]).astype(numpy.int)
    P1 = P1.astype(numpy.float32) / Y.shape[0]


    loss = entropy(P0,P1)

    if numpy.isinf(loss):
        loss = 1

    return loss
# ----------------------------------------------------------------------------------------------------------------------
def get_Mutual_Information(df,idx_target,i1,i2):
    # df = df.dropna()
    # df = hash_categoricals(df)
    columns = df.columns
    target = columns[idx_target]

    c1, c2 = columns[i1], columns[i2]
    I = mutual_info_classif(df[[c1, c2]], df[target]).sum()

    return I
# ----------------------------------------------------------------------------------------------------------------------
def remove_dups(df):
    df = df[~df.index.duplicated(keep='first')]
    return df
# ----------------------------------------------------------------------------------------------------------------------
def remove_by_quantiles(df0,idx_target,q_left,q_right):

    df=df0.copy()

    columns = df0.columns
    idx = numpy.delete(numpy.arange(0, columns.shape[0]), idx_target)
    types = numpy.array([str(t) for t in df0.dtypes])

    for column, typ in zip(columns[idx], types[idx]):
        if typ not in ['object', 'category', 'bool']:
            idx = numpy.full(df.shape[0], True)
            ql = df[column].quantile(q_left)
            qr = df[column].quantile(q_right)
            if q_left > 0: idx = ((idx) & (df[column] >= ql))
            if q_right > 0: idx = (idx) & (df[column] <= qr)
            df.iloc[idx,df.columns.get_loc(column)] = numpy.nan

    return df
# ----------------------------------------------------------------------------------------------------------------------
def re_order_by_freq(df,idx_values=1,th=0.01,max_count=None):
    C = df.iloc[:,idx_values].value_counts().sort_index(ascending=False).sort_values(ascending=False)

    critical = sum(C.values) * th

    idx_is_good  = [C.values[i:].sum() >= critical for i in range(C.values.shape[0])]
    c_is_good = C.index.values[idx_is_good]
    if max_count is not None:
        c_is_good=c_is_good[:max_count]

    df = df[df.iloc[:,idx_values].isin(c_is_good)].copy()
    dct_map = dict((k,v)for k,v in zip(C.index.values, numpy.argsort(-C.values)))
    df['#'] = df.iloc[:,idx_values].map(dct_map)
    df_res = df.sort_values(by='#').copy().iloc[:,:-1]

    return df_res
# ----------------------------------------------------------------------------------------------------------------------
def my_agg(df,cols_groupby,cols_value,aggs,list_res_names=None,order_idx=None,ascending=True):
    dct_agg={}
    for v,a in zip(cols_value,aggs):
        if a=='top': a=(lambda x: x.value_counts().index[0] if any(x.value_counts()) else numpy.nan)

        if isinstance(a,list):
            dct_agg[v]=[item for item in a]
        else:
            dct_agg[v] = a

    df_res = df.groupby(cols_groupby).agg(dct_agg)
    df_res = df_res.reset_index()

    df_res.columns = [''.join(col) for col in df_res.columns]

    if list_res_names is not None:
        df_res = df_res.rename(columns=dict(zip(df_res.columns[-len(list_res_names):],list_res_names)))
    if order_idx is not None:
        df_res = df_res.sort_values(by=df_res.columns[order_idx], ascending=ascending)

    return df_res
# ---------------------------------------------------------------------------------------------------------------------
