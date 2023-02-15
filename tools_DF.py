import pandas as pd
import numpy
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from collections import Counter
from tabulate import tabulate
import struct
# ----------------------------------------------------------------------------------------------------------------------
def df_to_XY(df,idx_target,numpy_style=True):

    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    Y = df.iloc[:, [idx_target]].to_numpy().flatten()
    if numpy_style:
        X = df.iloc[:,idx].to_numpy()
    else:
        X = df.iloc[:, idx]
    return X,Y
# ---------------------------------------------------------------------------------------------------------------------
def XY_to_df(X,Y):
    df = pd.concat([pd.DataFrame(Y),pd.DataFrame(X)],axis=1)
    return df
# ---------------------------------------------------------------------------------------------------------------------
def get_categoricals_hash_map(df):

    res_dict= {}

    for column,typ in zip(df.columns,df.dtypes):
        is_categoirical = str(typ) in ['object', 'category']
        is_bool = str(typ) in ['bool']
        if not is_categoirical and not is_bool:continue

        K = [k for k in Counter(df[column]).keys()]
        T1 = [(isinstance(k, float) and numpy.isnan(k)) for k in K]
        T2 = [isinstance(k, bool) for k in K]
        if all([t1 or t2 for t1,t2 in zip(T1,T2)]):
            is_bool= True
            is_categoirical = False

        if is_categoirical:
            S = pd.Series(df[column])
            idx = df[column].isna()
            S = S.astype(str)
            S[idx] = 'N/A'

            types = numpy.array([str(type(v)) for v in S])
            values = S.values
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
        elif is_bool:
            res_dict[column] = {False:-1,numpy.nan:0,True:1}

    return res_dict
# ----------------------------------------------------------------------------------------------------------------------
def hash_categoricals(df):

    dct_hashmap = get_categoricals_hash_map(df)
    df_res = df.copy()

    for column,dct in zip(dct_hashmap.keys(),dct_hashmap.values()):
        df_res[column]=df_res[column].map(dct)

    return df_res
# ----------------------------------------------------------------------------------------------------------------------
def impute_na(df,strategy='constant',strategy_bool='str'):

    if df.shape[0]==0:
        return df

    str_na = 'N/A'

    imp = SimpleImputer(missing_values=numpy.nan, strategy=strategy)
    for column in df.columns:
        if column=='pressure':
            pp=0

        if any([isinstance(v, bool) for v in df[column]]):
            if strategy_bool == 'int':
                df[column] = df[column].fillna(numpy.nan).map(dict(zip([False,numpy.nan,True], [-1,0,1]))).astype('int32')
            elif strategy_bool == 'str':
                idx_true = df[column].isin([True,'True','TRUE','true']).values
                idx_false =df[column].isin([False,'False','FALSE','false']).values
                V1 = numpy.full(df.shape[0],str_na,dtype=numpy.chararray)
                V1[idx_true] = 'true'
                V1[idx_false] = 'false'
                df[column] = V1

        else:
            if all(pd.isna(df[column]).values.tolist()):
                df[column]='N/A'
                df[column]=df[column].astype(str)
            else:
                if any([isinstance(v, str) for v in df[column]]):
                    idx = df[column].isna()
                    S = pd.Series(df[column]).astype(str)
                    S[idx] = 'N/A'
                    df[column]=S
                else:
                    vvv = numpy.array([v for v in df[column].values])
                    df[column] = imp.fit_transform(vvv.reshape(-1, 1))

    return df
# ----------------------------------------------------------------------------------------------------------------------
def remove_dups(df):
    df = df[~df.index.duplicated(keep='first')]
    return df
# ----------------------------------------------------------------------------------------------------------------------
def remove_long_tail(df,idx_target=0,th=0.01,order=False):
    idxs = numpy.delete(numpy.arange(0, df.shape[1]), idx_target)

    for idx in idxs:
        max_count = 30 if str(df.dtypes[idx]) in ['object', 'category', 'bool'] else None
        C = df.iloc[:, idx].value_counts().sort_index(ascending=False).sort_values(ascending=False)
        #c_is_good = numpy.full(df.shape[0],True)
        if max_count is not None and C.shape[0]>max_count:
            critical = sum(C.values) * th
            idx_is_good = [C.values[i:].sum() >= critical for i in range(C.values.shape[0])]
            c_is_good = C.index.values[idx_is_good]
            c_is_good = c_is_good[:max_count]

            if order and str(df.dtypes[idx]) in ['object', 'category', 'bool']:
                dct_map = dict((k, v) for k, v in zip(C.index.values, numpy.argsort(-C.values)))
                df['#'] = df.iloc[:, idx].map(dct_map)

            df.iloc[~df.iloc[:, idx].isin(c_is_good),idx]=numpy.nan

    df.dropna(inplace=True)
    if '#' in df.columns:
        df = df.sort_values(by='#').iloc[:, :-1]

    return df
# ----------------------------------------------------------------------------------------------------------------------
def my_agg(df,cols_groupby,cols_value,aggs,list_res_names=None,order_idx=None,ascending=True):
    dct_agg={}
    for v,a in zip(cols_value,aggs):
        if a=='top': a=(lambda x: x.value_counts().index[0] if any(x.value_counts()) else numpy.nan)

        if isinstance(a,list):
            dct_agg[v]=[item for item in a]
        else:
            dct_agg[v] = a

    df_res = df.groupby(cols_groupby,dropna=False).agg(dct_agg)
    df_res = df_res.reset_index()

    df_res.columns = [''.join(col) for col in df_res.columns]

    if list_res_names is not None:
        df_res = df_res.rename(columns=dict(zip(df_res.columns[-len(list_res_names):],list_res_names)))
    if order_idx is not None:
        df_res = df_res.sort_values(by=df_res.columns[order_idx], ascending=ascending)

    return df_res
# ---------------------------------------------------------------------------------------------------------------------
def add_noise_smart(df,idx_target=0,A = 0.2):

    df_res  =df.copy()

    idx = numpy.delete(numpy.arange(0, df.shape[1]), idx_target)
    df.iloc[:,idx[0]]=df.iloc[:,idx[0]].astype(float)
    df.iloc[:,idx[1]]=df.iloc[:,idx[1]].astype(float)

    uX = df.iloc[:, idx[0]].unique()
    uY = df.iloc[:, idx[1]].unique()

    if uY.shape[0]>20 or uX.shape[0]>20:
        return df_res

    N = []
    for x in uX:
        for y in uY:
            idx_pos = (df.iloc[:, idx[0]] == x) & (df.iloc[:, idx[1]] == y) & (df.iloc[:, idx_target] > 0)
            idx_neg = (df.iloc[:, idx[0]] == x) & (df.iloc[:, idx[1]] == y) & (df.iloc[:, idx_target] <= 0)
            N.append(idx_pos.sum()+idx_neg.sum())

    aspect_xy= 1 if (uY.max() == uY.min()) else (uX.max()-uX.min())/(uY.max()-uY.min())
    dpi =  numpy.sqrt(max(N))/A

    for x in uX:
        for y in uY:
            idx_pos = (df.iloc[:, idx[0]] == x) & (df.iloc[:, idx[1]] == y) & (df.iloc[:, idx_target] > 0)
            idx_neg = (df.iloc[:, idx[0]] == x) & (df.iloc[:, idx[1]] == y) & (df.iloc[:, idx_target] <= 0)
            n_pos = idx_pos.sum()
            n_neg = idx_neg.sum()
            if n_pos+n_neg==0:continue

            r = [numpy.sqrt(n)/dpi for n in range(n_pos+n_neg)]
            step_alpha = numpy.pi / 45.0
            alpha = [0]
            for rr in r[1:]:
                prev = alpha[-1]
                alpha.append(prev+step_alpha/rr)

            nx,ny=r*numpy.sin(alpha),r*numpy.cos(alpha)
            if aspect_xy>1:
                ny/=aspect_xy
            else:
                nx*=aspect_xy

            noise_x, noise_y = numpy.random.multivariate_normal([0, 0], [[max(nx)/1000, 0], [0, max(ny)/1000]], len(r)).T
            nx+=noise_x
            ny+=noise_y

            na = min(idx_pos.sum(),idx_neg.sum())
            nx_a,ny_a = nx[:na],ny[:na]
            nx_b,ny_b = nx[na:],ny[na:]

            df_res.iloc[numpy.where(idx_pos)[0],idx[0]]+=nx_a if n_pos<n_neg else nx_b
            df_res.iloc[numpy.where(idx_pos)[0],idx[1]]+=ny_a if n_pos<n_neg else ny_b
            df_res.iloc[numpy.where(idx_neg)[0],idx[0]]+=nx_b if n_pos<n_neg else nx_a
            df_res.iloc[numpy.where(idx_neg)[0],idx[1]]+=ny_b if n_pos<n_neg else ny_a

    return df_res
# ---------------------------------------------------------------------------------------------------------------------
def from_multi_column(df,idx_time):
    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_time)
    col_time = columns[idx_time]

    df_res = pd.DataFrame()

    for col in columns[idx]:
        value = df.loc[:, col]
        df_frame = pd.DataFrame({col_time:df.iloc[:, 0],'label':col,'value':value})
        df_res = pd.concat([df_res,df_frame],ignore_index=True)

    return df_res
# ---------------------------------------------------------------------------------------------------------------------
def to_multi_column(df,idx_time,idx_label,idx_value,replace_nan=True,order_by_value=False):
    columns = [c for c in df.columns]
    col_time = columns[idx_time]
    col_label = columns[idx_label]
    col_value = columns[idx_value]

    #df.to_csv('xxx.csv',index = False)
    df.drop_duplicates(inplace=True)

    df_res = df.pivot(index=col_time, columns=col_label)[col_value]
    if replace_nan:
        df_res.replace({numpy.NaN: 0}, inplace=True)
    if order_by_value:
        cols = numpy.array([c for c in df_res.columns])
        values = df_res.sum(axis=0).values
        idx = numpy.argsort(-values)
        df_res = df_res[cols[idx]]

        values = df_res.sum(axis=1).values
        idx = numpy.argsort(-values)
        df_res=df_res.iloc[idx,:]

    df_res.reset_index(level=0, inplace=True)

    dct_new_columns = dict(zip(df_res.columns,[str(c) for c in df_res.columns]))
    df_res = df_res.rename(columns=dct_new_columns)

    return df_res
# ---------------------------------------------------------------------------------------------------------------------
def preprocess(df,dct_methods,skip_first_col=False,do_binning=False):

    if df.shape[0]==0:
        return df

    pt = preprocessing.PowerTransformer()
    dct_rename={}
    q = [0, .2, .4, .6, .8, 1]

    for c in range(df.shape[1]):
        if skip_first_col and c==0:continue
        col = df.columns[c]

        #try convert to numeric
        S = [str(u) for u in df[col].unique()]
        U = [s[1:] if (len(s)>0 and s[0]=='-') else s for s in S]
        U = [u.split('e+')[0] for u in U]
        U = [u.split('e-')[0] for u in U]
        I4 = [str(u).replace('.', '', 1).isdigit() for u in U]
        if all(I4):
            df[col] = df[col].astype(float)
            II = [u.is_integer() for u in df[col].unique()]
            if all(II): df[col] = df[col].astype(int)
            if do_binning and df[col].unique().shape[0]>50:
                U, Q = pd.qcut(df[col].rank(method='first'), q=q, retbins=True)
                df[col] = U.map(dict((k, v) for k, v in zip(U, [float(u.left) for u in U])))
                dct_rename[col] = col + '_bin'
                df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.encode('ascii', 'ignore').str.decode('ascii')

        if col in dct_methods:
            if dct_methods[col]=='log':
                df[col]= pt.fit_transform(df[col].values.reshape((-1, 1)))
                dct_rename[col]=col+'_log'
            elif dct_methods[col]=='binning':
                U, Q = pd.qcut(df[col].rank(method='first'), q=q, retbins=True)
                dct = dict((k,v) for k,v in zip(U,[float(u.left) for u in U]))
                df[col] = U.map(dct)
                df[col] = df[col].astype(float)
                dct_rename[col] = col + '_bin'
            elif dct_methods[col]=='ignore':
                df[col] = numpy.nan
            elif dct_methods[col]=='cat':
                df[col]=df[col].astype(str)
                dct_rename[col] = col + '_cat'

    df = df.rename(columns=dct_rename)

    return df
# ---------------------------------------------------------------------------------------------------------------------
def apply_filter(df,col_name,filter,inverce=False):
    if df.shape[0]==0:
        return df

    if filter is None:
        return df
    elif isinstance(filter,(list,tuple,numpy.ndarray,pd.Series)):
        idx = numpy.full(df.shape[0], True)
        if len(filter)==0:
            idx= ~idx
        elif len(filter)==1:
            idx = (df[col_name] == filter[0])
        elif len(filter)==2:
            if isinstance(filter,(list,tuple,numpy.ndarray)):
                if filter[0] is not None:
                    idx = (idx) & (df[col_name]>=filter[0])
                if filter[1] is not None:
                    idx = (idx) & (df[col_name]<filter[1])
            else:
                idx = df[col_name].isin(filter)
        elif len(filter)>2:
            idx = df[col_name].isin(filter)
    else:
        #idx0 = (df[col_name] == filter)
        if pd.DataFrame([filter]).isna()[0][0]:
            idx = df[col_name].isna()
        else:
            idx = (df[col_name] == filter)



    if inverce:
        idx=~idx

    return df[idx]
# ---------------------------------------------------------------------------------------------------------------------
def prettify(df,showheader=True,showindex=True,tablefmt='psql',filename_out=None):
    res = tabulate(df, headers=df.columns if showheader else [], tablefmt=tablefmt, showindex=showindex)

    if filename_out is not None:
        with open(filename_out, 'w', encoding='utf-8') as f:
            f.write(res)

    return res
# ---------------------------------------------------------------------------------------------------------------------

def fetch(df1,col_name1,df2,col_name2,col_value,col_new_name=None):
    df_res = df1.copy()

    if isinstance(col_value,list):
        for c,col in enumerate(col_value):
            if isinstance(col_name2, list):
                lst = col_name2 + [col]
                ddd = df2[lst].drop_duplicates(subset=col_name2)
            else:
                ddd = df2[[col_name2, col]].drop_duplicates(subset=[col_name2])
            V = pd.merge(df1[col_name1], ddd, how='left',left_on=col_name1, right_on=col_name2)[col]
            df_res[col if col_new_name is None else col_new_name[c]] = [v for v in V.values]
    else:
        if isinstance(col_name2,list):
            ddd = df2[col_name2+[col_value]].drop_duplicates(subset=col_name2)
        else:
            ddd = df2[[col_name2, col_value]].drop_duplicates(subset=[col_name2])
        V = pd.merge(df1[col_name1],ddd,how='left',left_on=col_name1,right_on=col_name2)[col_value]
        df_res[(col_value if col_new_name is None else col_new_name)] = [v for v in V.values]

    return df_res
# ---------------------------------------------------------------------------------------------------------------------
def is_categorical(df,col_name):

    types = numpy.array([str(t) for t in df.dtypes])
    typ = types[df.columns==col_name]
    is_categorical = (typ in ['object', 'category', 'bool'])

    return is_categorical
# ---------------------------------------------------------------------------------------------------------------------
def pretty_size(value):
    base_fmt = '%3d'
    if   value > 1e+15: res = (base_fmt+'P')% int(value/1e+15)
    elif value > 1e+12: res = (base_fmt+'T')% int(value/1e+12)
    elif value > 1e+9 : res = (base_fmt+'G')% int(value/1e+9)
    elif value > 1e+6 : res = (base_fmt+'M')% int(value/1e+6)
    elif value > 1e+3 : res = (base_fmt+'k')% int(value/1e+3)
    else:           res = '%d' % value

    return res
# ---------------------------------------------------------------------------------------------------------------------
def apply_format(df,format='%.2f',str_na=''):

    df_res = df.copy()

    if isinstance(df,(pd.Series)):
        df_res = df_res.apply(lambda x: (format % x) if not numpy.isnan(x) else str_na).astype(str)
    else:
        for c in df.columns:
            df_res[c] = df_res[c].apply(lambda x: (format % x) if not numpy.isnan(x) else str_na).astype(str)

    return df_res
# ---------------------------------------------------------------------------------------------------------------------
def get_seasonality_daily(df):

    mean = df.iloc[:, 1].mean()
    col = df.columns[1]
    df['avg'] = df.rolling(7).mean()
    df['DOW'] =pd.to_datetime(df.iloc[:,0], format='%Y-%m-%d', errors='ignore').dt.strftime('%a')
    df['DOW_num'] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%d', errors='ignore').dt.dayofweek
    df.iloc[:,1] = df.iloc[:,1]/df['avg']
    df = df.dropna()
    df_agg = my_agg(df,['DOW','DOW_num'],[col],['mean'],list_res_names=['DOW'],order_idx=1)
    df_agg.iloc[:,2]*=mean

    return df_agg.iloc[:,[0,2]]
# ---------------------------------------------------------------------------------------------------------------------
def my_append(df,idx_col,values):
    if len(values)==0:return df
    #if values.shape[0]==0:return df

    df_temp = pd.DataFrame({df.columns[idx_col]:values})
    df_res = df.append(df_temp, ignore_index=True)
    return df_res
# ---------------------------------------------------------------------------------------------------------------------
def build_hierarchical_dataframe0(df, cols_labels_level, col_size, cols_metric=None, function=None):
    df_hierarchical = pd.DataFrame(columns=['id', 'parent_id', 'size', 'metric'])
    for i, level in enumerate(cols_labels_level):
        df_tree = pd.DataFrame(columns=['id', 'parent_id', 'size', 'metric'])

        g = cols_labels_level[i:]
        df_agg = df.groupby(g).sum()
        df_agg2 = df.groupby(g)

        df_agg = df_agg.reset_index()
        df_tree['id'] = df_agg[level].copy()
        if i < len(cols_labels_level) - 1:
            df_tree['parent_id'] = df_agg[cols_labels_level[i + 1]].copy()
        else:
            df_tree['parent_id'] = 'TOTAL'
        df_tree['size'] = df_agg[col_size]
        if cols_metric is not None and function is not None:
            aaa = function(*[df_agg[col] for col in cols_metric])
            df_tree['metric'] = aaa

        df_hierarchical = df_hierarchical.append(df_tree, ignore_index=True)

    total = pd.Series(dict(id='TOTAL', parent_id=numpy.nan, size=numpy.nan,metric=numpy.nan))
    df_hierarchical = df_hierarchical.append(total, ignore_index=True)
    df_hierarchical = df_hierarchical.sort_values(by=['parent_id','id'])

    return df_hierarchical
# ---------------------------------------------------------------------------------------------------------------------
def build_hierarchical_dataframe(df, cols_labels_level, col_size, concat_labels=False, dct_function=None, metrics_for_aggs=True,description_total='TOTAL'):

    cols_metric = dct_function['cols_metric'] if dct_function is not None else None
    metric_function = dct_function['metric_function'] if dct_function is not None else None
    do_calc_metrics = cols_metric is not None and metric_function is not None
    df0 = df.copy()

    if concat_labels:
        for col,col_agg in zip(cols_labels_level[::-1][1:],cols_labels_level[::-1][:-1]):
            df[col] = df[col_agg].astype(str) + '_' + col + '_' +df[col].astype(str)

    res = pd.DataFrame()
    for i, level in enumerate(cols_labels_level):
        res_level = pd.DataFrame()
        for l in df[level].unique():
            idx = (df[level] == l)
            df_level = df[idx]
            df_metric_level = df0[idx]
            parend_ids = df_level[cols_labels_level[i + 1]] if i < len(cols_labels_level) - 1 else description_total
            if do_calc_metrics:
                metric = metric_function(*[numpy.array(df_metric_level[col]) for col in cols_metric])
            else:
                metric = numpy.nan
            res_level = res_level.append(pd.DataFrame({'id':df_level[level],'parent_id':parend_ids,'size':df_level[col_size],'metric':metric}))

        if (i>0):
            res_level_agg = my_agg(res_level,cols_groupby=['id','parent_id'],cols_value=['size'],aggs=['sum']).copy()
            if do_calc_metrics and metrics_for_aggs:
                res_level_agg['metric'] = [metric_function(*[numpy.array(df0[df[level] == id][col]) for col in cols_metric]).mean() for id in res_level_agg['id']]
            else:
                res_level_agg['metric'] = numpy.nan
            res = res.append(res_level_agg)
        else:
            res = res.append(res_level)

    if do_calc_metrics and metrics_for_aggs:
        metric = metric_function(*[numpy.array(df0[col]) for col in cols_metric]).mean()
    else:
        metric = numpy.nan

    res = res.sort_values(by=['parent_id', 'id'])
    res = res.append(pd.Series({'id':description_total, 'parent_id':'', 'size':df[col_size].sum(), 'metric':metric}),ignore_index=True)
    return res
# ---------------------------------------------------------------------------------------------------------------------
def remap_counts(df,list_values):

    df_res = pd.DataFrame()

    for feature_value,is_nan in zip(list_values,(pd.DataFrame(list_values).isna().values).flatten()):
        if feature_value == '*':
            df_segm = df.copy()
        elif is_nan:
            df_segm = df[df.iloc[:, 0].isna()].copy()
        elif feature_value != '~':
            df_segm = df[df.iloc[:,0]==feature_value].copy()
        else:
            df_segm = df[~df.iloc[:,0].isin(list_values)].copy()
        df_segm.iloc[:, 0] = feature_value
        df_res = df_res.append(df_segm)

    df_res = my_agg(df_res,cols_groupby=df_res.columns[0],cols_value=[df_res.columns[1]],aggs=['sum'])

    return df_res
# ---------------------------------------------------------------------------------------------------------------------
def get_delta(df_agg1,df_agg2,absolute=False):

    S0 = pd.concat([df_agg1.iloc[:, 0], df_agg2.iloc[:, 0]]).rename(df_agg1.columns[0])
    S0.drop_duplicates(inplace=True)

    df = pd.merge(S0, df_agg1, how='left', on=[df_agg1.columns[0]])
    df = pd.merge(df, df_agg2, how='left', on=[df_agg2.columns[0]])
    df.fillna(0, inplace=True)
    df['delta'] = df.iloc[:,-1]-df.iloc[:,-2]
    if not absolute:
        df['delta']/= (df.iloc[:, -2]+(1e-4))
        cap = 2
        df['delta'] = df['delta'].apply(lambda x: cap if x> cap else x)
        df['delta'] = df['delta'].apply(lambda x:-cap if x<-cap else x)

    return df.iloc[:,[0,-1]]
# ---------------------------------------------------------------------------------------------------------------------
def to_hex(df):
    df2 = df.copy()
    for c in df.columns:
        df2[c]=df[c].apply(lambda x:''.join(format(b, '02x') for b in bytearray(struct.pack('f', x))))

    return df2
# ---------------------------------------------------------------------------------------------------------------------
def from_hex(df):
    df2 = df.copy()
    for c in df.columns:
        df2[c] = df[c].apply(lambda str_hex:struct.unpack('f', b''.join([bytes.fromhex(str_hex[i:i + 2]) for i in range(0, len(str_hex), 2)]))[0])

    return df2.astype(float)
# ---------------------------------------------------------------------------------------------------------------------