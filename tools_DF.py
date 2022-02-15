import pandas as pd
import numpy
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from collections import Counter
from tabulate import tabulate
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
def get_categoricals_hash_map(df):

    res_dict= {}

    for column,typ in zip(df.columns,df.dtypes):
        is_categoirical = str(typ) in ['object', 'category']
        is_bool = str(typ) in ['bool']
        if not is_categoirical and not is_bool:continue

        if column=='adult_male':
            ii=0

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

    imp = SimpleImputer(missing_values=numpy.nan, strategy=strategy)
    for column in df.columns:
        if any([isinstance(v, bool) for v in df[column]]):
            if strategy_bool == 'int':
                df[column] = df[column].fillna(numpy.nan).map(dict(zip([False,numpy.nan,True], [-1,0,1]))).astype('int32')
            elif strategy_bool == 'str':
                idx_true = (df[column].isin([True,'True','TRUE','true']))
                idx_false = (df[column].isin([False,'False','FALSE','false']))
                V1 = pd.Series(numpy.full(df.shape[0],'n/a')).astype(str)
                V1[idx_true] = 'true'
                V1[idx_false] = 'false'
                df[column] = V1

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

    df_res = df.groupby(cols_groupby).agg(dct_agg)
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
        df_res = df_res.append(df_frame, ignore_index=True)

    return df_res
# ---------------------------------------------------------------------------------------------------------------------
def to_multi_column(df,idx_time,idx_label,idx_value,order_by_value=False):
    columns = [c for c in df.columns]
    col_time = columns[idx_time]
    col_label = columns[idx_label]
    col_value = columns[idx_value]

    #df.to_csv(self.folder_out+'xxx.csv',index = False)
    df.drop_duplicates(inplace=True)

    df_res = df.pivot(index=col_time, columns=col_label)[col_value]
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
def preprocess(df,dct_methods):
    if df.shape[0]==0:
        return df

    pt = preprocessing.PowerTransformer()
    dct_rename={}

    for col in df.columns:
        if col in dct_methods:
            if dct_methods[col]=='log':
                df[col]= pt.fit_transform(df[col].values.reshape((-1, 1)))
                dct_rename[col]=col+'_log'
            elif dct_methods[col]=='numeric':
                df[col] = df[col].astype(float)
            elif dct_methods[col]=='ignore':
                df[col] = numpy.nan
            elif dct_methods[col]=='cat':
                df[col]=df[col].astype(str)
                dct_rename[col] = col + '_cat'

    df = df.rename(columns=dct_rename)

    return df
# ---------------------------------------------------------------------------------------------------------------------
def apply_filter(df,col_name,filter,inverce=False):

    if filter is None:
        return df
    elif isinstance(filter,(list,tuple,numpy.ndarray,pd.Series)):
        idx = numpy.full(df.shape[0], True)
        if len(filter)==0:
            idx= ~idx
        elif len(filter)==1:
            idx = (df[col_name] == filter)
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
        idx = (df[col_name] == filter)

    if inverce:
        idx=~idx

    return df[idx]
# ---------------------------------------------------------------------------------------------------------------------
def pretty_print(df):
    print(tabulate(df,headers=df.columns,tablefmt='psql'))
    return
# ---------------------------------------------------------------------------------------------------------------------
def save_XL(filename_out,dataframes,sheet_names):

    writer = pd.ExcelWriter(filename_out, engine='openpyxl')
    for df,sheet_name in zip(dataframes,sheet_names):
        df.to_excel(writer, sheet_name=sheet_name,index=False)

    writer.save()
    return
# ---------------------------------------------------------------------------------------------------------------------
def fetch(df1,col_name1,df2,col_name2,col_value):
    df_res = df1.copy()

    if isinstance(col_value,list):
        for col in col_value:
            V = pd.merge(df1[col_name1], df2[[col_name2, col]].drop_duplicates(subset=[col_name2]), how='left',left_on=col_name1, right_on=col_name2)[col]
            df_res[col] = [v for v in V.values]
    else:
        ddd = df2[[col_name2, col_value]].drop_duplicates(subset=[col_name2])
        V = pd.merge(df1[col_name1],ddd,how='left',left_on=col_name1,right_on=col_name2)[col_value]
        df_res[col_value] = [v for v in V.values]

    return df_res
# ---------------------------------------------------------------------------------------------------------------------