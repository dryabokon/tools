import pandas as pd
import numpy
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
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
def hash_categoricals(df,drop_na=True):

    df_res = df.copy()
    if drop_na:
        df_res = df_res.dropna()

    col_types = numpy.array([str(t) for t in df.dtypes])
    are_categoirical = \
        numpy.array([cc in ['object', 'category', 'bool']
                     for cc in col_types])
    C = numpy.arange(0, df.shape[1])[are_categoirical]

    columns = df.columns.to_numpy()
    for column in columns[C]:
        vv = df.loc[:, column].dropna()

        keys = numpy.unique(vv.to_numpy())
        values = numpy.arange(0,len(keys))
        dct = dict(zip(keys,values))
        df_res[column] = df[column].map(dct)
        df_res = df_res.astype({column: 'int32'})

    return df_res
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
