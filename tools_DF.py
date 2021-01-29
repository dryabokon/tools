import numpy
# ----------------------------------------------------------------------------------------------------------------------
def df_to_XY(df,idx_target,keep_categoirical=True,drop_na=True):
    if drop_na:
        df = df.dropna()
    columns = df.columns.to_numpy()
    col_types = numpy.array([str(t) for t in df.dtypes])
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    if keep_categoirical==False:
        are_categoirical = numpy.array([cc in ['object', 'category', 'bool'] for cc in col_types])
        are_categoirical = numpy.delete(are_categoirical, idx_target)
        idx = idx[~are_categoirical]

    X = df.iloc[:,idx].to_numpy()
    Y = df.iloc[:,[idx_target]].to_numpy().flatten()
    return X,Y
# ----------------------------------------------------------------------------------------------------------------------
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
def hash_categoricals(df):

    columns = df.columns.to_numpy()
    col_types = numpy.array([str(t) for t in df.dtypes])
    are_categoirical = numpy.array([cc in ['object', 'category', 'bool'] for cc in col_types])
    C = numpy.arange(0, df.shape[1])[are_categoirical]

    for column in columns[C]:
        vv = df.loc[:, column].dropna()

        keys = numpy.unique(vv.to_numpy())
        values = numpy.arange(0,len(keys))
        dct = dict(zip(keys,values))
        df[column] = df[column].map(dct)

    return df
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