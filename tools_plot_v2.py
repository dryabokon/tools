import numpy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest,f_regression, chi2
# ----------------------------------------------------------------------------------------------------------------------
def plot_regression_df(df, idx_target, idx_num, idx_cat, filename_out=None):
    columns = df.columns.to_numpy()
    name_num, name_cat, name_target = columns[[idx_num, idx_cat, idx_target]]
    seaborn.lmplot(x=name_num, y=name_target, col=name_cat, hue=name_cat, data=df, y_jitter=.02, logistic=True,truncate=False)
    if filename_out is not None:
        plt.savefig(filename_out)
    return


# ----------------------------------------------------------------------------------------------------------------------
def plot_regression_YX(Y, X,logistic=False,filename_out=None):
    columns = ['Y', 'X']
    A = numpy.hstack((Y.reshape((-1,1)), X.reshape((-1,1))))
    df = pd.DataFrame(data=A, columns=columns)

    seaborn.lmplot(data=df, x=columns[1], y=columns[0], y_jitter=.02, logistic=logistic, truncate=False)
    if filename_out is not None:
        plt.savefig(filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_feature_importance_XGB(X,Y,header,N=5,filename_out=None):

    model = XGBClassifier()
    model.fit(X, Y)
    feature_importances = model.get_booster().get_score()
    values = numpy.array([ v[1] for v in feature_importances.items()])


    idx = numpy.argsort(-values)
    values,header = values[idx],header[idx]

    fig= plt.figure()
    ax = fig.gca()
    ax.pie(values[:N],  labels=header[:N], autopct='%1.1f%%',shadow=False, startangle=90)
    if filename_out is not None:
        plt.savefig(filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_feature_importance_LM(X,Y,header,N=5,filename_out=None):

    fs = SelectKBest(score_func=f_regression, k='all')

    for c in range(X.shape[1]):
        min_value = X[:,c].min()
        if min_value < 0:
            X[:, c]+=-min_value


    fs.fit(X, Y)
    values = fs.scores_

    idx = numpy.argsort(-values)
    values,header = values[idx],header[idx]

    fig= plt.figure()
    ax = fig.gca()
    ax.pie(values[:N],  labels=header[:N], autopct='%1.1f%%',shadow=False, startangle=90)
    if filename_out is not None:
        plt.savefig(filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------