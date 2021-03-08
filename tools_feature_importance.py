import pandas as pd
from sklearn.feature_selection import SelectKBest,f_regression, f_classif, mutual_info_classif
from sklearn import linear_model
from sklearn.metrics import r2_score
from xgboost import XGBClassifier
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
def select_features(X, Y):
    #score_func = mutual_info_classif
    score_func = f_regression
    fs = SelectKBest(score_func=score_func, k='all')

    for c in range(X.shape[1]):
        min_value = X[:,c].min()
        if min_value < 0:
            X[:, c]+=-min_value

    fs.fit(X, Y)
    xxx = fs.scores_
    norm = fs.scores_.sum()
    result =  xxx/norm

    return result
# ----------------------------------------------------------------------------------------------------------------------
def preprocess(df, idx_target):
    df = df.dropna()
    df = tools_DF.hash_categoricals(df)

    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    X = df.iloc[:, idx].to_numpy()
    Y = df.iloc[:, idx_target].to_numpy()
    return X,Y
# ----------------------------------------------------------------------------------------------------------------------
def feature_imporance_F_score(df, idx_target=0):

    X, Y = preprocess(df,idx_target)

    f_scores = select_features(X, Y)
    f_scores = f_scores/f_scores.sum()

    return f_scores
# ----------------------------------------------------------------------------------------------------------------------
def feature_imporance_C(df, idx_target=0):
    #https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py

    X, Y = preprocess(df, idx_target)
    ridgereg = linear_model.Ridge(alpha=0.001, normalize=True)
    ridgereg.fit(X, Y)
    values = numpy.abs(ridgereg.coef_ * X.std(axis=0))
    return  values
# ----------------------------------------------------------------------------------------------------------------------
def feature_imporance_R2(df, idx_target=0):

    X, Y = preprocess(df, idx_target)
    regr = linear_model.LinearRegression()

    R2s = []
    for i in range(X.shape[1]):
        idx = numpy.delete(numpy.arange(0, X.shape[1]), i)
        x = X[:,idx]
        regr.fit(x, Y)
        Y_pred = regr.predict(x).flatten()
        R2s.append(100*r2_score(Y, Y_pred))

    R2s = 1-numpy.array(R2s)
    R2s-= R2s.min()

    return R2s
# ----------------------------------------------------------------------------------------------------------------------
def feature_imporance_XGB(df, idx_target):
    X, Y = preprocess(df, idx_target)

    model = XGBClassifier()
    model.fit(X, Y)
    feature_importances = model.get_booster().get_score()
    # values = numpy.array([v[1] for v in feature_importances.items()])
    values = numpy.zeros(X.shape[1])
    for v in feature_importances.items():
        values[int(v[0][1:])] = v[1]

    return numpy.array(values)
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_feature_importance(df, idx_target):
    columns = df.columns.to_numpy()[numpy.delete(numpy.arange(0, df.shape[1]), idx_target)]
    S1 = feature_imporance_F_score(df, idx_target)
    S2 = feature_imporance_R2(df, idx_target)
    S3 = feature_imporance_C(df, idx_target)
    S4 = feature_imporance_XGB(df, idx_target)


    df = pd.DataFrame({
        'F_score': S1,
        'R2': S2,
        'C': S3,
        'XGB': S4,
        'features': columns
    })

    return df
# ----------------------------------------------------------------------------------------------------------------------
