import pandas as pd
from sklearn.feature_selection import SelectKBest,f_regression, f_classif, mutual_info_classif
from sklearn import linear_model
from sklearn.metrics import r2_score
from xgboost import XGBClassifier,XGBRegressor
import numpy
import time
from scipy.stats import chi2, chisquare
import shap
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
def select_features(X, Y):
    score_func = mutual_info_classif
    #score_func = f_regression
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
    Y = df.iloc[:, idx_target].to_numpy().astype(numpy.int)
    return X,Y
# ----------------------------------------------------------------------------------------------------------------------
def feature_imporance_F_score(df, idx_target=0):

    X, Y = preprocess(df,idx_target)

    f_scores = select_features(X, Y)
    f_scores = f_scores/f_scores.sum()

    return f_scores
# ----------------------------------------------------------------------------------------------------------------------
def feature_importance_C(df, idx_target=0):
    #https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py

    X, Y = preprocess(df, idx_target)
    ridgereg = linear_model.Ridge(alpha=0.001, normalize=True)
    ridgereg.fit(X, Y)
    values = numpy.abs(ridgereg.coef_ * X.std(axis=0))
    return  values
# ----------------------------------------------------------------------------------------------------------------------
def feature_importance_SHAP(df, idx_target=0):

    X, Y = preprocess(df, idx_target)

    #model = XGBRegressor().fit(X, Y)
    model = XGBClassifier().fit(X, Y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X).values

    values = numpy.abs(numpy.mean(shap_values,axis=0))

    return values
# ----------------------------------------------------------------------------------------------------------------------
def feature_importance_R2(df, idx_target=0):

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
def feature_importance_XGB(df, idx_target):
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
def feature_imporance_info(df, idx_target):
    X, Y = preprocess(df, idx_target)

    feature_importances = mutual_info_classif(X,Y)

    return feature_importances
# ----------------------------------------------------------------------------------------------------------------------
def feature_importance_cross_H(df, idx_target):
    X, Y = preprocess(df, idx_target)

    xp = X[Y>0]
    xn = X[Y<=0]
    n_p = xp.shape[0]
    n_n = xn.shape[0]

    cross_H = []


    for i in range(X.shape[1]):
        p_p1 = xp[:,i].sum()/n_p
        p_p0 = (1-p_p1)
        p_n1 = xn[:,i].sum()/n_n
        p_n0 = 1-p_n1

        v = chisquare([p_p0,p_p1],[p_n0,p_n1])[0]
        cross_H.append(v)

    return numpy.array(cross_H)
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_feature_importance(df, idx_target,do_debug=False):
    columns = df.columns.to_numpy()[numpy.delete(numpy.arange(0, df.shape[1]), idx_target)]

    start_time = time.time()
    S_XGB = feature_importance_XGB(df, idx_target)
    if do_debug:
        print('F_score %s sec\n\n' % (time.time() - start_time))

    start_time = time.time()
    S_R2 = feature_importance_R2(df, idx_target)
    if do_debug:
        print('R2 %s sec\n\n' % (time.time() - start_time))

    start_time = time.time()
    S_C = feature_importance_C(df, idx_target)
    if do_debug:
        print('C %s sec\n\n' % (time.time() - start_time))

    start_time = time.time()
    S_F_score = feature_imporance_F_score(df, idx_target)
    if do_debug:
        print('XGB %s sec\n\n' % (time.time() - start_time))

    start_time = time.time()
    S_SHAP = feature_importance_SHAP(df, idx_target)
    if do_debug:
        print('XGB %s sec\n\n' % (time.time() - start_time))

    #start_time = time.time()
    #S6 = feature_imporance_info(df, idx_target)
    #if do_debug:
    #    print('info %s sec\n\n' % (time.time() - start_time))

    df = pd.DataFrame({
        'features': columns,
        'XGB': S_XGB,
        'F_score': S_F_score,
        'R2': S_R2,
        'C': S_C,
        'SHAP': S_SHAP,
    })

    df.sort_values(by=df.columns[1],inplace=True,ascending=False)

    return df
# ----------------------------------------------------------------------------------------------------------------------
