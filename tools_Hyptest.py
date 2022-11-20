import pandas as pd
import numpy
from scipy.stats import chi2, chisquare, entropy
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact,f_oneway
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
# from collections import Counter
# from scipy.spatial import distance
# from scipy.special import rel_entr
# from sklearn.metrics import mutual_info_score
# ----------------------------------------------------------------------------------------------------------------------
# https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss
# Chi-square goodness of fit: whether the observed proportions for a categorical variable differ from hypothesized proportions
# Chi-square test: if there is a relationship between two categorical variables.
# Fisher’s exact test: used when you want to conduct a chi-square test but one or more of your cells has an expected frequency of five or less.
# One sample t-test: whether a sample mean of a normally distributed interval variable significantly differs from a hypothesized value
# Two independent samples t-test: compare the means of a normally distributed interval dependent variable for two independent groups
# One-way analysis of variance (ANOVA):  is used when you have a categorical independent variable (with two or more categories) and a normally distributed interval dependent variable and you wish to test for differences in the means of the dependent variable broken down by the levels of the independent variable
# ----------------------------------------------------------------------------------------------------------------------
# https://stats.stackexchange.com/questions/4044/measuring-the-distance-between-two-multivariate-distributions
# Measuring distance between two distributions
# Heuristic: # Minkowski-form, Weighted-Mean-Variance (WMV),
# Nonparametric test statistics: 2 (Chi Square),Kolmogorov-Smirnov (KS),Cramer/von Mises (CvM),
# Information-theory divergences: Kullback-Liebler (KL), Jensen–Shannon divergence (metric), Jeffrey-divergence (numerically stable and symmetric)
# Ground distance measures
# ----------------------------------------------------------------------------------------------------------------------
class HypTest(object):
    def __init__(self):
        return
# ----------------------------------------------------------------------------------------------------------------------
    def verbose_cr_value(self,stat_value, critical_value):
        if stat_value < critical_value:
            print('stat_value < critical_value')
            print('%1.2f            < %1.2f' % (stat_value, critical_value))
            print('Accept H0: fit OK')
        else:
            print("critical_value <= stat_value")
            print('%1.2f          <= %1.2f' % (critical_value, stat_value))
            print('Reject H0: no fit')
        print()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def verbose_p_value(self,p_value, a_significance_level):

        if p_value <= a_significance_level:
            print("p value <= significance_level")
            print('%1.4f    < %1.2f' % (p_value, a_significance_level))
            print('Reject H0: no fit')
        else:
            print("significance_level < p value")
            print('%1.2f               < %1.4f' % (a_significance_level, p_value))
            print('Accept H0: fit OK')
        print()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_chi2_stats(self,stat_value_chi2,deg_of_freedom,a_significance_level=0.05):

        critical_value = chi2.ppf(1 - a_significance_level, deg_of_freedom)
        X_good = numpy.linspace(0, critical_value, 50)
        X_bad = numpy.linspace(critical_value, critical_value * 2, 50)

        Y_good = chi2.pdf(X_good, deg_of_freedom)
        Y_bad = chi2.pdf(X_bad, deg_of_freedom)

        plt.plot(X_good, Y_good, color=(0, 0, 0.5))
        plt.plot(X_bad, Y_bad, color=(0.5, 0, 0))

        plt.fill_between(X_good, Y_good, color=(0.7, 0.8, 1))
        plt.fill_between(X_bad, Y_bad, color=(1, 0.9, 0.9))

        marker = 'bo' if stat_value_chi2<critical_value else 'ro'

        plt.plot(stat_value_chi2,0,marker)
        plt.tight_layout()
        plt.show()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def normalize(self,n_obs, n_exp):
        X_obs, X_exp = numpy.array(n_obs), numpy.array(n_exp)
        X_exp_normalized = X_exp * X_obs.sum() / (X_exp.sum())
        return X_obs, X_exp_normalized
# ----------------------------------------------------------------------------------------------------------------------
    def is_same_distribution_chi2(self, n_obs, n_exp, a_significance_level=0.05, deg_of_freedom=None,verbose=False,do_cross_check=False):

        # H0 =  "no difference" situation, samples are from same G
        # p_value = how likely it is that your data would have occurred by random chance (i.e. that the null hypothesis is true)

        if deg_of_freedom is None:
            deg_of_freedom = len(n_exp) - 1

        if any([(x2==0 and x1>0) for x1,x2 in zip(n_obs, n_exp)]):
            result,p_value = False,0
        else:
            X_obs, X_exp_normalized = self.normalize(n_obs, n_exp)
            stat_value_chi2, p_value  = chisquare(X_obs, X_exp_normalized)
            result = (a_significance_level<=p_value)

            if do_cross_check:
                p_value_check1 = chi2.sf(stat_value_chi2, deg_of_freedom)
                critical_value = chi2.ppf(1 - a_significance_level, deg_of_freedom)
                result_check1 = stat_value_chi2 < critical_value
                if verbose:
                    self.verbose_cr_value(stat_value_chi2, critical_value)

        if verbose:
            self.verbose_p_value(p_value, a_significance_level)
            self.plot_chi2_stats(stat_value_chi2, deg_of_freedom, a_significance_level)

        return result, p_value
# ----------------------------------------------------------------------------------------------------------------------
    def is_same_distribution_fisher(self,n_obs, n_exp,a_significance_level=0.05,verbose=False):
        #conduct a chi-square test when min(freq_exp) <= 5

        n_obs, n_exp = numpy.array(n_obs,dtype=numpy.int32), numpy.array(n_exp,dtype=numpy.int32)

        if n_obs.shape[0]!=2:
            return None,None

        oddsratio, p_value = fisher_exact([n_obs, n_exp])
        result = p_value <= a_significance_level

        if verbose:
            self.verbose_p_value(p_value, a_significance_level)

        return result, p_value
# ----------------------------------------------------------------------------------------------------------------------
    def is_same_distribution_Anova(self,X_obs, X_exp,a_significance_level=0.05,verbose=False):
        # tests the null hypothesis that two or more groups have the same population mean

        stat_value, p_value = f_oneway(X_obs, X_exp)
        result = p_value <= a_significance_level

        return result, p_value
# ---------------------------------------------------------------------------------------------------------------------
    def classification_metrics_aggs_cat(self, df_agg1, df_agg2):
        S0 = pd.concat([df_agg1.iloc[:,0],df_agg2.iloc[:,0]]).rename(df_agg1.columns[0])
        S0.drop_duplicates(inplace=True)

        df1 = pd.merge(S0, df_agg1, how='left', on=[df_agg1.columns[0]])
        df1.fillna(0, inplace=True)
        df2 = pd.merge(S0, df_agg2, how='left', on=[df_agg1.columns[0]])
        df2.fillna(0, inplace=True)

        if (df1.shape[0] == df2.shape[0] == 1):
            f1,P,R  = 0,0,0
        else:
            if df1.iloc[:, 1].sum()*df2.iloc[:, 1].sum() == 0:
                f1, P, R = 1, 1, 1
            else:
                Major_X, Minor_X = numpy.array(df1.iloc[:, 1].values), numpy.array(df2.iloc[:, 1].values)
                if Major_X.sum() < Minor_X.sum():
                    Major_X, Minor_X = Minor_X, Major_X

                strategy0 = numpy.array([1 if x0/Major_X.sum()>x1/Minor_X.sum() else 0 for x0,x1 in zip(Major_X, Minor_X)])
                strategy1 = 1 - strategy0

                hits0 = [x*(y==1) for x,y in zip(Minor_X,strategy0)]
                fp0   = [x*(y==1) for x,y in zip(Major_X,strategy0)]
                pos_dec0 = [(x0 + x1) if y == 1 else 0 for x0, x1, y in zip(Major_X, Minor_X, strategy0)]

                hits1 = [x*(y==1) for x,y in zip(Minor_X,strategy1)]
                fp1   = [x*(y==1) for x,y in zip(Major_X,strategy1)]
                pos_dec1 = [(x0 + x1) if y == 1 else 0 for x0, x1, y in zip(Major_X, Minor_X, strategy1)]

                R0 = sum(hits0)/sum(Minor_X)
                P0 = 1- sum(fp0)/(sum(pos_dec0)+1e-4)

                R1 = sum(hits1) / sum(Minor_X)
                P1 = 1 - sum(fp1) / (sum(pos_dec1)+1e-4)

                if numpy.isnan(P0*R0*2/(P0+R0+1e-4)):
                    f1 = P1*R1*2 /(P1+R1+1e-4)
                elif numpy.isnan(P1*R1*2/(P1+R1+1e-4)):
                    f1 = P0*R0*2/(P0+R0+1e-4)
                else:
                    if P1*R1*2 /(P1+R1+1e-4) > P0*R0*2/(P0+R0+1e-4):
                        f1 = P1*R1*2 /(P1+R1+1e-4)
                        P,R = P1,R1
                    else:
                        f1 = P0 * R0 * 2 / (P0 + R0 + 1e-4)
                        P, R = P0, R0

        return f1,P,R
# ---------------------------------------------------------------------------------------------------------------------
    def f1_score(self, S_raw0, S_raw1, is_categorical,return_PR=False):

        S_raw0 = S_raw0.values.flatten()
        S_raw1 = S_raw1.values.flatten()

        if is_categorical:
            C0 = pd.Series(S_raw0).value_counts()
            C1 = pd.Series(S_raw1).value_counts()
            df_agg0 = pd.DataFrame({'K': C0.index.values, 'V': C0.values})
            df_agg1 = pd.DataFrame({'K': C1.index.values, 'V': C1.values})
            f1,P,R = self.classification_metrics_aggs_cat(df_agg0, df_agg1)
        else:
            if S_raw0.shape[0]>S_raw1.shape[0]:
                y = numpy.concatenate([numpy.full(S_raw0.shape[0], 0),numpy.full(S_raw1.shape[0], 1)],axis=0)
                s = numpy.array([s for s in S_raw0]+[s for s in S_raw1]).astype(numpy.float32)
            else:
                y = numpy.concatenate([numpy.full(S_raw1.shape[0], 0), numpy.full(S_raw0.shape[0], 1)], axis=0)
                s = numpy.array([s for s in S_raw1] + [s for s in S_raw0]).astype(numpy.float32)

            precisions, recalls, thresholds = metrics.precision_recall_curve(y, s)
            idx_th = numpy.argmax([p * r / (p + r + 1e-4) for p, r in zip(precisions, recalls)])
            P = precisions[idx_th]
            R = recalls[idx_th]
            f1 = P*R*2/(P+R+1e-4)

        if return_PR:
            return f1,P,R

        return f1
# ---------------------------------------------------------------------------------------------------------------------
    def f1_score_2d(self, df0, df1):

        C0 = df0.value_counts()
        C1 = df1.value_counts()

        df_agg0 = pd.DataFrame({'K': C0.index.values, 'V': C0.values})
        df_agg1 = pd.DataFrame({'K': C1.index.values, 'V': C1.values})

        N = 50

        if df_agg0.shape[0]<N and df_agg1.shape[0]<N:
            f1,P,R = self.classification_metrics_aggs_cat(df_agg0, df_agg1)
        else:
            df = pd.concat([df0, df1])
            df = tools_DF.hash_categoricals(df)
            X, Y = df.values, numpy.array([0] * df0.shape[0] + [1] * df1.shape[0])
            model = LogisticRegression().fit(X,Y)
            precisions, recalls, thresholds = metrics.precision_recall_curve(Y, model.predict_proba(X)[:,1])
            idx_th = numpy.argmax([p * r / (p + r + 1e-4) for p, r in zip(precisions, recalls)])
            P,R = precisions[idx_th],recalls[idx_th]
            f1 = P * R * 2 / (P + R + 1e-4)

        return f1
# ---------------------------------------------------------------------------------------------------------------------
    def f1_scores(self, df, idx_target):

        columns = df.columns
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
        unique_targets = df.iloc[:, idx_target].unique().tolist()

        Q = numpy.zeros((len(idx),len(idx)))

        for i in range(len(idx)-1):
            print(i)
            for j in range(i+1,len(idx)):
                c1, c2 = columns[idx[i]], columns[idx[j]]
                df_temp = df[[target, c1, c2]].copy()
                df_temp.dropna(inplace=True)
                if len(unique_targets) == 2:
                    Q[i,j] = Q[j,i]= self.f1_score(df_temp[df_temp[target] == unique_targets[0]].iloc[:, 1:],
                                                                df_temp[df_temp[target] == unique_targets[1]].iloc[:, 1:], False)

        Q = pd.DataFrame(Q,columns=columns[idx],index=columns[idx])

        return Q
# ---------------------------------------------------------------------------------------------------------------------
    def encode_p_value(self,p_value):

        if p_value is None:
            res = 'na'
        else:
            res = '%f'%p_value
        #elif (p_value>1e-6) or (p_value==0):res = '%1.6f'%p_value
        #else:res = '0.e%05d'%(numpy.log(p_value)/numpy.log(10))
        res=res.replace('.', '_')
        return res
# ---------------------------------------------------------------------------------------------------------------------