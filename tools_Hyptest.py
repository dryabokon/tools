import pandas as pd
import numpy
from scipy.stats import chi2, chisquare, entropy
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact,f_oneway
from scipy.spatial import distance
from scipy.special import rel_entr
from sklearn.metrics import mutual_info_score
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
            #self.plot_chi2_stats(stat_value_chi2, deg_of_freedom, a_significance_level)

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
    def distribution_distance_aggs(self,df_agg1,df_agg2):
        S0 = pd.concat([df_agg1.iloc[:,0],df_agg2.iloc[:,0]]).rename(df_agg1.columns[0])
        S0.drop_duplicates(inplace=True)

        df_ref = pd.merge(S0, df_agg1, how='left', on=[df_agg1.columns[0]])
        df_ref.fillna(0, inplace=True)
        df_insp = pd.merge(S0, df_agg2, how='left', on=[df_agg1.columns[0]])
        df_insp.fillna(0, inplace=True)

        if (df_insp.shape[0] == df_ref.shape[0] == 1):
            res = 0
        else:
            X_obs, X_exp = numpy.array(df_insp.iloc[:, 1].values), numpy.array(df_ref.iloc[:, 1].values)
            if X_exp.sum() * X_obs.sum() == 0:
                res = 1
            else:
                X_exp = X_exp/(X_exp.sum())
                X_obs = X_obs/(X_obs.sum())
                res = distance.minkowski(X_obs, X_exp,2)

        return res
# ---------------------------------------------------------------------------------------------------------------------
    def distribution_distance(self, S_raw1, S_raw2):

        C1 = S_raw1.value_counts()
        C2 = S_raw2.value_counts()
        df_agg1 = pd.DataFrame({'K':C1.index.values,'V':C1.values})
        df_agg2 = pd.DataFrame({'K':C2.index.values,'V':C2.values})
        res = self.distribution_distance_aggs(df_agg1,df_agg2)

        return res
# ---------------------------------------------------------------------------------------------------------------------
    def distribution_distances(self,df, idx_target):

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
                    Q[i,j] = Q[j,i]= self.distribution_distance(df_temp[df_temp[target] == unique_targets[0]].iloc[:, 1:],
                                                                df_temp[df_temp[target] == unique_targets[1]].iloc[:, 1:])

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