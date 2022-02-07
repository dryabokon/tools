import hashlib
import numpy
import pandas as pd
import os
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_plot_v2
import tools_Hyptest
import tools_DF
# ---------------------------------------------------------------------------------------------------------------------
class EDA:
    def __init__(self, folder_in, folder_out,dark_mode=True):

        self.folder_in = folder_in
        self.folder_out = folder_out
        self.filename_prk_in = self.folder_in + 'asdad0'
        self.P = tools_plot_v2.Plotter(folder_out, dark_mode=dark_mode)
        self.HT = tools_Hyptest.HypTest()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def print_metadata(self, filename_in):
        metadata = ParquetFile(filename_in).metadata
        print(metadata)

        schema = pq.read_schema(filename_in, memory_map=True)
        for name, pa_dtype in zip(schema.names, schema.types):
            print(name,':',pa_dtype)

        return

# ---------------------------------------------------------------------------------------------------------------------
    def remove_dups_small_file(self, filename_in, filename_out,sep=','):
        df = pd.read_csv(filename_in,sep=',')
        df.drop_duplicates(inplace=True)
        df.to_csv(filename_out,sep=sep,index=False)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def remove_dups_large_file(self,filename_in,filename_out):

        if filename_in == filename_out:
            temp_file = self.folder_out + 'delme.csv'
        else:
            temp_file = filename_out

        with open(filename_in, "r") as f_in:
            with open(temp_file, "w") as f_out:
                seen = set()
                for line in f_in:
                    line_hash = hashlib.md5(line.encode()).digest()
                    if line_hash not in seen:
                        seen.add(line_hash)
                        f_out.write(line)

        if filename_in == filename_out:
            tools_IO.remove_file(filename_in)
            os.rename(temp_file,filename_out)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def to_ratios(self,df,idx_time):
        columns = df.columns.to_numpy()
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_time)

        value_sum = numpy.sum(df.iloc[:,idx].values,axis=1)
        for r in range(df.shape[0]):
            df.iloc[r,idx]/=(value_sum[r]/100)

        return df
# ---------------------------------------------------------------------------------------------------------------------
    def get_top_labels(self, df_multicolumn, max_objects, start_index):

        L = []
        Labels = numpy.array(df_multicolumn.columns.tolist())

        for step in range(df_multicolumn.shape[0]):
            values = df_multicolumn.iloc[step, start_index:].values
            idx = (start_index + numpy.array(numpy.argsort(-values)))[:max_objects]
            L.append(Labels[idx])

        labels = numpy.unique(numpy.array(L).flatten())

        # for label in labels:
        #     ss = str(label)
        #     xxx = df_multicolumn[ss]
        #     yyy = xxx.sum()


        values = numpy.array([df_multicolumn[str(label)].sum() for label in labels])
        idx = numpy.array(numpy.argsort(-values))
        labels = labels[idx]
        labels=labels[:max_objects*2]
        labels=labels[::-1]


        # values = numpy.array([df_norm[labels].iloc[-1] for label in labels])
        # idx = numpy.array(numpy.argsort(-values))
        # labels = labels[idx][::-1]

        return  labels
# ---------------------------------------------------------------------------------------------------------------------
    def get_TS_one_frame(self, df_norm, top_objects, start_index=2, cumul=True):

        labels = self.get_top_labels(df_norm,top_objects,start_index)

        X = df_norm.iloc[:, 0].values
        bottom = numpy.zeros(df_norm.shape[0])
        tops, bottoms = [], []
        for label in labels:
            Y = df_norm[label].to_numpy()
            if cumul:
                bottoms.append(bottom * 1.0)
                tops.append(Y + bottom)
                bottom += Y
            else:
                tops.append(Y)
                bottoms.append(Y)

        return X, labels, numpy.array(tops), numpy.array(bottoms)
# ---------------------------------------------------------------------------------------------------------------------
    def plot_static_timeline_chart(self, dataframe, filename_out, ts_time=None,to_ratios=True, in_format_x=None, out_format_x=None, major_step=None, top_objects=10,mode='pointplot',figsize=(8, 6)):

        df = pd.read_csv(dataframe, sep=',') if isinstance(dataframe,str) else dataframe

        if in_format_x is not None:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format=in_format_x, errors='ignore')

        df = tools_DF.to_multi_column(df, idx_time=0, idx_label=1, idx_value=2)
        if ts_time is not None:
            df = pd.merge(ts_time, df, how='left', on=[df.columns[0]])

        if to_ratios:df = self.to_ratios(df, idx_time=0)

        X, labels, tops, bottoms = self.get_TS_one_frame(df, top_objects, start_index=1, cumul=True)
        df2 = pd.DataFrame({'X':X})
        for label, ts in zip(labels, tops):
            df2[label] = ts

        idx_target = numpy.arange(1, df2.shape[1])

        self.P.TS_seaborn(df2, idxs_target=idx_target.tolist()[::-1],idxs_fill=idx_target-1,idx_time=0,
                          out_format_x=out_format_x,major_step=major_step,transparency=0.25,
                          mode=mode, lw=1,remove_xticks=False,figsize=figsize,filename_out=filename_out)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def patch_col_week(self,df,col_name):

        values = [str(v) for v in df[col_name].values]
        start = pd.to_datetime(pd.Series([(v[:4]) + '-01-01' for v in values]))
        days = [int(v[4:]) * 7 for v in values]
        days = pd.Series([pd.Timedelta(days=d) for d in days])
        start += days
        df[col_name] =start
        return df
# ---------------------------------------------------------------------------------------------------------------------
    def group_monthly(self, df_daily, col_time):
        df_monthly = df_daily.copy()
        S1 = [y+m+d for y, m, d in zip(df_daily[col_time].dt.year.astype(str),df_daily[col_time].dt.month.astype(str),df_daily[col_time].dt.day.astype(str))]
        df_monthly[col_time] = pd.to_datetime(S1,format='%Y%m%d')
        return df_monthly
# ---------------------------------------------------------------------------------------------------------------------
    def unify_records(self, df_month, df_week, df_day, level='month',date_start=None, date_stop=None):

        col_time = 'mswhen'
        df_month[col_time] = pd.to_datetime(df_month[col_time], format='%Y%m', errors='ignore')
        df_day[col_time] = pd.to_datetime(df_day[col_time], format='%Y%m%d', errors='ignore')
        df_week = self.patch_col_week(df_week, col_time)

        if level=='month':
            df_day[col_time]  = pd.to_datetime(df_day[col_time].dt.to_period('M').astype(str), format='%Y-%m',errors='ignore')
            df_week[col_time] = pd.to_datetime(df_week[col_time].dt.to_period('M').astype(str), format='%Y-%m',errors='ignore')
            df_res = df_month.append([df_day, df_week], ignore_index=True)
        else:
            return None

        if date_start is not None: df_res = df_res[df_res[col_time] >= date_start]
        if date_stop is not None:  df_res = df_res[df_res[col_time] <  date_stop]
        df_res = df_res[['reportuserid', 'msval', 'mswhen', 'auxvalue']]
        df_res = tools_DF.my_agg(df_res, ['reportuserid', 'mswhen', 'auxvalue'], ['msval'], ['sum'],order_idx=1, ascending=False)

        return df_res
# ---------------------------------------------------------------------------------------------------------------------
    def plot_static(self, df_month_patched,msname, level,unique=True,to_ratios=False, n_tops=6,exclude_labels=None,out_format_x='%Y-%b',figsize=(8,6)):

        agg = 'count' if unique else 'sum'
        suffix = 'unique' if unique else 'total'
        filename_out ='%s_%s.png' % (msname,suffix)

        if exclude_labels is not None:
            df_month_patched=df_month_patched[~df_month_patched['auxvalue'].isin(exclude_labels)]


        df_month_agg_dyn = tools_DF.my_agg(df_month_patched, ['mswhen', 'auxvalue'], ['msval'], [agg], order_idx=0,ascending=True)
        df_month_agg_dyn_sum = tools_DF.my_agg(df_month_agg_dyn, ['auxvalue'], ['msval'], ['sum'], order_idx=1,ascending=False)

        # df_month_agg_dyn.to_csv(self.folder_out+'df_month_agg_dyn.csv',index=False)
        # df_month_agg_dyn_sum.to_csv(self.folder_out + 'df_month_agg_dyn_sum.csv', index=False)


        self.P.plot_squarify(df_month_agg_dyn_sum[:3*n_tops], stat='#', idx_label=0, idx_count=1,alpha=0.1,figsize=(15, 6),filename_out=filename_out)

        self.plot_static_timeline_chart(df_month_agg_dyn, '%s_%s_%s_total.png' % (msname,level,suffix),
                                        to_ratios=to_ratios, in_format_x='%Y-%m-%d', out_format_x=out_format_x,
                                        major_step=30, top_objects=n_tops,figsize=figsize)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def plot_dynamics(self,df_month_patched,msname,unique=True,n_tops=6):

        agg = 'count' if unique else 'sum'
        suffix = 'unique' if unique else 'total'
        df_month_agg_dyn = tools_DF.my_agg(df_month_patched, ['mswhen', 'auxvalue'], ['msval'], [agg], order_idx=0,ascending=True)
        df_month_agg_dyn['mswhen'] = df_month_agg_dyn['mswhen'].dt.strftime('%Y-%m-%d')

        filename_out = '%s_%s.gif' % (msname, suffix)

        tools_IO.remove_files(self.folder_out, '*.png')
        #self.UP.plot_dynamics_histo(df_month_agg_dyn, col_time='mswhen', col_label='auxvalue', col_value='msval', n_tops=n_tops,n_extra=12)
        #tools_animation.folder_to_animated_gif_imageio(self.folder_out, self.folder_out + filename_out,framerate=6)
        return

# ---------------------------------------------------------------------------------------------------------------------
    def pivot_user_visits(self, df_month, df_week, df_day, col_label, col_userid,date_start=None,date_stop=None):

        df_unified = self.unify_records(df_month, df_week, df_day, level='month',date_start=date_start, date_stop=date_stop)
        df_res = tools_DF.my_agg(df_unified, [col_userid, col_label], ['msval'], ['sum'], order_idx=1,ascending=False)
        df_res = tools_DF.to_multi_column(df_res,idx_time=0,idx_label=1,idx_value=2,order_by_value=True)
        return df_res

# ---------------------------------------------------------------------------------------------------------------------
    def stat_test_is_same_distribution(self, df_ref0, df_insp0):

        columns = df_ref0.columns

        df0 = pd.Series(numpy.unique(df_ref0.iloc[:,0].values.tolist()+df_insp0.iloc[:, 0].values.tolist()),name=columns[0],index=None)

        df_ref = pd.merge(df0, df_ref0, how='left', on=[columns[0]])
        df_ref.fillna(0, inplace=True)

        df_insp = pd.merge(df0, df_insp0, how='left', on=[columns[0]])
        df_insp.fillna(0, inplace=True)

        dct_res = {}
        dct_res['method'] = '-'
        dct_res['p_value'] = numpy.nan

        if df_insp.iloc[:,1].sum()>0:
            if (df_insp.shape[0] == df_ref.shape[0] == 1):
                dct_res['method'] = '-'
                dct_res['p_value'] = 1.0
            elif (df_insp.iloc[:,1].values.shape[0] == df_ref.iloc[:,1].values.shape[0] == 2) and (df_ref.iloc[:,1].min()<=5) and (df_insp.iloc[:,1].min()<=5):
                 dct_res['method'] = 'Fisher'
                 dct_res['p_value'] = self.HT.is_same_distribution_fisher(df_insp.iloc[:, 1].values, df_ref.iloc[:, 1].values)[1]
            else:
                dct_res['method'] = 'Chi2'
                dct_res['p_value'] = self.HT.is_same_distribution_chi2(df_insp.iloc[:, 1].values, df_ref.iloc[:, 1].values)[1]

        return dct_res
# ---------------------------------------------------------------------------------------------------------------------