import datetime
import cv2
import pandas as pd
import numpy
from scipy.interpolate import make_interp_spline
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_DF
import tools_draw_numpy
import tools_plot_v2
import tools_time_convertor
import tools_animation
# ---------------------------------------------------------------------------------------------------------------------
class Plotter_dancing:

    def __init__(self,folder_out,dark_mode=False):
        self.folder_out = folder_out
        self.P = tools_plot_v2.Plotter(folder_out, dark_mode=dark_mode)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def interpolate_DF(self, df, col_time0, col_label, col_value, col_rank, n_extra, k=1):
        # if n_extra==0:
        #     return df

        labels = numpy.unique(df[col_label].to_numpy())
        times_orig = numpy.sort(numpy.unique(df[col_time0].to_numpy()))
        dct_time_to_id = dict((t, i) for i, t in enumerate(times_orig))
        dct_id_to_time = dict((i, t) for i, t in enumerate(times_orig))

        times = numpy.array([dct_time_to_id[t] for t in times_orig])

        col_time = 'new_time'
        df[col_time] = df[col_time0].map(dct_time_to_id, )

        times_interp = numpy.array(
            [(numpy.linspace(times[t], times[t + 1], n_extra)[:-1]) for t in range(times.shape[0] - 1)]).flatten()
        times_interp = numpy.insert(times_interp, times_interp.shape[0], times[-1])

        df_time = pd.Series(times, name=col_time)
        df_i_res_all = pd.DataFrame()

        for label in labels:
            df_f = df[df[col_label] == label]
            df_f = pd.merge(df_time, df_f, on=col_time, how='left')
            df_i_local = pd.DataFrame({'frameID': numpy.arange(0, times_interp.shape[0]), col_time: times_interp})

            for c in [col_value, col_rank]:
                values = df_f[c].to_numpy()
                idx = ~numpy.isnan(values)
                if (idx * 1).sum() >= 2:
                    spl = make_interp_spline(times[idx], values[idx], k=k)
                    values_i = spl(times_interp)
                else:
                    values_i = numpy.full(times_interp.shape[0], numpy.nan)

                spl_flag = make_interp_spline(times, 1 * idx, k=1)
                idx_i = spl_flag(times_interp)

                values_i[idx_i < 1] = numpy.nan

                for i, idx in enumerate(numpy.arange(0, values_i.shape[0] + 1, n_extra - 1)):
                    values_i[idx] = values[i]

                df_i = pd.DataFrame({col_time: times_interp, c: values_i})
                df_i_local = pd.merge(df_i_local, df_i, on=col_time, how='left')

            df_i_local[col_label] = label
            df_i_local['frameID'] = numpy.arange(0, df_i_local.shape[0])
            df_i_res_all = pd.concat([df_i_res_all,df_i_local], ignore_index=True)

        df_i_res_all[col_time] = df_i_res_all[col_time].apply(lambda x: dct_id_to_time[int(x)])
        df_i_res_all.rename(columns={col_time: col_time0}, inplace=True)

        return df_i_res_all
# ---------------------------------------------------------------------------------------------------------------------
    def __get_dynamics_df(self, df, col_time, col_label, col_value, pos_now=-1, n_last=1, keep_n_largest=3):

        times = numpy.sort(numpy.unique(df[col_time].to_numpy()))

        df_now = df[df[col_time] == times[pos_now]][[col_label, col_value]]
        df_now.rename(columns={col_value: 'now'}, inplace=True)
        df_prev = df[df[col_time] == times[pos_now - n_last]][[col_label, col_value]]
        df_prev.rename(columns={col_value: 'before'}, inplace=True)
        df_res = pd.merge(df_now, df_prev, on=col_label, how='left')

        df_res['delta'] = (df_res['now'] - df_res['before']) / df_res['before']
        df_res.rename(columns={col_label: 'label'}, inplace=True)
        df_res['max_delta'] = df_res['delta'].max()

        return df_res
# ---------------------------------------------------------------------------------------------------------------------
    def get_dynamics_data(self, dataframe, col_time, col_label, col_value, n_last=1,keep_n_largest=1000):
        df = pd.read_csv(dataframe) if isinstance(dataframe,str) else dataframe

        times = numpy.sort(numpy.unique(df[col_time].to_numpy()))
        df_res_all = pd.DataFrame()

        for pos_now in range(0,times.shape[0]):
            df_res = self.__get_dynamics_df(df, col_time, col_label, col_value, pos_now=pos_now, n_last=n_last, keep_n_largest=keep_n_largest)
            df_res['time'] = times[pos_now]
            df_res['rank'] = tools_IO.rank(df_res['now'].values)
            df_res_all = pd.concat([df_res_all,df_res], ignore_index=True)

        df_res_all.rename(columns={'now': 'value'}, inplace=True)
        df_res_all.sort_values(by=['time','rank'], inplace=True)
        df_res_all = df_res_all[['time','rank','label','value','before','delta','max_delta']]
        return df_res_all
# ---------------------------------------------------------------------------------------------------------------------
    def plot_values_pos(self, df, col_value, col_label, col_rank='rank',max_rank=None,max_value=None,xticks=None,yticks=None,out_format_x=None, colors=None, alpha=0.0,legend=None):

        X = numpy.insert(numpy.unique(df[col_value].to_numpy()), 0, 0)

        labels, tops,bottoms= [],[],[]

        if col_rank in df.columns.to_numpy():
            ranks = df[col_rank].to_numpy()
        else:
            ranks = 1+numpy.arange(0,df.shape[0],1)[::-1]

        if max_rank is None:
            max_rank = numpy.nanmax(ranks)

        for j in range(df.shape[0]):
            if ranks[j]>=max_rank:continue
            bottom = numpy.full(X.shape[0], numpy.nan)
            top = numpy.full(X.shape[0],numpy.nan)
            bottom[:numpy.where(X == df[col_value].iloc[j])[0][0] + 1] = ranks[j] - 0.25
            top[:numpy.where(X == df[col_value].iloc[j])[0][0] + 1] = ranks[j] + 0.25
            tops.append(top)
            bottoms.append(bottom)
            labels.append(df[col_label].iloc[j])





        if max_value is None or numpy.isnan(max_value):
            xlim = None
        else:
            xlim = (0, max_value)

        ylim = (1-0.25,max_rank+0.25)
        y_label_min_max = (1-0.25,max_rank+0.25)

        colors_f = colors if colors is not None else None
        image = self.plot_TS_plots(X, labels, numpy.array(tops), numpy.array(bottoms),colors=colors_f,
                                                 blacksigns=True, alpha=alpha, labels_pos='left', align='left',
                                                invert_y=True,xticks=xticks, yticks=yticks, xlim=xlim,ylim=ylim,out_format_x=out_format_x,
                                                 y_label_min_max=y_label_min_max,legend=legend)

        return [image]
# ---------------------------------------------------------------------------------------------------------------------
    def enrich_by_rank(self, df, col_time, col_value,col_rank):

        df_res = pd.DataFrame()
        for time in numpy.sort(numpy.unique(df[col_time].to_numpy())):
            df_time = df[df[col_time]==time].copy()
            v = -df_time[col_value].values
            order = v.argsort()
            ranks = 1+order.argsort()
            df_time[col_rank] = ranks
            #df_res = df_res.append(df_time, ignore_index=True)
            df_res = pd.concat([df_res,df_time], ignore_index=True)

        return df_res
# ---------------------------------------------------------------------------------------------------------------------
    def fetch_start_stop(self,df_start,df_stop,col_time,col_label,col_value,col_rank,max_rank):

        dct = dict(zip([c for c in df_start.columns], [t for t in df_start.dtypes]))

        for index, row in df_start.iterrows():
            label = row[col_label]
            if label not in df_stop[col_label].values:
                row[col_rank]=max_rank+1
                row[col_value] = 0
                row[col_time] = df_stop[col_time].iloc[0]
                #df_stop0 = df_stop.append(row, ignore_index=True)
                df_row = pd.DataFrame(row).T.astype(dct)
                df_stop = pd.concat([df_stop,df_row], ignore_index=True,axis=0)


        for index, row in df_stop.iterrows():
            label = row[col_label]
            if label not in df_start[col_label].values:
                row[col_rank]=max_rank+1
                row[col_value] = 0
                row[col_time] = df_start[col_time].iloc[0]
                #df_start = df_start.append(row, ignore_index=True)
                df_row = pd.DataFrame(row).T.astype(dct)
                df_start = pd.concat([df_start,df_row], ignore_index=True,axis=0)

        df_res = pd.concat([df_start, df_stop], ignore_index=True, axis=0)
        #df_res[col_time]=df_res[col_time].astype(numpy.datetime)
        #df_res[col_time] = pd.to_datetime(df_res[col_time])

        return df_res
# ---------------------------------------------------------------------------------------------------------------------
    def plot_dynamics_histo(self, df0, col_time,col_label,col_value,in_format_x=None,out_format_x=None,n_tops=5, n_extra=12):

        self.init_layout(name='960_670_large_right_pad')

        col_rank = 'rank'
        df0 = self.enrich_by_rank(df0, col_time, col_value, col_rank)
        df0[col_time] = tools_time_convertor.str_to_datetime(df0[col_time], format=in_format_x)
        times = pd.Series(numpy.sort(numpy.unique(df0[col_time].values)))

        max_value = df0[col_value].max()

        for pos_now in numpy.arange(0, times.shape[0] - 1):
            ii=0
            df_start = df0[df0[col_time] == times[pos_now]  ].copy()
            df_stop  = df0[df0[col_time] == times[pos_now+1]].copy()

            df_start = df_start[df_start[col_rank] <= n_tops]
            df_stop  = df_stop [df_stop [col_rank] <= n_tops]

            df_e = self.fetch_start_stop(df_start,df_stop,col_time,col_label,col_value,col_rank,n_tops)
            #df_e.to_csv(self.folder_out + 'df_e.txt', index=False)
            df_i = self.interpolate_DF(df_e,col_time,col_label,col_value,col_rank,n_extra, k=1)
            #df_i.to_csv(self.folder_out + 'df_i.txt', index=False)

            legend = tools_time_convertor.datetime_to_str(times[pos_now],out_format_x)

            for frame in numpy.arange(0,n_extra):
                image = self.plot_values_pos(df_i[df_i['frameID']==frame],col_value, col_label, col_rank, max_rank=n_tops,max_value=max_value,alpha=0.3,legend=legend)[0]
                cv2.imwrite(self.folder_out + 'Dynamics_{time}_{frame}.png'.format(time=legend,frame='%04d'%(1+frame)),image)

                if (pos_now==(times.shape[0] - 2) and (frame==(n_extra-1))):
                    legend = tools_time_convertor.datetime_to_str(times.iloc[-1], out_format_x)
                    image = self.plot_values_pos(df_i[df_i['frameID'] == frame], col_value, col_label, col_rank,max_rank=n_tops, max_value=max_value, alpha=0.3, legend=legend)[0]
                    cv2.imwrite(self.folder_out + 'Dynamics_{time}_{frame}.png'.format(time=legend, frame='%04d' % (1)), image)
            print(legend)

        tools_animation.folder_to_animated_gif_imageio(self.folder_out, self.folder_out+'animation.gif', mask='*.png', framerate=8,stop_ms=3000)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def init_layout(self,name):

        if name=='960_670_large_right_pad':
            self.W, self.H = 960, 670
            self.pad_left, self.pad_bottom, self.pad_right, self.pad_top = 0, 0, 0.70, 1
            self.margin_top, self.margin_bottom = 17, 42
            self.margin_left, self.margin_right = 23, 23

        if name == '960_670':
            self.W, self.H = 960, 670
            self.pad_left, self.pad_bottom, self.pad_right, self.pad_top = 0, 0, 1, 1
            self.margin_top, self.margin_bottom = 44, 70
            self.margin_left, self.margin_right = 60, 60

        return
# ---------------------------------------------------------------------------------------------------------------------
    def __distance(self,x1,x2):
        if not isinstance(x1,str):
            res = x1-x2
        else:
            datetime_object1 = datetime.datetime.strptime(x1, '%Y-%m-%d')
            datetime_object2 = datetime.datetime.strptime(x2, '%Y-%m-%d')
            res = (datetime_object1-datetime_object2).days

        return res
# ---------------------------------------------------------------------------------------------------------------------
    def __get_labels_positions(self, X, labels, values,values_min_max=None,labels_pos='right',invert_y=False):

        pos_lines_x, pos_lines_y,labels_line = [], [], []
        scatter_x, scatter_y, labels_scatter = [], [], []
        for l in range(len(labels)):
            label, value = labels[l], values[l]
            for i in range(1,X.shape[0]-1):
                if     numpy.isnan(value[i]):continue
                if labels_pos=='right':
                    if not numpy.isnan(value[i + 1]):continue
                    scatter_x.append(X[i])
                    scatter_y.append(value[i])
                    labels_scatter.append(label)
                if labels_pos=='left':
                    if not numpy.isnan(value[i - 1]): continue
                    scatter_x.append(X[i-1])
                    scatter_y.append(value[i-1])
                    labels_scatter.append(label)

            if (not numpy.isnan(value[0])) and (numpy.isnan(value[1]) or labels_pos=='left'):
                scatter_x.append(X[0])
                scatter_y.append(value[0])
                labels_scatter.append(label)

            if (not numpy.isnan(value[-1])) and labels_pos=='right':
                scatter_x.append(X[-1])
                scatter_y.append(value[-1])
                labels_scatter.append(label)

        if values_min_max is None:
            values_min_max = (float(numpy.nanmin(values.flatten())),float(numpy.nanmax(values.flatten())))

        bottom_pos, top_pos = values_min_max[0],values_min_max[1]
        pos_scatter_x = [self.margin_left + (self.W - self.margin_right - self.margin_left)*(self.__distance(p, X[0]) / self.__distance(X[-1], X[0])) for p in scatter_x]
        if invert_y is False:
            pos_lines_y = [self.margin_top + (top_pos - p) * (self.H - self.margin_top - self.margin_bottom) / (top_pos-bottom_pos) for p in pos_lines_y]
            pos_scatter_y = [self.margin_top + (top_pos - p) * (self.H - self.margin_top - self.margin_bottom) / (top_pos-bottom_pos) for p in scatter_y]
        else:
            pos_lines_y = [self.margin_top + (p-bottom_pos) * (self.H - self.margin_top - self.margin_bottom) / (top_pos-bottom_pos) for p in pos_lines_y]
            pos_scatter_y = [self.margin_top + (p-bottom_pos) * (self.H - self.margin_top - self.margin_bottom) / (top_pos-bottom_pos) for p in scatter_y]

        return labels_line, pos_lines_x,pos_lines_y,labels_scatter, scatter_x,scatter_y,pos_scatter_x,pos_scatter_y
# ---------------------------------------------------------------------------------------------------------------------
    def __plot_labels_positions(self,image, labels, positions_x,positions_y,colors=None,blacksigns=True,align='right',font_size=24):

        for l in range(len(labels)):
            label, position_x,position_y = labels[l], positions_x[l],positions_y[l]
            if numpy.isnan(position_x):continue
            if numpy.isnan(position_y):continue

            if colors is None:
                color = self.P.get_color(label)
            else:
                color = colors[l]

            if blacksigns:
                color = self.P.clr_font*255

            description = '%s' % label
            shift_x, shift_y,sx,sy = tools_draw_numpy.get_position_sign(sign=description,W=image.shape[1],H=image.shape[0],font_size=font_size)

            if align == 'right':
                align_x = - sx / 2 - 8
            elif align == 'left':
                align_x = sx / 2 + 8
            else:
                align_x = 0


            image = tools_draw_numpy.draw_text(image, description, (int(position_x +align_x+ shift_x), int(position_y + shift_y)),color, clr_bg=None, font_size=font_size)

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def plot_TS_plots(self, X, labels, tops, bottoms, colors=None, lw=1,linestyle=None, fill=True,alpha = 0.6, blacksigns=False,
                      labels_pos= 'right', align='right',xticks=None, yticks=None, xlim=None, ylim=None, y_label_min_max=None,
                      invert_y=False,major_step=None,out_format_x=None,legend=None,figsize=None):

        figsize = (self.W / 100, self.H / 100) if figsize is None else figsize
        fig = plt.figure(figsize=figsize,facecolor=self.P.clr_bg)
        fig = self.P.turn_light_mode(fig)

        labels_line, pos_lines_x, pos_lines_y, labels_scatter, scatter_x, scatter_y, pos_scatter_x, pos_scatter_y = \
            self.__get_labels_positions(X, labels, (tops + bottoms) / 2, values_min_max=y_label_min_max, labels_pos=labels_pos, invert_y=invert_y)

        for l in range(len(labels)):
            label, top, bottom = labels[l], tops[l], bottoms[l]
            color = self.P.get_color(label)[[2, 1, 0]] / 255 if colors is None else colors[l][[2, 1, 0]] / 255
            plt.plot(X,top,color=color, linewidth=lw,linestyle=linestyle,zorder=1)
            plt.plot(X,bottom, color=color, linewidth=lw,linestyle=linestyle,zorder=1)
            if fill:
                plt.fill_between(X, top, bottom, color=color * (1 - alpha) + self.P.clr_bg * alpha,zorder=0)

        if major_step is not None:
            xtick_labels_classic, idx_visible = self.P.get_xtick_labels(pd.DataFrame({'time':X,'label':0}),0,'%Y-%m-%d',major_step)
            xtick_labels, idx_visible = self.P.get_xtick_labels(pd.DataFrame({'time': X, 'label': 0}), 0, out_format_x,major_step)
            ax = plt.gca()
            ax.minorticks_on()
            if idx_visible is not None:
                ax.set_xticks(mdates.date2num(tools_time_convertor.str_to_datetime(xtick_labels_classic[idx_visible])))
                ax.set_xticklabels(xtick_labels[idx_visible])
        else:
            plt.xticks([])

        if isinstance(yticks,str) and yticks=='auto':
            plt.gca().tick_params(axis="y", direction="in",pad = -50)
        elif yticks is not None:
            plt.yticks(yticks)
            plt.gca().tick_params(axis="y", direction="in",pad = -10)
        else:
            plt.yticks([])

        if xlim is not None:
            plt.gca().set_xlim([xlim[0], xlim[1]])

        if ylim is not None:plt.gca().set_ylim([ylim[0], ylim[1]])
        if invert_y:plt.gca().invert_yaxis()


        plt.grid(color=self.P.clr_grid)
        image = self.P.get_image(fig, fig.get_facecolor())
        plt.close(fig)

        image = self.__plot_labels_positions(image, labels_scatter, pos_scatter_x, pos_scatter_y, colors=None,blacksigns=blacksigns, align=align)

        if legend is not None:
            image = numpy.concatenate([image,numpy.full((40,image.shape[1],3),255*self.P.clr_bg,dtype=numpy.uint8)],)
            image = self.__plot_labels_positions(image, [legend], [image.shape[1]//2],[image.shape[0]-30],colors=(255,255,255),align='center')

        return image
# =====================================================================================================================
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
        values = numpy.array([df_multicolumn[str(label)].sum() for label in labels])
        idx = numpy.array(numpy.argsort(-values))
        labels = labels[idx]
        labels=labels[:max_objects*2]
        labels=labels[::-1]
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

        df_MC = tools_DF.to_multi_column(df, idx_time=0, idx_label=1, idx_value=2)
        if ts_time is not None:
            df_MC = pd.merge(ts_time, df_MC, how='left', on=[df.columns[0]])

        if to_ratios:
            df_MC = self.to_ratios(df_MC, idx_time=0)

        X, labels, tops, bottoms = self.get_TS_one_frame(df_MC, top_objects, start_index=1, cumul=True)
        df2 = pd.DataFrame({'X':X})
        for label, ts in zip(labels, tops):
            df2[label] = ts

        idx_target = numpy.arange(1, df2.shape[1])[::-1]

        self.P.TS_seaborn(df2, idxs_target=idx_target.tolist(),idxs_fill=idx_target-1,idx_time=0,
                          out_format_x=out_format_x,major_step=major_step,transparency=0.25,
                          mode=mode, lw=1,remove_xticks=False,figsize=figsize,filename_out=filename_out)

        return
# =====================================================================================================================
    def plot_stacked_data(self, df, idx_time=0, idx_label=1, idx_value=2,top_objects=3,in_format_x=None, out_format_x=None,major_step=28,legend=None,alpha=0.6,figsize=(8, 6),filename_out=None):

        self.init_layout(name='960_670')
        df_MC = tools_DF.to_multi_column(df, idx_time=idx_time, idx_label=idx_label, idx_value=idx_value,replace_nan=True) #replace_nan=False

        X, labels, tops, bottoms = self.get_TS_one_frame(df_MC, top_objects=top_objects, start_index=1, cumul=True)
        X = tools_time_convertor.str_to_datetime(X,format=in_format_x).values

        image = self.plot_TS_plots(X, labels, tops, bottoms,y_label_min_max=[numpy.nanmin(bottoms),numpy.nanmax(tops)],
                                   yticks='auto',out_format_x=out_format_x,major_step=major_step,legend=legend,alpha=alpha,figsize=figsize)
        if filename_out is not None:
            cv2.imwrite(self.folder_out + filename_out, image)

        return image
# ---------------------------------------------------------------------------------------------------------------------
