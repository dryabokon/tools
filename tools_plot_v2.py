import cv2
import io
from mpl_toolkits.mplot3d import Axes3D
import numpy
import seaborn
import pandas as pd
import operator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib import colors
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA, TruncatedSVD
import squarify
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Plotter(object):
    def __init__(self,folder_out=None,dark_mode=False):
        self.folder_out = folder_out
        self.dark_mode = dark_mode
        self.init_colors()

        #self.dct_color = {}
        #self.colors = tools_draw_numpy.get_colors(256, colormap='jet',alpha_blend=0.2, clr_blend=(0, 0, 0),shuffle=False)
        #self.colors = tools_draw_numpy.get_colors(256, colormap='rainbow', alpha_blend=0.0,clr_blend=(0, 0, 0), shuffle=True)
        #self.colors = tools_draw_numpy.get_colors(256, colormap='tab10', alpha_blend=0.0,clr_blend=(0, 0, 0), shuffle=False)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_colors(self,cmap='tab20',shuffle=True):
        numpy.random.seed(111)

        self.colors = tools_draw_numpy.get_colors(32, colormap=cmap).astype(numpy.int)

        new_c = []
        for n in range(self.colors.shape[0]):
            color = self.colors[n]
            for alpha in numpy.linspace(0.3, 0.8, 7, endpoint=False):
                c = (1 - alpha) * numpy.array(color) + (alpha) * numpy.array((128, 128, 128))
                new_c.append(numpy.array(c).astype('int').reshape((1, 3)))

        self.colors = numpy.concatenate((self.colors, numpy.array(new_c).reshape((-1, 3))))

        if shuffle:
            idx = numpy.random.choice(self.colors.shape[0],self.colors.shape[0])
            self.colors=self.colors[idx]
        self.dct_color ={}
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_color(self, label,alpha_blend=0.0):
        if label not in self.dct_color:
            if isinstance(label, numpy.int32):
                n = label
            else:
                n = numpy.array([ord(l) for l in label]).sum() % self.colors.shape[0]
            self.dct_color[label] = self.colors[n]

        return self.dct_color[label]*(1-alpha_blend) + numpy.array((1,1,1))*(alpha_blend)
# ----------------------------------------------------------------------------------------------------------------------
    def set_color(self,label,value):
        self.dct_color[label] = value
        return
# ----------------------------------------------------------------------------------------------------------------------
    def turn_light_mode(self,fig,ax=None):

        seaborn.set_theme()
        plt.style.use('default')
        plt.rcParams.update({'figure.max_open_warning': 0})

        if ax is None:
            ax = plt.gca()
        ax.set_axisbelow(True)

        if self.dark_mode:
            seaborn.set(style="darkgrid")
            self.clr_bg = numpy.array((43, 43, 43)) / 255
            self.clr_grid = numpy.array((64, 64, 64)) / 255
            self.clr_font = 'white'
            #self.clr_border = 'darkgray'
            self.clr_border = self.clr_bg

        else:
            seaborn.set(style="whitegrid")
            seaborn.set_style("whitegrid")
            self.clr_bg = numpy.array((1, 1, 1))
            self.clr_grid = 'lightgray'
            self.clr_font = 'black'
            self.clr_border = 'gray'

        seaborn.set_style('ticks', {'axes.edgecolor': self.clr_border, 'xtick.color': self.clr_font, 'ytick.color': self.clr_font,'ztick.color': self.clr_font})
        fig.patch.set_facecolor(self.clr_bg)
        ax.set_facecolor(self.clr_bg)

        ax.spines['bottom'].set_color(self.clr_border)
        ax.spines['top'].set_color(self.clr_border)
        ax.spines['left'].set_color(self.clr_border)
        ax.spines['right'].set_color(self.clr_border)
        ax.xaxis.label.set_color(self.clr_font)
        ax.yaxis.label.set_color(self.clr_font)

        ax.tick_params(which='both', color=self.clr_font)
        ax.tick_params(axis='x', colors=self.clr_font)
        ax.tick_params(axis='y', colors=self.clr_font)

        try:
            ax.zaxis.label.set_color(self.clr_font)
            ax.tick_params(axis='z', colors=self.clr_font)
            ax.w_xaxis.set_pane_color(self.clr_grid)
            ax.w_yaxis.set_pane_color(self.clr_grid)
            ax.w_zaxis.set_pane_color(self.clr_grid)

        except:
            pass


        return fig
# ----------------------------------------------------------------------------------------------------------------------

    def recolor_legend_seaborn(self,legend):
        legend._legend_title_box._text.set_color(self.clr_font)
        for text in legend.get_texts(): text.set_color(self.clr_font)
        frame = legend.get_frame()
        frame.set_facecolor(self.clr_bg)
        frame.set_edgecolor(self.clr_border)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def recolor_legend_plt(self,legend):
        for text in legend.get_texts(): text.set_color(self.clr_font)
        frame = legend.get_frame()
        frame.set_facecolor(self.clr_bg)
        frame.set_edgecolor(self.clr_border)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def empty(self,figsize=(3.5,3.5)):
        fig = plt.figure(figsize=figsize)
        self.turn_light_mode()

        plt.plot(0, 0)
        plt.grid(color=self.clr_grid)
        plt.tight_layout()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_image(self, image):
        fig = plt.figure()
        self.turn_light_mode()

        plt.imshow(image[:, :, [2, 1, 0]])
        plt.axis('off')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_regression_df(self,df, idx_target, idx_num, idx_cat, filename_out=None):
        fig = plt.figure()
        self.turn_light_mode()

        columns = df.columns.to_numpy()
        name_num, name_cat, name_target = columns[[idx_num, idx_cat, idx_target]]
        seaborn.lmplot(x=name_num, y=name_target, col=name_cat, hue=name_cat, data=df, y_jitter=.02, logistic=True,truncate=False)
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_regression_YX(self,Y, X,logistic=False,filename_out=None):
        fig = plt.figure()
        fig = self.turn_light_mode(fig)

        columns = ['Y', 'X']
        A = numpy.hstack((Y.reshape((-1,1)), X.reshape((-1,1))))
        df = pd.DataFrame(data=A, columns=columns)

        seaborn.lmplot(data=df, x=columns[1], y=columns[0], y_jitter=.02, logistic=logistic, truncate=False)
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_hist(self,X,bins=None,x_range=None,figsize=(3.5,3.5),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        plt.hist(X,bins)

        plt.grid(color=self.clr_grid,which='major')
        if x_range is not None:
            plt.xlim(x_range)

        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_1D_features_pos_neg(self,X, Y, labels=True, bins=None,colors=None,palette='tab10',xlim=None,filename_out=None):

        fig = plt.figure()
        fig = self.turn_light_mode(fig)

        patches = []

        plt.xlim([-1, X.max()+1])

        if colors is None:
            colors = seaborn.color_palette(palette)

        for i,y in enumerate(numpy.unique(Y)):
            col = colors[0] if int(y)<=0 else colors[1]
            plt.hist(X[Y==y], bins=bins,color=col,alpha=0.7,width=1.0,align='mid',density=False)

            if labels:
                patches.append(mpatches.Patch(color=col,label=y))

        plt.grid(color=self.clr_grid)
        if xlim is not None:
            plt.xlim(xmin=xlim[0], xmax=xlim[1])

        if labels:
            plt.legend(handles=patches)

        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_2D_features(self, df, x_range=None, y_range=None, add_noice=False, remove_legend=False, transparency=0.0, colors=None,palette='tab10', figsize=(8, 6), filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        if add_noice:
            noice = 0.05 - 0.1 * numpy.random.random_sample(df.shape)
            df.iloc[:,1:]+= noice[:,1:]

        my_pal = colors if colors is not None else palette
        J = seaborn.scatterplot(data=df,x=df.columns[1],y=df.columns[2],hue=df.columns[0],palette=my_pal,edgecolor=None,alpha=1-transparency)

        plt.grid(color=self.clr_grid)
        if remove_legend:
            plt.legend([], [], frameon=False)
        else:
            legend = J._get_patches_for_fill.axes.legend_
            self.recolor_legend_seaborn(legend)

        if x_range is not None:plt.xlim(x_range)
        if y_range is not None:plt.ylim(y_range)

        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())

        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def plot_2D_features_cumul(self, df, figsize=(3.5,3.5),remove_legend=False,filename_out=None):
        def max_element_by_value(dct):return max(dct.items(), key=operator.itemgetter(1))


        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        X,Y = tools_DF.df_to_XY(df,idx_target=0)

        dict_pos,dict_neg ={},{}
        for x in X[Y>0]:
            if tuple(x) not in dict_pos:dict_pos[tuple(x)]=1
            else:dict_pos[tuple(x)]+=1

        for x in X[Y<=0]:
            if tuple(x) not in dict_neg:dict_neg[tuple(x)]=1
            else:dict_neg[tuple(x)]+=1


        col_neg  = (0, 0.5, 1, 1)
        col_pos  = (1, 0.5, 0, 1)
        col_gray = (0.5, 0.5, 0.5, 1)

        min_size = 4
        max_size = 20
        norm = max(max_element_by_value(dict_pos)[1], max_element_by_value(dict_neg)[1]) / max_size

        for x in dict_pos.keys():
            if tuple(x) not in dict_neg.keys():
                plt.plot(x[0], x[1], 'ro', color=col_pos, markeredgewidth=0, markersize=max(min_size, dict_pos[tuple(x)] / norm))
            else:
                plt.plot(x[0], x[1], 'ro', color=col_gray, markeredgewidth=0, markersize=max(min_size, (dict_pos[tuple(x)] + dict_neg[tuple(x)]) / norm))

                if dict_pos[tuple(x)] < dict_neg[tuple(x)]:
                    plt.plot(x[0], x[1], 'ro', color=col_neg, markeredgewidth=0, markersize=max(min_size, (dict_neg[tuple(x)] - dict_pos[tuple(x)]) / norm))
                else:
                    plt.plot(x[0], x[1], 'ro', color=col_pos, markeredgewidth=0, markersize=max(min_size, (-dict_neg[tuple(x)] + dict_pos[tuple(x)]) / norm))

        for x in dict_neg:
            if tuple(x) not in dict_pos:
                plt.plot(x[0], x[1], 'ro', color=col_neg, markeredgewidth=0, markersize=max(min_size, dict_neg[tuple(x)] / norm))

        plt.grid(color=self.clr_grid)
        if remove_legend:
            plt.legend([], [], frameon=False)

        plt.xlabel(df.columns[1])
        plt.ylabel(df.columns[1])

        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())
        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def plot_2D_features_in_3D(self, df, x_range=None, y_range=None, add_noice=False, remove_legend=False,transparency=0.0, palette='tab10', figsize=(6, 6), filename_out=None):

        fig = plt.figure(figsize=figsize)

        if add_noice:
            noice = 0.05 - 0.1 * numpy.random.random_sample(df.shape)
            df.iloc[:,1:]+= noice[:,1:]

        ax = Axes3D(fig)
        fig.add_axes(ax,rect=[0, 0, 1, 1])
        fig = self.turn_light_mode(fig,ax)

        sc = ax.scatter(df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 0], s=40, c=df.iloc[:, 0], marker='o', cmap=palette, alpha=1)
        ax.set_xlabel(df.columns[1])
        ax.set_ylabel(df.columns[2])
        ax.set_zlabel(df.columns[0])

        if remove_legend:
            plt.legend([], [], frameon=False)

        if x_range is not None:plt.xlim(x_range)
        if y_range is not None:plt.ylim(y_range)

        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

        #plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())

        return fig
#-----------------------------------------------------------------------------------------------------------------------
    def plot_tp_fp(self,tpr,fpr,roc_auc,caption='',figsize=(3.5,3.5),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)
        color = seaborn.color_palette(palette='tab10', n_colors=1)
        plt.grid(which='major', color=self.clr_grid, linestyle='--')
        plt.plot(fpr, tpr, color=color[0], lw=3, label='%s %.2f' % (caption, roc_auc))
        plt.plot([0, 1.05], [0, 1.05], color='lightgray', lw=1, linestyle='--')

        legend = plt.legend(loc='lower right')
        self.recolor_legend_plt(legend)

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        plt.clf()
        plt.close(fig)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_PR(self, precisions,recalls, mAP, caption='', figsize=(3.5, 3.5), filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        idx =numpy.argmax([p * r / (p + r + 1e-4) for p, r in zip(precisions, recalls)])

        colors = seaborn.color_palette(palette='tab10', n_colors=1)
        plt.grid(which='major', color=self.clr_grid, linestyle='--')
        plt.plot(recalls, precisions, color=colors[0], lw=3, label='%s %.2f' % (caption, mAP))
        plt.scatter(recalls[idx], precisions[idx], color=colors[0])

        precision_random = precisions[recalls==1].max()
        plt.plot([0, 1], [precision_random, precision_random], color='lightgray', lw=1, linestyle='--')

        legend = plt.legend(loc='upper left')
        self.recolor_legend_plt(legend)

        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())

        plt.clf()
        plt.close(fig)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_TP_FP_PCA_scatter(self,scores, X,Y,filename_out=None):

        precision, recall, thresholds = metrics.precision_recall_curve(Y, scores)

        th = thresholds[numpy.argmax([p * r / (p + r + 1e-4) for p, r in zip(precision, recall)])]
        indicator = numpy.full(Y.shape[0], '--', dtype=numpy.chararray)

        indicator[(scores < th) * (Y <= 0)] = 'Hit TN'
        indicator[(scores >= th) * (Y > 0)] = '   Hit TP'
        indicator[(scores >= th) * (Y <= 0)] = '  FP'
        indicator[(scores < th) * (Y > 0)] = ' Miss'

        colors = [plt.get_cmap('tab10')(i) for i in [7,3,1,2]]

        X_PCA = TSNE(n_components=2).fit_transform(X)
        df_PCA = pd.DataFrame({'Y': indicator, 'x1': X_PCA[:, 0], 'x2': X_PCA[:, 1]})
        df_PCA.sort_values(by=df_PCA.columns[0], ascending=False, inplace=True)
        # df_PCA.to_csv(self.folder_out+'df_PCA.csv',index=False)
        self.plot_2D_features(df_PCA, remove_legend=False, transparency=0.5, colors=colors,filename_out=filename_out)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_TP_FP_rectangles(self,scores, X,Y,filename_out=None):

        fig = plt.figure()
        fig = self.turn_light_mode(fig)
        ax = plt.gca()

        precisions, recalls, thresholds = metrics.precision_recall_curve(Y, scores)
        th = thresholds[numpy.argmax([p * r / (p + r + 1e-4) for p, r in zip(precisions, recalls)])]

        n_hits_TN = ((scores < th)  * (Y <= 0)).sum()
        n_FPs =     ((scores >= th) * (Y <= 0)).sum()
        n_miss =    ((scores < th)  * (Y > 0)).sum()
        n_TPs =     ((scores >= th) * (Y > 0)).sum()
        #colors = [plt.get_cmap('tab10')(i) for i in [7, 3, 1, 2]]   #gray,red,amber,green
        colors = [plt.get_cmap('tab20')(i) for i in [15, 6, 2, 4]]   #gray,red,amber,green

        yyy =(Y>0).sum()/Y.shape[0]
        xxx1 = n_miss/(n_miss+n_TPs)        #1-recalls[th]
        xxx2 = min(0.995,n_hits_TN/(n_hits_TN+n_FPs))

        ax.add_patch(mpatches.Rectangle((0, yyy), xxx2, 1-yyy,facecolor=colors[0],edgecolor=None,fill=True,lw=1,label='hits_TN'))
        ax.add_patch(mpatches.Rectangle((xxx2, yyy), 1-xxx2, 1 - yyy, facecolor=colors[2], edgecolor=None, fill=True, lw=1,label='FP'))
        ax.add_patch(mpatches.Rectangle((0, 0)   ,xxx1,   yyy,facecolor=colors[1],edgecolor=None,fill=True, lw=1))
        ax.add_patch(mpatches.Rectangle((xxx1,0), 1-xxx1, yyy, facecolor=colors[3], edgecolor=None, fill=True, lw=1))

        plt.tight_layout()
        if filename_out is not None:

            image = self.get_image(fig, self.clr_bg)
            image = tools_draw_numpy.draw_text(image, 'Recall=%.2f' % (n_TPs / (n_miss + n_TPs)), (100,50), color_fg=0, clr_bg=self.clr_bg, font_size=24)
            image = tools_draw_numpy.draw_text(image, 'Precision=%.2f' % (n_TPs / (n_FPs + n_TPs)), (100,80),color_fg=0, clr_bg=self.clr_bg, font_size=24)
            image = tools_draw_numpy.draw_text(image, 'FP rate=%.2f' % (n_FPs / (n_FPs + n_TPs)), (100, 110),color_fg=255*numpy.array(colors[2])[[2,1,0]], clr_bg=self.clr_bg, font_size=24)
            image = tools_draw_numpy.draw_text(image, 'Miss rate=%.2f' % (n_miss / (n_miss + n_TPs)), (100, 140),color_fg=255*numpy.array(colors[1])[[2,1,0]], clr_bg=self.clr_bg, font_size=24)

            image = tools_draw_numpy.draw_text(image, '# Pos %d' % ((Y>0).sum()), (300, 50),color_fg=0, clr_bg=self.clr_bg,font_size=24)
            image = tools_draw_numpy.draw_text(image, '# Neg %d' % ((Y<=0).sum()), (300, 80), color_fg=0,clr_bg=self.clr_bg, font_size=24)
            image = tools_draw_numpy.draw_text(image, '# FPs %d' % n_FPs, (300, 110),color_fg=255 * numpy.array(colors[2])[[2, 1, 0]], clr_bg=self.clr_bg,font_size=24)
            image = tools_draw_numpy.draw_text(image, '# Misses %d' % n_miss , (300, 140),color_fg=255 * numpy.array(colors[1])[[2, 1, 0]], clr_bg=self.clr_bg,font_size=24)
            image = tools_draw_numpy.draw_text(image, '# Hits %d' % n_TPs, (300, 170),color_fg=255 * numpy.array(colors[3])[[2, 1, 0]], clr_bg=self.clr_bg,font_size=24)
            cv2.imwrite(self.folder_out + filename_out,image)
            #plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())


        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_fact_predict(self,y_fact,y_pred,figsize=(3.5,3.5),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)
        pal = 'tab10'
        color = seaborn.color_palette(palette=pal, n_colors=1)

        plt.grid(which='major', color=self.clr_grid, linestyle='--')
        interval = [min(numpy.min(y_fact),numpy.min(y_pred)),max(numpy.max(y_fact),numpy.max(y_pred))]
        plt.plot(interval, interval, color='gray', lw=1, linestyle='--',zorder=0)
        #MAPE = numpy.mean(numpy.abs((y_fact - y_pred) / y_fact)) * 100
        #RMSE = 100*numpy.sqrt((((y_fact-interval[0])/(interval[1]-interval[0])-(y_pred-interval[0])/(interval[1]-interval[0]))**2).mean())
        R2 = r2_score(y_fact,y_pred)
        plt.scatter(y_fact, y_pred,color=color[0],edgecolors=self.clr_border,marker='o',label='R2 = %.2f'%R2,zorder=1)
        legend = plt.legend(loc="lower right")
        self.recolor_legend_plt(legend)
        plt.tight_layout()

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())
        return
# ----------------------------------------------------------------------------------------------------------------------
    def jointplots_df(self,df0, idx_target=0,transparency=0,palette='tab10'):

        columns = df0.columns.to_numpy()
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)


        for i in range(len(idx)):
            c = columns[idx[i]]
            df = df0[[target, c]]
            df = df.dropna()
            df = tools_DF.hash_categoricals(df)

            fig = plt.figure()
            fig = self.turn_light_mode(fig)
            plt.grid(color=self.clr_grid)
            J = seaborn.histplot(data=df, x=c, hue=target, palette=palette,element='poly',legend=True)

            legend = J._get_patches_for_fill.axes.legend_
            self.recolor_legend_seaborn(legend)
            plt.savefig(self.folder_out + 'plot_%02d_%02d_%s.png' % (i, i,c),facecolor=fig.get_facecolor())
            plt.close()

        for i in range(len(idx)-1):
            for j in range(i+1,len(idx)):
                c1, c2 = columns[idx[i]], columns[idx[j]]
                df = df0[[target, c1, c2]]
                df = df.dropna()
                df = tools_DF.hash_categoricals(df)

                fig = plt.figure()
                fig = self.turn_light_mode(fig)
                J = seaborn.jointplot(data=df, x=c1, y=c2, hue=target,palette=palette,edgecolor=None,alpha=1-transparency)
                J.ax_joint.grid(color=self.clr_grid)
                J.ax_joint.set_facecolor(self.clr_bg)
                J.ax_marg_x.set_facecolor(self.clr_bg)
                J.ax_marg_y.set_facecolor(self.clr_bg)
                J.ax_joint.xaxis.label.set_color(self.clr_font)
                J.ax_joint.yaxis.label.set_color(self.clr_font)

                legend = J.ax_joint.legend()
                self.recolor_legend_plt(legend)

                I = int(10000 * mutual_info_classif(df.iloc[:, [1, 2]], df.iloc[:, 0]).sum())
                file_out = 'pairplot_%02d_%02d_%s_%s.png'%(i,j,c1,c2)
                plt.savefig(self.folder_out + file_out,facecolor=fig.get_facecolor())
                plt.close(fig)

                f_handle = open(self.folder_out + "descript.ion", "a+")
                f_handle.write("%s %s\n" % (file_out, '%03d' % I))
                f_handle.close()


        return
# ----------------------------------------------------------------------------------------------------------------------
    def pairplots_df(self,df0, idx_target=0,palette = 'tab10',transparency=0,cumul_mode=False,add_noise=False,mode2d=True):

        f_handle = open(self.folder_out + "descript.ion", "w+")
        f_handle.close()

        columns = df0.columns.to_numpy()
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

        if add_noise:
            transparency = 0.95

        for i in range(len(idx)-1):
            for j in range(i+1,len(idx)):
                c1, c2 = columns[idx[i]], columns[idx[j]]
                df = df0[[target,c1,c2]]
                df = df.dropna()
                df = tools_DF.hash_categoricals(df)
                I = int(10000*mutual_info_classif(df.iloc[:,[1, 2]], df.iloc[:,0]).sum())
                file_out = 'pairplot_%02d_%02d_%s_%s_%02d.png' % (i, j, c1, c2, I)
                if cumul_mode:
                    self.plot_2D_features_cumul(df, remove_legend=True,filename_out=file_out)
                else:
                    if mode2d:
                        self.plot_2D_features(df, add_noice=add_noise, transparency=transparency, remove_legend=True, palette=palette, filename_out=file_out)
                    else:
                        self.plot_2D_features_in_3D(df, add_noice=add_noise, transparency=transparency, remove_legend=True,palette=palette, filename_out=file_out)

                f_handle = open(self.folder_out + "descript.ion", "a+")
                f_handle.write("%s %s\n" % (file_out, '%03d'%I))
                f_handle.close()


        for i in range(len(idx)):
            c1 = columns[idx[i]]
            df = df0[[target, c1]]
            df = df.dropna()
            df = tools_DF.hash_categoricals(df)

            #bins = numpy.arange(-0.5, df[[c1]].max() + 0.5, 0.25)
            bins = None
            self.plot_1D_features_pos_neg(df[[c1]].to_numpy(), df[target].to_numpy(), labels=True, bins=bins,filename_out='plot_%02d_%02d_%s.png' % (i, i,c1))


        return
# ----------------------------------------------------------------------------------------------------------------------
    def pairplots_df_3d(self, df0, idx_target=0, palette='tab10',add_noise=False):

        f_handle = open(self.folder_out + "descript.ion", "w+")
        f_handle.close()

        columns = df0.columns.to_numpy()
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
        transparency = 0

        for i in range(len(idx) - 1):
            for j in range(i + 1, len(idx)):
                c1, c2 = columns[idx[i]], columns[idx[j]]
                df = df0[[target, c1, c2]]
                df = df.dropna()
                df = tools_DF.hash_categoricals(df)
                I = int(100 * mutual_info_classif(df.iloc[:, [1, 2]], df.iloc[:, 0]).sum())
                file_out = 'pairplot_%02d_%02d_%s_%s_%02d.png' % (i, j, c1, c2, I)

                if add_noise:
                    noice = 0.05 - 0.1 * numpy.random.random_sample(df.shape)
                    df.iloc[:, 1:] += noice[:, 1:]

                ax = plt.axes(projection='3d')
                ax.scatter(df.iloc[:,1], df.iloc[:,2], df.iloc[:,0], c=df.iloc[:,0], cmap=palette, linewidth=0.5)
                plt.savefig(self.folder_out + file_out)


                f_handle = open(self.folder_out + "descript.ion", "a+")
                f_handle.write("%s %s\n" % (file_out, '%03d' % I))
                f_handle.close()

        return

# ----------------------------------------------------------------------------------------------------------------------
    def plot_target_feature(self, df0, idx_target,figsize=(3.5,3.5)):

        columns = df0.columns.to_numpy()
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

        for i in idx:
            fig = plt.figure(figsize=figsize)
            fig = self.turn_light_mode(fig)
            seaborn.scatterplot(data=df0, x=columns[i], y=target,edgecolor=self.clr_border)
            plt.savefig(self.folder_out + '%s.png'%columns[i], facecolor=fig.get_facecolor())

        return

# ----------------------------------------------------------------------------------------------------------------------
    def TS_matplotlib(self, df, idxs_target, idx_time, colors=None, lw=1, idxs_fill=None, remove_legend=False, x_range=None, y_range=None, palette='tab10', figsize=(15, 3), filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        if idx_time is None:
            XX = numpy.linspace(0, df.shape[0] - 1, df.shape[0])
        else:
            XX = df.iloc[:, idx_time]

        if not isinstance(idxs_target, list): idxs_target = [idxs_target]

        if colors is None:
            colors = seaborn.color_palette(palette=palette, n_colors=len(idxs_target))

        patches = []

        for i, idx_target in enumerate(idxs_target):
            patches.append(mpatches.Patch(color=colors[i], label=df.columns[idx_target]))
            plt.plot(XX, df.iloc[:,idx_target],color = colors[i],linewidth=lw)

        if idxs_fill is not None:
            plt.fill_between(XX,df.iloc[:,idxs_fill[0]],df.iloc[:,idxs_fill[1]],color=self.clr_grid)

        if remove_legend:
            plt.legend([], [], frameon=False)
        else:
            legend = plt.legend(handles=patches,loc="upper left")
            self.recolor_legend_plt(legend)

        plt.grid(color=self.clr_grid, which='major')

        if x_range is not None:plt.xlim(x_range)
        #plt.rcParams.update({'xtick.color': self.clr_font, 'ytick.color': self.clr_font})

        #plt.xticks([])
        #plt.yticks([])
        plt.tight_layout()

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out, facecolor=fig.get_facecolor())
            plt.close()

        plt.close(fig)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_xtick_labels(self,df,idx_time,out_format_x,major_step):

        if idx_time is None:
            xtick_labels = numpy.arange(df.shape[0],dtype='int')
            idx_visible = numpy.arange(0,df.shape[0],step=int(major_step))
        elif 'int' in str(df[df.columns[idx_time]].dtypes):
            xtick_labels = df.iloc[:, idx_time].values
            idx_visible = numpy.arange(0, df.shape[0], step=int(major_step))
        elif 'datetime' in str(df[df.columns[idx_time]].dtypes):
            xtick_labels = df[df.columns[idx_time]].dt.strftime(out_format_x).values
            idx_visible = []
            values = df.iloc[:, idx_time].values
            last_visible = values[0]
            for i, x in enumerate(values):
                delta = ((x - last_visible) / numpy.timedelta64(1, 'D'))
                if i == 0 or (delta >= major_step):
                    idx_visible.append(i)
                    last_visible = x
        else:
            xtick_labels = None
            idx_visible = None

        return xtick_labels,idx_visible
# ----------------------------------------------------------------------------------------------------------------------
    def TS_seaborn(self, df, idxs_target, idx_time, idx_hue=None,mode='pointplot', idxs_fill=None,remove_legend=False, remove_xticks=True, x_range=None,
                   out_format_x=None, major_step=None, lw=2,transparency=0, figsize=(15, 3), filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)
        columns = [c for c in df.columns]


        X = columns[idx_time] if idx_time is not None else df.index
        if not isinstance(idxs_target, list):idxs_target = [idxs_target]
        colors = [self.get_color(columns[idx_target])[[2, 1, 0]] / 255 for idx_target in idxs_target]
        seaborn.set(style="ticks", rc={'lines.linewidth': lw,'lines.markersize': 1.0})
        hue = df.columns[idx_hue] if idx_hue is not None else None

        for i,idx_target in enumerate(idxs_target):
            if   mode == 'pointplot'  :g = seaborn.pointplot(data=df, x=X, y=df.columns[idx_target], scale=0.25,color=colors[i],markers='o', label=df.columns[idx_target],errwidth=4)
            elif mode == 'scatterplot':g = seaborn.scatterplot(data=df, x=X, y=df.columns[idx_target],size=numpy.full(df.shape[0],2.25),color=colors[i],alpha=1-transparency,edgecolor=None,markers='0',label=df.columns[idx_target])
            elif mode == 'barplot'    :g = seaborn.barplot(data=df,x=X, y=df.columns[idx_target],hue=hue,color=colors[i])
            else:                      g = seaborn.lineplot(data=df, x=X, y=df.columns[idx_target],color=colors[i],label=df.columns[idx_target])
            g.set_xlabel('')
            g.set_ylabel('')

        if idxs_fill is not None and mode=='pointplot':
            for idx_fill in idxs_fill:
                xy = g.lines[int(len(g.lines) / len(idxs_target) * idx_fill)].get_xydata()
                plt.fill_between(x=xy[:,0], y1=xy[:,1], color=colors[idx_fill],alpha=1-transparency)

            patches = [mpatches.Patch(color=(colors[i][0],colors[i][1],colors[i][2],1-transparency), label=columns[idx_target]) for i, idx_target in enumerate(idxs_target)]
        else:
            patches = [mpatches.Patch(color=colors[i], label=columns[idx_target]) for i, idx_target in enumerate(idxs_target)]

        plt.grid(color=self.clr_grid,which='major')

        if major_step is not None:
            xtick_labels,idx_visible = self.get_xtick_labels(df,idx_time,out_format_x,major_step)
            ax = plt.gca()
            ax.minorticks_on()
            if idx_visible is not None:
                ax.set_xticks(idx_visible)
                ax.set_xticklabels(xtick_labels[idx_visible])

        if x_range is not None:
            plt.xlim(x_range)
        elif remove_xticks:
            g.set(xticks=[])

        if remove_legend:
            plt.legend([], [], frameon=False)
        else:
            legend = plt.legend(handles=patches,loc="upper left")
            self.recolor_legend_plt(legend)

        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        return fig

# ----------------------------------------------------------------------------------------------------------------------

    def plot_tops_bottoms(self, X, tops, bottoms, colors, filleds, lw=1,alpha=0.6,xticks=None, yticks=None, xlim=None, ylim=None,invert_y=False,figsize=(15, 3),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)
        plt.rc_context({'font.size': 14})


        clr_bg = numpy.array(self.clr_bg)[[2, 1, 0]]

        if xticks is not None:
            plt.xticks(xticks)
        else:
            plt.xticks([])

        if isinstance(yticks, str) and yticks == 'auto':
            plt.gca().tick_params(axis="y", direction="in", pad=-50)
        elif yticks is not None:
            plt.yticks(yticks)
            plt.gca().tick_params(axis="y", direction="in", pad=-10)
        else:
            plt.yticks([])

        if xlim is not None: plt.gca().set_xlim([xlim[0], xlim[1]])
        if ylim is not None: plt.gca().set_ylim([ylim[0], ylim[1]])

        if invert_y: plt.gca().invert_yaxis()

        for top, bottom,color,filled in zip(tops,bottoms,colors,filleds):
            if filled == -1:
                linestyle = '--'
            else:
                linestyle ='-'

            color = color[[2,1,0]]/255
            plt.plot(X, top, color=color, linewidth=lw, zorder=1, ls=linestyle)
            plt.plot(X, bottom, color=color, linewidth=lw, zorder=1, ls=linestyle)
            if filled==1:
                plt.fill_between(X, top, bottom, color=color * (1 - alpha) + clr_bg * alpha, zorder=0)

        plt.tight_layout()
        plt.grid(color=self.clr_grid)
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out, facecolor=fig.get_facecolor())
            plt.close()

        plt.close(fig)
        return

# ----------------------------------------------------------------------------------------------------------------------
    def plot_TS_separatly(self, df, idx_target):
        df = tools_DF.hash_categoricals(df)

        for i, feature in enumerate(df.columns):
            #color = seaborn.color_palette(palette='Dark2')[0] if i == idx_target else None
            self.TS_matplotlib(df, idxs_target=[i], idx_time=None, filename_out='%s.png' % feature)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_tSNE(self,df, idx_target,palette='tab10',filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_TSNE = TSNE(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_TSNE), axis=1), columns=['tSNE', 'x0', 'x1'])
        #df = df.astype({'tSNE': 'int32'})
        df.sort_values(by=df.columns[0], inplace=True)
        self.plot_2D_features(df, remove_legend=True, palette=palette, transparency=0.3, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_PCA(self,df, idx_target,palette='tab10',filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_PCA = PCA(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_PCA), axis=1), columns=['PCA', 'x0', 'x1'])
        #df = df.astype({'PCA': 'int32'})
        df.sort_values(by=df.columns[0],inplace=True)
        self.plot_2D_features(df, remove_legend=True, palette=palette, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_UMAP(self,df, idx_target,palette='tab10',filename_out=None):
        import umap
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_UMAP = umap.UMAP(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_UMAP), axis=1), columns=['UMAP', 'x0', 'x1'])
        df = df.astype({'UMAP': 'int32'})
        self.plot_2D_features(df, remove_legend=True, palette=palette, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_SVD(self,df, idx_target,palette='tab10',filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_SVD = TruncatedSVD(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_SVD),axis=1),columns=['SVD','x0','x1'])
        df = df.astype({'SVD': 'int32'})
        self.plot_2D_features(df, remove_legend=True, palette=palette, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_LLE(self,df, idx_target,palette='tab10',filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_LLE = LocallyLinearEmbedding(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_LLE),axis=1),columns=['LLE','x0','x1'])
        df = df.astype({'LLE': 'int32'})
        self.plot_2D_features(df, remove_legend=True, palette=palette, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_ISOMAP(self,df, idx_target,palette='tab10',filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_ = Isomap(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_), axis=1), columns=['ISOMAP', 'x0', 'x1'])
        df = df.astype({'ISOMAP': 'int32'})
        self.plot_2D_features(df, remove_legend=True, palette=palette, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_contourf(self,X0,X1,xx, yy, grid_confidence,x_range=None,y_range=None,xlabel=None,ylabel=None,figsize=(3.5,3.5),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        map_RdBu = [cm.RdBu(i) for i in numpy.linspace(0, 1, 15)]
        map_PuOr = [cm.PuOr(i) for i in numpy.linspace(0, 1, 15)]
        my_cmap = map_RdBu[7:][::-1] + map_PuOr[:7][::-1]
        my_cmap = colors.ListedColormap(my_cmap)
        #my_cmap = cm.coolwarm

        plt.contourf(xx, yy, grid_confidence, cmap=my_cmap, alpha=.25)
        plt.plot(X0[:, 0], X0[:, 1], 'ro', color='#1F77B4', alpha=0.75)
        plt.plot(X1[:, 0], X1[:, 1], 'ro', color='#FF7F0E', alpha=0.75)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        if x_range is not None:plt.xlim(x_range)
        if y_range is not None:plt.ylim(y_range)

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        plt.close(fig)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_hor_bars(self,values,header,legend=None,figsize=(3.5,6),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        colors = seaborn.color_palette(palette='tab10', n_colors=1)

        y_pos = numpy.arange(len(header))
        idx = numpy.argsort(values)
        plt.barh(y_pos, values[idx],color=colors[0])
        plt.yticks(y_pos,header[idx])
        plt.tight_layout()
        plt.xticks([])

        if legend is not None:
            legend = plt.legend([legend],loc="lower right")
            self.recolor_legend_plt(legend)

        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out,facecolor=fig.get_facecolor())

        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def plot_pie(self,values,header,filename_out=None):

        fig = plt.figure()
        fig = self.turn_light_mode(fig)
        plt.pie(values,  labels=header, autopct='%1.1f%%',shadow=False, startangle=90)
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def plot_squarify(self,df, idx_label, idx_count, stat='%',idx_color=None, col255=None,alpha=0.0,palette='tab20',figsize=(15, 5),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        weights = numpy.array(df.iloc[:, idx_count], dtype=numpy.int)
        labels = numpy.array(df.iloc[:, idx_label], dtype=numpy.str)
        labels = numpy.array([label.replace(' ', '\n') for label in labels])
        if col255 is None:
            col255 = tools_draw_numpy.get_colors(256, shuffle=True, colormap=palette)


        if idx_color is not None:
            col_rank = numpy.array(df.iloc[:, idx_color], dtype=numpy.float32)
            col_rank -= col_rank.min()
            col_rank /= (col_rank.max() / 255)
            col_rank = col_rank.astype(int)
            colors = col255[col_rank]
        else:
            #colors = col255[numpy.linspace(0, 255, labels.shape[0]).astype(numpy.int)]
            colors = numpy.array([self.get_color(label,alpha_blend=alpha) for label in labels])

        colors = colors[:, [2, 1, 0]] / 255


        W = sum(weights)
        if stat=='%':
            labels2 = ['%s %.1f%%' % (l, float(100 * w / W)) for w, l in zip(weights, labels)]
        elif stat=='#':
            labels2 = ['%s %d' % (l, w) for w, l in zip(weights, labels)]
        else:
            labels2 = labels

        squarify.plot(sizes=weights, label=labels2, color=colors, pad=False,clr_fg=self.clr_font)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.folder_out + filename_out, facecolor=numpy.array(self.clr_bg))
        plt.clf()
        plt.close(fig)
        return

# ---------------------------------------------------------------------------------------------------------------------
    def get_image(self, fig, clr_bg):
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor(numpy.array(self.clr_bg)[[2, 1, 0]] / 255)

        plt.tight_layout()

        ax.tick_params(which='major', length=0)

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', facecolor=clr_bg)
        io_buf.seek(0)
        image = numpy.reshape(numpy.frombuffer(io_buf.getvalue(), dtype=numpy.uint8),newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:, :, [2, 1, 0]]
        io_buf.close()
        return image
# ----------------------------------------------------------------------------------------------------------------------
