import os
import cv2
import io
from mpl_toolkits.mplot3d import Axes3D
import numpy
import seaborn
import pandas as pd
import operator
import matplotlib
import math
matplotlib.use('Agg')
from matplotlib import pyplot as plt,colors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings( 'ignore', module = 'seaborn' )
warnings.filterwarnings( 'ignore', module = 'matplotlib')
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_draw_numpy
import tools_Hyptest
import tools_time_convertor
# ----------------------------------------------------------------------------------------------------------------------
class Plotter(object):
    def __init__(self,folder_out=None,dark_mode=False):
        self.folder_out = folder_out
        self.dark_mode = dark_mode
        self.init_base_colors()

        self.init_colors()
        #self.turn_light_mode(None)
        self.io_buf = io.BytesIO()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_base_colors(self):
        self.color_black = numpy.array((0, 0, 0))
        self.color_white = numpy.array((255, 255, 255))
        self.color_gray = numpy.array((64, 64, 64))
        self.color_light_gray = numpy.array((180, 180, 180))

        self.color_bright_red = numpy.array((0, 32, 255))
        self.color_red = numpy.array((22, 74, 223))
        self.color_amber = numpy.array((0, 128, 255))
        self.color_coral = numpy.array((0, 90, 255))

        self.color_aqua = numpy.array((180, 200, 0))
        self.color_diamond = numpy.array((120, 200, 0))
        self.color_green = numpy.array((0, 200, 0))
        self.color_grass = numpy.array((0, 200, 140))

        self.color_marsala = numpy.array((0, 0, 180))
        self.color_gold = numpy.array((0, 200, 255))
        self.color_grass_dark = numpy.array((63, 77, 73))
        self.color_grass_dark2 = numpy.array((97, 111, 107))
        self.color_blue = numpy.array((255, 128, 0))
        self.color_sky = numpy.array((173, 149, 83))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def set_font_size(self,font_size):
        plt.rc_context({'font.size': font_size})
        return
# ----------------------------------------------------------------------------------------------------------------------
    def set_label_size(self,label_size):
        plt.rc_context({'axes.labelsize':label_size})
        return
# ----------------------------------------------------------------------------------------------------------------------
    def turn_light_mode(self, fig, ax=None):

        seaborn.set_theme()
        plt.style.use('default')
        plt.rcParams.update({'figure.max_open_warning': 0})
        #plt.rcParams.update({'font.family': 'calibri'})
        plt.rcParams.update({'font.family': 'DejaVu Sans'})

        if ax is None:
            ax = plt.gca()
        ax.set_axisbelow(True)

        ax.tick_params(axis="y", direction="in", pad=-22)

        if self.dark_mode:
            seaborn.set(style="darkgrid")
            self.clr_bg = numpy.array((43, 43, 43)) / 255
            self.clr_grid = numpy.array((64, 64, 64)) / 255
            self.clr_font = numpy.array((1,1,1))#'white'
            self.clr_border = self.clr_bg

        else:
            seaborn.set(style="whitegrid")
            seaborn.set_style("whitegrid")
            self.clr_bg = numpy.array((1, 1, 1))
            #self.clr_bg = numpy.array((0.95, 0.95, 0.95))

            self.clr_grid = numpy.array((192, 192, 192)) / 255  # 'lightgray'
            self.clr_font = numpy.array((0,0,0))#'black'
            self.clr_border = numpy.array((1, 1, 1, 0))  # gray'

        seaborn.set_style('ticks', {'axes.edgecolor': self.clr_border, 'xtick.color': self.clr_font,
                                    'ytick.color': self.clr_font, 'ztick.color': self.clr_font})
        if fig is not None:
            fig.patch.set_facecolor(self.clr_bg)
        ax.set_facecolor(self.clr_bg)

        ax.spines['bottom'].set_color(self.clr_border)
        ax.spines['top'].set_color(self.clr_border)
        ax.spines['left'].set_color(self.clr_border)
        ax.spines['right'].set_color(self.clr_border)
        ax.xaxis.label.set_color(self.clr_font)
        ax.yaxis.label.set_color(self.clr_font)

        ax.tick_params(which='both', color=self.clr_font)
        # ax.tick_params(axis='x', colors=self.clr_font)
        # ax.tick_params(axis='y', colors=self.clr_font)

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
    def init_colors(self,N=32,cmap='tab20',shuffle=True):#tab20 nipy_ nipy_spectral
        numpy.random.seed(112)#113
        self.colors = tools_draw_numpy.get_colors(N, colormap=cmap).astype(int)

        new_c = []
        # for n in range(self.colors.shape[0]):
        #     color = self.colors[n]
        #     for alpha in numpy.linspace(0.3, 0.8, 7, endpoint=False):
        #         c = (1 - alpha) * numpy.array(color) + (alpha) * numpy.array((128, 128, 128))
        #         new_c.append(numpy.array(c).astype('int').reshape((1, 3)))

        # array(new_c).reshape((-1, 3))))

        if shuffle:
            self.colors=self.colors[numpy.random.choice(self.colors.shape[0],self.colors.shape[0])]
        self.dct_color ={}
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_color(self, label,alpha_blend=0.0):
        if label not in self.dct_color:
            if isinstance(label, numpy.int32) or isinstance(label, int) or isinstance(label, float) or isinstance(label, numpy.int64):
                n = int(label)
            else:
                n = numpy.array([ord(l) for l in label]).sum() % self.colors.shape[0]
            self.dct_color[label] = self.colors[n]

        res = (self.dct_color[label] * (1 - alpha_blend)) + (numpy.array((255, 255, 255)) * (alpha_blend))
        return res.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
    def set_color(self,label,value):
        self.dct_color[label] = value
        return
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
    def empty(self,figsize=(3.5,3.5),filename_out=None):
        fig = plt.figure(figsize=figsize)
        self.turn_light_mode(fig)

        plt.plot(0, 0)
        plt.grid(color=self.clr_grid)
        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())
        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def plot_image(self, image,filename_out):
        fig = plt.figure(figsize=(image.shape[1]/10.0,image.shape[0]/10.0))
        self.turn_light_mode(fig)

        plt.imshow(image[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_regression_df(self,df, idx_target, idx_num, idx_cat, filename_out=None):
        fig = plt.figure()
        self.turn_light_mode(fig)

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
    def plot_hist(self, X, bins=None, x_range=None, xlabel=None, color=(0.5,0.5,0.5), markers_x=None, marker_colors=None,markers_shape=None,figsize=(10, 2), filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        if markers_x is not None:
            for i,marker_x in enumerate(markers_x):
                if marker_x is None:
                    continue
                marker_color = marker_colors[i] if marker_colors is not None else (0,0,0)
                marker = markers_shape[i] if markers_shape is not None else 'o'
                plt.scatter(marker_x, -1, color=marker_color, marker=marker, zorder=+2)

        plt.hist(X,bins,color=color,stacked=True)

        plt.grid(color=self.clr_grid,which='major')
        if x_range is not None:
            plt.xlim(x_range)

        plt.yticks([])

        if xlabel is not None:
            plt.xlabel(xlabel)

        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())

        return self.get_image(fig, self.clr_bg)
# ----------------------------------------------------------------------------------------------------------------------
    def plot_1D_features_pos_neg(self,X, Y, labels=True, bins=None,colors=None,palette='tab10',transparency=0.5,xlim=None,filename_out=None):

        fig = plt.figure()
        fig = self.turn_light_mode(fig)

        patches = []

        plt.xlim([-5, X.max()+5])

        if colors is None:
            colors = seaborn.color_palette(palette)

        for i,y in enumerate(numpy.sort(numpy.unique(Y))):
            col = colors[i]
            x = X[Y == y] if len(X.shape)==1 else X[:,i]
            plt.hist(x, bins=bins,color=col,alpha=1-transparency,width=1.0,align='mid',density=False)

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
    def patch_labels(self,labels_str,dct):

        dct_inv = dict(zip(dct.values(),dct.keys()))

        label_res = []
        for l in labels_str:
            if '−' in l:
                label_res.append('')
            elif float(l) == math.floor(float(l)) and int(float(l)) in dct_inv.keys():
                label_res.append(dct_inv[int(float(l))])
            else:
                label_res.append('')
        return label_res
# ----------------------------------------------------------------------------------------------------------------------
    def plot_2D_features(self, df, x_range=None, y_range=None, add_noice=False, remove_legend=False, invert_y=False,transparency=0.0, colors=None,marker_size=3,palette='tab10', figsize=(8, 6), filename_out=None,cut=0):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        categoricals_hash_map = {}
        if add_noice:
            categoricals_hash_map = tools_DF.get_categoricals_hash_map(df)
            df = tools_DF.hash_categoricals(df)
            df = tools_DF.add_noise_smart(df)


        my_pal = colors if colors is not None else palette
        plt.rcParams.update({'lines.markersize': marker_size})
        J = seaborn.scatterplot(data=df,x=df.columns[1],y=df.columns[2],hue=df.columns[0],marker='o',palette=my_pal,edgecolor='none',alpha=1-transparency)

        plt.grid(color=self.clr_grid)

        #plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.15),ncol = 1)
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

        if remove_legend:
            plt.legend([], [], frameon=False)
        #else:
            #legend = J._get_patches_for_fill.axes.legend_
            #self.recolor_legend_seaborn(legend)

        if x_range is not None:plt.xlim(x_range)
        if y_range is not None:plt.ylim(y_range)
        if invert_y: plt.gca().invert_yaxis()

        plt.tight_layout()
        ax = plt.gca()
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())

        if add_noice and (df.columns[1] in categoricals_hash_map):
            labels_new = self.patch_labels([str(item.get_text()) for item in plt.gca().get_xticklabels()],categoricals_hash_map[df.columns[1]])
            #ax.set_xticks(range(len(labels_new)))
            ax.set_xticklabels(labels_new)

        if add_noice and (df.columns[2] in categoricals_hash_map):
            labels_new = self.patch_labels([str(item.get_text()) for item in plt.gca().get_yticklabels()],categoricals_hash_map[df.columns[2]])
            #ax.set_yticks(range(len(labels_new)))
            ax.set_yticklabels(labels_new)


        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())

        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def plot_2D_features_cumul(self, df, figsize=(6,6),remove_legend=False,filename_out=None):
        def max_element_by_value(dct):return max(dct.items(), key=operator.itemgetter(1))

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        X,Y = tools_DF.df_to_XY(df,idx_target=0)
        y0,y1 = sorted(set(Y))[0],sorted(set(Y))[1]

        dict_pos,dict_neg ={},{}
        for x in X[Y!=y0]:
            if tuple(x) not in dict_pos:dict_pos[tuple(x)]=1
            else:dict_pos[tuple(x)]+=1

        for x in X[Y==y0]:
            if tuple(x) not in dict_neg:dict_neg[tuple(x)]=1
            else:dict_neg[tuple(x)]+=1

        col_neg  = self.get_color(y0)[[2,1,0]]/255.0#(0, 0.5, 1, 1)
        col_pos  = self.get_color(y1)[[2,1,0]]/255.0#(1, 0.5, 0, 1)
        col_gray = (0.5, 0.5, 0.5, 1)

        min_size = 4
        max_size = 20
        norm = max(max_element_by_value(dict_pos)[1], max_element_by_value(dict_neg)[1]) / max_size

        for x in dict_pos.keys():
            if tuple(x) not in dict_neg.keys():
                plt.plot(x[0], x[1], 'o', color=col_pos, markeredgewidth=0, markersize=max(min_size, dict_pos[tuple(x)] / norm))
            else:
                plt.plot(x[0], x[1], 'o', color=col_gray, markeredgewidth=0, markersize=max(min_size, (dict_pos[tuple(x)] + dict_neg[tuple(x)]) / norm))

                if dict_pos[tuple(x)] < dict_neg[tuple(x)]:
                    plt.plot(x[0], x[1], 'o', color=col_neg, markeredgewidth=0, markersize=max(min_size, (dict_neg[tuple(x)] - dict_pos[tuple(x)]) / norm))
                else:
                    plt.plot(x[0], x[1], 'o', color=col_pos, markeredgewidth=0, markersize=max(min_size, (-dict_neg[tuple(x)] + dict_pos[tuple(x)]) / norm))

        for x in dict_neg:
            if tuple(x) not in dict_pos:
                plt.plot(x[0], x[1], 'o', color=col_neg, markeredgewidth=0, markersize=max(min_size, dict_neg[tuple(x)] / norm))

        plt.grid(color=self.clr_grid)

        if remove_legend:
            plt.legend([], [], frameon=False)
        else:
            patches = [mpatches.Patch(color=col_neg, label=y0),mpatches.Patch(color=col_pos, label=y1)]
            legend = plt.legend(handles=patches)
            self.recolor_legend_plt(legend)

        plt.xlabel(df.columns[1])
        plt.ylabel(df.columns[2])

        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())
        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def plot_2D_features_in_3D(self, df, x_range=None, y_range=None, add_noice=False, remove_legend=False,transparency=0.0, palette='tab10', figsize=(6, 6), filename_out=None):

        fig = plt.figure(figsize=figsize)

        if add_noice:
            df = tools_DF.add_noise_smart(df)

        ax = Axes3D(fig)
        #fig.add_axes(ax,rect=[0, 0, 1, 1])
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


        plt.show()


        return fig
#-----------------------------------------------------------------------------------------------------------------------
    def plot_tp_fp(self,tpr,fpr,roc_auc,caption='',figsize=(6,6),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)
        color = seaborn.color_palette(palette='tab10', n_colors=1)
        plt.grid(which='major', color=self.clr_grid, linestyle='--')
        plt.plot(fpr, tpr, color=color[0], lw=3, label='%s %.2f' % (caption, roc_auc))
        plt.plot([0, 1.05], [0, 1.05], color='lightgray', lw=1, linestyle='--')

        legend = plt.legend(loc='lower right')
        self.recolor_legend_plt(legend)

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out)

        plt.clf()
        plt.close(fig)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_PR(self, precisions,recalls, mAP=0, caption='', figsize=(3.5, 3.5), filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        idx =numpy.argmax([p * r / (p + r + 1e-4) for p, r in zip(precisions, recalls)])

        colors = seaborn.color_palette(palette='tab10', n_colors=1)
        plt.grid(which='major', color=self.clr_grid, linestyle='--')
        plt.plot(recalls, precisions, color=colors[0], lw=3, label='%s %.2f' % (caption, mAP))
        plt.scatter(recalls[idx], precisions[idx], color=colors[0])

        #precision_random = precisions[recalls==1].max()
        #plt.plot([0, 1], [precision_random, precision_random], color='lightgray', lw=1, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

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

        colors = [plt.get_cmap('tab20')(i) for i in [1, 6, 7, 2]]



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
    def histoplots_df(self,df0,idx_target=0,transparency=0.25,remove_legend=False,legend_loc = 'upper left',markersize=2,calc_info=True,figsize=(6,6),filename_out=None):

        columns = df0.columns
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, columns.shape[0]), idx_target)

        unique_targets = df0.iloc[:,idx_target].unique().tolist()
        HT = tools_Hyptest.HypTest()

        for column in columns[idx]:

            fig = plt.figure(figsize=figsize)
            fig = self.turn_light_mode(fig)
            plt.grid(color=self.clr_grid)

            df = df0[[target, column]].copy()
            df.dropna(inplace=True)
            is_categorical = tools_DF.is_categorical(df,column)
            if is_categorical:
                df[column] = df[column].astype(str)

            if calc_info:
                I = 0
                if len(unique_targets) == 2:
                    I = int(100 * HT.f1_score(df[df[target] == unique_targets[0]].iloc[:, 1:],
                                              df[df[target] == unique_targets[1]].iloc[:, 1:], is_categorical))

                df.rename(columns={column: '%s_%02d' % (column, I)}, inplace=True)
                column = '%s_%02d' % (column, I)

            df = tools_DF.remove_long_tail(df,order=True)
            orientation = 'horizontal' if df[column].unique().shape[0] >3 and is_categorical else 'vertical'

            df['#']=1
            df_agg = tools_DF.my_agg(df,[target,column],['#'],['count'],order_idx=-1 if is_categorical else None,ascending=False)
            for t in unique_targets:
                df_t = tools_DF.apply_filter(df_agg,target,t).copy()

                if len(unique_targets)==2 and t==unique_targets[1]:
                    df_t['#'] = -df_t['#']
                if is_categorical:
                    if orientation == 'horizontal':
                        plt.barh(y=df_t[column], width=df_t['#'], color=self.get_color(t)[[2, 1, 0]] / 255.0,alpha=1-transparency,label=t)
                    else:
                        plt.bar(x=df_t[column], height=df_t['#'], color=self.get_color(t)[[2,1,0]] / 255.0,alpha=1-transparency,label=t)
                else:
                    df_t.sort_values(by=column, inplace=True)
                    clr_fill = tools_draw_numpy.blend(self.get_color(t)[[2, 1, 0]], 255 * self.clr_bg, 1-transparency) / 255.0
                    plt.plot(df_t[column], df_t['#'], marker='o',markersize=markersize,color=self.get_color(t)[[2, 1, 0]] / 255.0,lw=1, label=t)
                    plt.fill_between(df_t[column], df_t['#'], df_t['#'] * 0, color=clr_fill,zorder=0)

            plt.xlabel(column)
            if remove_legend:
                plt.legend([], [], frameon=False)
            else:
                legend = plt.legend(loc=legend_loc, bbox_to_anchor=(0.0, 1.15),ncol = 1)
                self.recolor_legend_plt(legend)
            try:
                plt.savefig(self.folder_out + (filename_out if filename_out is not None else 'histo_%s.png' % (column)), facecolor=fig.get_facecolor())
                #df_agg.to_csv(self.folder_out + 'df_%s.csv' % (column),index=False)
            except:
                pass

            plt.clf()
            plt.close(fig)

            if calc_info:
                mode = 'a+' if os.path.exists(self.folder_out + 'descript.ion') else 'w'
                with open(self.folder_out + "descript.ion", mode=mode) as f_handle:
                    f_handle.write("%s %s\n" % ((filename_out if filename_out is not None else 'histo_%s.png' % (column)), '%03d' % I))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def jointplots_df(self,df0,idx_target=0,transparency=0.25,palette='tab10',remove_legend=False,figsize=(6,6)):

        columns = df0.columns
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, columns.shape[0]), idx_target)

        for i in range(len(idx)-1):
            for j in range(i+1,len(idx)):
                c1, c2 = columns[idx[i]], columns[idx[j]]
                df = df0[[target, c1, c2]].copy()
                df.dropna(inplace=True)
                is_categorical1 = str(df.dtypes[1]) in ['object', 'category', 'bool']
                is_categorical2 = str(df.dtypes[2]) in ['object', 'category', 'bool']

                if is_categorical1:df[c1] = df[c1].astype(str)
                if is_categorical2:df[c2] = df[c2].astype(str)

                df = tools_DF.remove_long_tail(df)

                fig = plt.figure(figsize=figsize)
                fig = self.turn_light_mode(fig)
                J = seaborn.jointplot(data=df, x=c1, y=c2, hue=target,palette=palette,edgecolor=None,alpha=1-transparency,legend=(not remove_legend))
                J.ax_joint.grid(color=self.clr_grid)
                J.ax_joint.set_facecolor(self.clr_bg)
                J.ax_marg_x.set_facecolor(self.clr_bg)
                J.ax_marg_y.set_facecolor(self.clr_bg)
                J.ax_joint.xaxis.label.set_color(self.clr_font)
                J.ax_joint.yaxis.label.set_color(self.clr_font)

                if remove_legend:
                    plt.legend([], [], frameon=False)
                else:
                    legend = J.ax_joint.legend()
                    self.recolor_legend_plt(legend)

                I = int(100 * mutual_info_classif(df.iloc[:, [1, 2]], df.iloc[:, 0]).sum())
                file_out = 'jointplot_%02d_%02d_%s_%s.png'%(i,j,c1,c2)
                plt.savefig(self.folder_out + file_out,facecolor=fig.get_facecolor())
                plt.close(fig)

                mode = 'a+' if os.path.exists(self.folder_out + 'descript.ion') else 'w'
                f_handle = open(self.folder_out + 'descript.ion', mode)
                f_handle.write("%s %s\n" % (file_out, '%03d' % I))
                f_handle.close()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def pairplots_df(self,df0, idx_target=0,df_Q=None,marker_size=6,transparency=0.75,cumul_mode=False,add_noise=False,mode2d=True,remove_legend=False,figsize=(6,6)):

        columns = df0.columns.to_numpy()
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

        unique_targets = df0.iloc[:, idx_target].unique().tolist()
        HT = tools_Hyptest.HypTest()

        colors =[self.get_color(t)[[2, 1, 0]] / 255.0 for t in unique_targets]

        for i in range(len(idx)-1):
            for j in range(i+1,len(idx)):
                c1, c2 = columns[idx[i]], columns[idx[j]]

                if df_Q is not None and df_Q.loc[c1,c2]==False:continue

                df = df0[[target,c1,c2]].copy()
                df.dropna(inplace=True)

                is_categorical1 = str(df.dtypes.iloc[1]) in ['object', 'category', 'bool']
                is_categorical2 = str(df.dtypes.iloc[2]) in ['object', 'category', 'bool']

                if is_categorical1: df[c1] = df[c1].astype(str)
                if is_categorical2: df[c2] = df[c2].astype(str)

                df = tools_DF.remove_long_tail(df)

                if df[c1].unique().shape[0]==1 or df[c2].unique().shape[0]==1:
                    continue

                I=0
                if len(unique_targets) == 2:
                    I = int(100 * HT.f1_score_2d(df[df[target] == unique_targets[0]].iloc[:, 1:], df[df[target] == unique_targets[1]].iloc[:,1:]))

                file_out = 'pairplot_%02d_%02d_%s_%s_%02d.png' % (i, j, c1, c2, I)


                if cumul_mode==True or (cumul_mode=='auto' and len(set(df.iloc[:,0]))==2 and df.iloc[:,1:].value_counts().shape[0]<122):
                    self.plot_2D_features_cumul(df, remove_legend=remove_legend,figsize=figsize,filename_out=file_out)
                else:
                    if mode2d:
                        self.plot_2D_features(df,add_noice=add_noise, transparency=transparency,marker_size=marker_size,colors=colors,remove_legend=remove_legend, figsize=figsize, filename_out=file_out)
                    else:
                        self.plot_2D_features_in_3D(df, add_noice=add_noise, transparency=transparency, remove_legend=remove_legend,figsize=figsize, filename_out=file_out)

                if remove_legend:
                    plt.legend([], [], frameon=False)

                df.to_csv(self.folder_out + 'df_%s_%s.csv' % (c1, c2),index=False)
                mode = 'a+' if os.path.exists(self.folder_out + 'descript.ion') else 'w'
                f_handle = open(self.folder_out + "descript.ion", mode)
                f_handle.write("%s %s\n" % (file_out, '%03d'%I))
                f_handle.close()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def pairplots_df_3d(self, df0, idx_target=0, palette='tab10',add_noise=False):

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

                mode = 'a+' if os.path.exists(self.folder_out + 'descript.ion') else 'w'
                f_handle = open(self.folder_out + "descript.ion", mode)
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
    def TS_matplotlib(self, df, idxs_target, idx_time, colors=None, lw=1, idxs_fill=None, remove_legend=False, x_range=None, out_format_x=None,out_locator_x=None, palette='tab10', figsize=(15, 3), filename_out=None):

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
            for idx_fill in idxs_fill[::-1]:
                #plt.fill_between(XX,df.iloc[:,idxs_fill[0]],df.iloc[:,idxs_fill[1]],color=self.clr_grid)
                y1 = df.iloc[:, idx_fill]
                y2 = numpy.zeros(shape=XX.shape[0])
                plt.fill_between(x=XX, y1=y1, y2=y2,color=self.clr_grid, zorder=-2)

        if remove_legend:
            plt.legend([], [], frameon=False)
        else:
            legend = plt.legend(handles=patches,loc="upper left")
            self.recolor_legend_plt(legend)

        if out_format_x is not None:
            plt.gca().xaxis.set_major_formatter(out_format_x)
        if out_locator_x is not None:
            plt.gca().xaxis.set_major_locator(out_locator_x)
        if x_range is not None:
            plt.xlim(x_range)
        plt.grid(color=self.clr_grid, which='major')

        plt.tight_layout()

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out, facecolor=fig.get_facecolor())
            plt.close()

        plt.close(fig)
        fig.clf()

        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def get_xtick_labels(self,df,idx_time,out_format_x,major_step,minor_step=None):

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

            # last_visible = values[0]
            # for i, x in enumerate(values):
            #     delta = ((x - last_visible) / numpy.timedelta64(1, 'D'))
            #     if i == 0 or (delta >= major_step):
            #         idx_visible.append(i)
            #         last_visible = x

            last_visible = values[-1]
            for i, x in enumerate(values[::-1]):
                delta = ((last_visible - x) / numpy.timedelta64(1, 'D'))
                if i==0 or (delta >= major_step):
                    idx_visible.append(values.shape[0]-1-i)
                    last_visible = x
            idx_visible = idx_visible[::-1]

        else:
            xtick_labels = None
            idx_visible = None

        return xtick_labels,idx_visible
# ----------------------------------------------------------------------------------------------------------------------
    def TS_seaborn(self, df, idxs_target, idx_time, idx_hue=None,bg_image=None,
                   mode='pointplot', idxs_fill=None,remove_legend=False,legent_loc='upper left',remove_grid=False,
                   remove_xticks=False,remove_yticks=False, x_range=None,y_range=None,out_format_x=None, out_locator_x=None,invert_y=False,lw=2,transparency=0, figsize=(15, 3), filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)
        columns = [c for c in df.columns]

        X = columns[idx_time] if idx_time is not None else df.index
        if not isinstance(idxs_target, list):idxs_target = [idxs_target]
        colors = [self.get_color(columns[idx_target])[[2, 1, 0]] / 255 for idx_target in idxs_target]
        seaborn.set(style="ticks", rc={'lines.linewidth': lw,'lines.markersize': 1.0})
        hue = df.columns[idx_hue] if idx_hue is not None else None

        for i,idx_target in enumerate(idxs_target):
            if   mode == 'pointplot'  :g = seaborn.pointplot(data=df, x=X, y=df.columns[idx_target], color=colors[i],markers='', label=df.columns[idx_target],errwidth=4)
            elif mode == 'scatterplot':g = seaborn.scatterplot(data=df, x=X, y=df.columns[idx_target],size=numpy.full(df.shape[0],2.25),color=colors[i],alpha=1-transparency,edgecolor=None,markers='0',label=df.columns[idx_target])
            elif mode == 'barplot'    :g = seaborn.barplot(data=df,x=X, y=df.columns[idx_target],hue=hue,color=colors[i])
            else:                      g = seaborn.lineplot(data=df, x=X, y=df.columns[idx_target],color=colors[i],label=df.columns[idx_target])
            g.set_xlabel('')
            g.set_ylabel('')

        if idxs_fill is not None and mode=='pointplot':
            for idx_fill in idxs_fill[::-1]:
                xy = g.lines[int(len(g.lines) / len(idxs_target) * idx_fill)].get_xydata()
                clr = tools_draw_numpy.blend(255*colors[idx_fill],255*self.clr_bg,1-transparency)/255.0
                plt.fill_between(x=xy[:,0], y1=xy[:,1], color=clr,zorder=-2)

        patches = [mpatches.Patch(color=colors[i], label=columns[idx_target]) for i, idx_target in enumerate(idxs_target)]

        if not remove_grid:
            plt.grid(color=self.clr_grid,which='major',zorder=-1)
        else:
            plt.gca().yaxis.grid(False)

        if out_format_x is not None:
            plt.gca().xaxis.set_major_formatter(out_format_x)
        if out_locator_x is not None:
            plt.gca().xaxis.set_major_locator(out_locator_x)
        if x_range is not None:
            plt.xlim(x_range)
        if remove_xticks:
            g.set(xticks=[])
        if y_range is not None:
            plt.ylim(y_range)
        if remove_yticks:
            g.set(yticks=[])
        if invert_y:
            plt.gca().invert_yaxis()
        if remove_legend:
            plt.legend([], [], frameon=False)
        else:
            #legend = plt.legend(handles=patches,loc=legent_loc, bbox_to_anchor=(0.0, -0.10), ncol=len(patches))
            legend = plt.legend(handles=patches)
            self.recolor_legend_plt(legend)

        plt.tight_layout()

        if bg_image is not None:
            g.imshow(bg_image[:,:,[2,1,0]],zorder=-1,aspect = g.get_aspect(),extent = g.get_xlim() + g.get_ylim())

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
    def plot_PCA(self,df, idx_target,method='PCA',filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        if method=='PCA':
            X = PCA(n_components=2).fit_transform(X)
        elif method == 'tSNE':
            N = 10000
            if Y.shape[0]>N:
                idx = numpy.random.choice(Y.shape[0],N,replace=False)
                X,Y = X[idx],Y[idx]

            X = TSNE(n_components=2).fit_transform(X)
        elif method == 'UMAP':
            import umap
            X = umap.UMAP(n_components=2).fit_transform(X)
        elif method == 'SVD':
            X = TruncatedSVD(n_components=2).fit_transform(X)
        elif method == 'LLE':
            X = LocallyLinearEmbedding(n_components=2).fit_transform(X)
        elif method == 'ISOMAP':
            X = Isomap(n_components=2).fit_transform(X)
        else:
            return

        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X), axis=1), columns=['Y', 'x0', 'x1'])
        df.sort_values(by=df.columns[0],inplace=True)
        #colors = [self.get_color(t)[[2, 1, 0]] / 255.0 for t in numpy.sort(numpy.unique(Y))]
        colors = [self.colors[n%(self.colors.shape[0])]   [[2, 1, 0]] / 255.0 for n in range(len(numpy.unique(Y)))]
        #colors = None
        self.plot_2D_features(df, remove_legend=False, colors=colors,marker_size=6,transparency=0.75,filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_contourf(self,X0,X1,xx, yy, grid_confidence,xlim=None,ylim=None,xlabel=None,ylabel=None,transparency=0,marker_size=10,figsize=(3.5,3.5),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        plt.rcParams.update({'lines.markersize': marker_size})

        map_RdBu = [cm.RdBu(i) for i in numpy.linspace(0, 1, 15)]
        map_PuOr = [cm.PuOr(i) for i in numpy.linspace(0, 1, 15)]
        my_cmap = map_RdBu[7:][::-1] + map_PuOr[:7][::-1]
        my_cmap = colors.ListedColormap(my_cmap)
        #my_cmap = cm.coolwarm

        plt.contourf(xx, yy, grid_confidence, cmap=my_cmap, alpha=.15)
        # plt.plot(X1[:, 0], X1[:, 1], 'ro', color='#FF7F0E', alpha=1 - transparency)
        # plt.plot(X0[:, 0], X0[:, 1], 'ro', color='#1F77B4', alpha=1-transparency)
        plt.scatter(x=X1[:, 0], y=X1[:, 1], marker='o', color='#FF7F0E', alpha=1 - transparency,edgecolors='none')
        plt.scatter(x=X0[:, 0], y=X0[:, 1], marker='o', color='#1F77B4', alpha=1-transparency,edgecolors='none')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        if xlim is not None:plt.xlim(xlim)
        if ylim is not None:plt.ylim(ylim)

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        plt.close(fig)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_hor_bars(self, values, labels, colors=None,legend=None,xticks=None, transparency=0,palette='tab10',figsize=(3.5, 6), filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        if colors is None:
            colors = seaborn.color_palette(palette=palette, n_colors=1)
            single_color = colors[0]
        else:
            single_color = [numpy.array(c)/255.0 for c in colors]

        y_pos = numpy.arange(len(labels))

        if len(values.shape)==1:
            plt.barh(y_pos, values,color=single_color,alpha=1-transparency,zorder=-2)
        else:
            for i in range(values.shape[1]):
                plt.barh(y_pos, values[:,i], color=single_color, alpha=1 - transparency)

        plt.yticks(y_pos, labels)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if xticks is not None:
            plt.xticks(xticks)

        plt.grid(color=self.clr_grid)
        plt.gca().tick_params(axis="y",length=0, direction="in")
        plt.gca().yaxis.grid(False)


        if legend is not None:
            legend = plt.legend([legend],loc="lower right")
            self.recolor_legend_plt(legend)

        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out,facecolor=fig.get_facecolor())

        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def plot_bars(self, y_values, x_values, legend=None, yticks=None,xlim=None,markers_x=None,marker_colors=None, markers_shape=None,colors=(0.5,0.5,0.5),figsize=(3.5, 6), filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        plt.plot(x_values, y_values, '-', color=colors)

        if markers_x is not None:
            for i,marker_x in enumerate(markers_x):
                if marker_x is None:
                    continue
                marker_color = marker_colors[i] if marker_colors is not None else (0,0,0)
                marker = markers_shape[i] if markers_shape is not None else 'o'
                plt.scatter(marker_x, -1, color=marker_color, marker=marker, zorder=+2)

        if xlim is not None:
            plt.xlim(xlim)

        #plt.xticks(x_pos, labels)
        #plt.xticks([])

        plt.grid(color=self.clr_grid, which='major')

        if yticks is not None:
            plt.yticks(yticks)
        else:
            plt.yticks([])

        if legend is not None:
            plt.xlabel(legend)

        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())


        return self.get_image(fig, self.clr_bg)
# ----------------------------------------------------------------------------------------------------------------------
    def plot_pie(self,values,header,filename_out=None):

        fig = plt.figure()
        fig = self.turn_light_mode(fig)
        plt.pie(values,  labels=header, autopct='%1.1f%%',shadow=False, startangle=90)
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        return fig
# ---------------------------------------------------------------------------------------------------------------------
    def plot_squarify(self, df, idx_label, idx_size, colors=None,idx_color_255=None, palette='viridis',stat='%', alpha=0, figsize=(10, 4), filename_out=None):
        import squarify
        fig = plt.figure(figsize=figsize)
        self.turn_light_mode(fig)

        padding = 3
        font_size = 18
        W, H = int(figsize[0]*100), int(figsize[1]*100)

        image = numpy.full((H, W, 3), self.clr_bg*255, dtype=numpy.uint8)
        sizes = df.iloc[:, idx_size].values

        labels = df.iloc[:, idx_label].apply(lambda x:str(x).replace(' ', '\n')).values
        if   stat=='%':labels = numpy.array(['%s %.1f%%' % (l, float(100 * w / sizes.sum())) for w, l in zip(sizes, labels)])
        elif stat=='#':labels = numpy.array(['%s %d' % (l, w) for l,w in zip(labels,sizes)])

        if colors is None:
            col255 = tools_draw_numpy.get_colors(256, shuffle=False, colormap=palette)
            if idx_color_255 is not None:
                colors = col255[[int(i) for i in df.iloc[:, idx_color_255]]]
            else:
                sss = sizes - numpy.min(sizes)
                sss = 255*sss/sss.max()
                colors = [col255[int(s)] for s in sss]

        sss = squarify.normalize_sizes(sizes/sizes.sum(), W, H)
        sq_res = numpy.array([(el['x'], el['y'], el['dx'], el['dy']) for el in squarify.squarify(sss, 0, 0, W, H)])

        if padding > 0:
            sq_res[:, 0] += padding
            sq_res[:, 2] -= 2 * padding
            sq_res[:, 1] += padding
            sq_res[:, 3] -= 2 * padding

        for sq, label, color in zip(sq_res, labels, colors):
            col_left, row_up, col_right, row_down = sq[0], sq[1], sq[0] + sq[2], sq[1] + sq[3]

            image = tools_draw_numpy.draw_rect(image, col_left, row_up, col_right, row_down, color,w=2,alpha_transp=alpha)
            pos0 = (int((col_left + col_right) / 2), int((row_up + row_down) / 2))
            shift_x, shift_y, sx, sy = tools_draw_numpy.get_position_sign(label, W, H, font_size,pos0)
            pos = int(pos0[0] + shift_x), int(pos0[1] + shift_y)
            image = tools_draw_numpy.draw_text(image, label,pos,255-self.clr_bg*255, None, font_size,alpha_transp=alpha)

        cv2.imwrite(self.folder_out + filename_out, image)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def plot_feature_correlation(self,df_Q,figsize=(8,8),filename_out=None):

        fig = plt.figure(figsize=figsize)
        seaborn.heatmap(df_Q, vmax=1,        annot=True, fmt='.2f', cmap='GnBu' , cbar_kws={"shrink": .5}, robust=True, square=True)

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        return
# ---------------------------------------------------------------------------------------------------------------------
    def scatter_3d(self,df1, df2=None,figsize=(8,8),filename_out=None):
        col0, col1, col2 = df1.columns[:3]
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection="3d")

        ax.scatter3D(df1.iloc[:, 0], df1.iloc[:, 1], df1.iloc[:, 2], c='blue', marker='.')
        if df2 is not None:
            ax.scatter3D(df2.iloc[:, 0], df2.iloc[:, 1], df2.iloc[:, 2], c='red', marker='.')

        ax.set_xlabel(col0)
        ax.set_ylabel(col1)
        ax.set_zlabel(col2)

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        return
# ---------------------------------------------------------------------------------------------------------------------
    def inplace_image(self, image, str_start, str_stop, major_step_days=7, out_format_x=None,figsize=(10, 4),filename_out=None):

        H, W = image.shape[:2]
        time_range = tools_time_convertor.generate_date_range(str_start, str_stop, freq='D')
        df = pd.DataFrame({'time': time_range, 'value_min': -1, 'value_max': H+1})
        self.set_color('value_min', self.color_black)
        self.set_color('value_max', self.color_black)
        self.TS_seaborn(df, idxs_target=[1, 2], idx_time=0, bg_image=image,mode='pointplot',
                        #idxs_fill=[0,1],
                        out_format_x=out_format_x,
                        remove_legend=True, remove_yticks=False,
                        y_range=[0, H],
                        lw=1,transparency=1,
                        figsize=figsize,filename_out=filename_out)

        return

# ---------------------------------------------------------------------------------------------------------------------
    def get_image(self, fig, clr_bg):

        self.turn_light_mode(fig)
        # ax = plt.gca()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.set_facecolor(numpy.array(self.clr_bg)[[2, 1, 0]] / 255)

        #plt.tight_layout()

        # ax.tick_params(which='major', length=0)

        fig.savefig(self.io_buf, format='raw', facecolor=clr_bg)
        self.io_buf.seek(0)
        image = numpy.reshape(numpy.frombuffer(self.io_buf.getvalue(), dtype=numpy.uint8),newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:, :, [2, 1, 0]]
        image = numpy.ascontiguousarray(image, dtype=numpy.uint8)

        return image
# ----------------------------------------------------------------------------------------------------------------------

