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
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_draw_numpy
from sklearn.feature_selection import mutual_info_classif
# ----------------------------------------------------------------------------------------------------------------------
class Plotter(object):
    def __init__(self,folder_out=None,dark_mode=False):
        self.folder_out = folder_out
        self.dark_mode = dark_mode
        return
# ----------------------------------------------------------------------------------------------------------------------
    def turn_light_mode(self,fig):

        seaborn.set_theme()
        plt.style.use('default')
        ax = plt.gca()
        ax.set_axisbelow(True)

        if self.dark_mode:
            seaborn.set(style="darkgrid")
            self.clr_bg = numpy.array((43, 43, 43)) / 255
            self.clr_grid = numpy.array((64, 64, 64)) / 255
            self.clr_font = 'white'
            self.clr_border = 'darkgray'

        else:
            seaborn.set(style="whitegrid")
            seaborn.set_style("whitegrid")
            self.clr_bg = numpy.array((1, 1, 1))
            self.clr_grid = 'lightgray'
            self.clr_font = 'black'
            self.clr_border = 'gray'

        seaborn.set_style('ticks', {'axes.edgecolor': self.clr_border, 'xtick.color': self.clr_font, 'ytick.color': self.clr_font})
        fig.patch.set_facecolor(self.clr_bg)
        ax.set_facecolor(self.clr_bg)

        ax.spines['bottom'].set_color(self.clr_border)
        ax.spines['top'].set_color(self.clr_border)
        ax.spines['left'].set_color(self.clr_border)
        ax.spines['right'].set_color(self.clr_border)
        ax.xaxis.label.set_color(self.clr_font)
        ax.yaxis.label.set_color(self.clr_font)
        ax.tick_params(axis='x', colors=self.clr_font)
        ax.tick_params(axis='y', colors=self.clr_font)

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
    def plot_hist(self,X,x_range=None,figsize=(3.5,3.5),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        plt.hist(X)
        if x_range is not None:
            plt.xlim(x_range)


        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out, facecolor=fig.get_facecolor())

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_1D_features_pos_neg(self,X, Y, labels=None, bins=None,filename_out=None):

        fig = plt.figure()
        fig = self.turn_light_mode(fig)

        patches = []

        plt.xlim([-1, X.max()+1])


        for i,y in enumerate(numpy.unique(Y)):
            if int(y)<=0:
                col = (0, 0.5, 1)
            else:
                col = (1, 0.5, 0)
            plt.hist(X[Y==y], bins=bins,color=col,alpha=0.5,width=0.25,align='mid')

            if labels is not None:
                patches.append(mpatches.Patch(color=col,label=y))

        plt.grid()

        if labels is not None:
            plt.legend(handles=patches)

        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_2D_features_v3(self, df, x_range=None, y_range=None,add_noice=False,remove_legend=False,transparency=0.0,palette='tab10',figsize=(3.5,3.5),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)

        if add_noice:
            noice = 0.05 - 0.1 * numpy.random.random_sample(df.shape)
            df.iloc[:,1:]+= noice[:,1:]

        plt.grid(color=self.clr_grid)
        J = seaborn.scatterplot(data=df,x=df.columns[1],y=df.columns[2],hue=df.columns[0],palette=palette,edgecolor=None,alpha=1-transparency)
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
    def plot_tp_fp(self,tpr,fpr,roc_auc,caption='',figsize=(3.5,3.5),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)
        color = seaborn.color_palette(palette='tab10', n_colors=1)

        lw = 3
        plt.grid(which='major', color=self.clr_grid, linestyle='--')
        plt.plot(fpr, tpr, color=color[0], lw=lw, label='AUC %s = %0.2f' % (caption, roc_auc))
        plt.plot([0, 1.05], [0, 1.05], color='lightgray', lw=1, linestyle='--')
        legend = plt.legend(loc="lower right")
        self.recolor_legend_plt(legend)

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        plt.close(fig)

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
    def jointplots_df(self,df0, idx_target=0):

        columns = df0.columns.to_numpy()
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
        pal = 'tab10'

        for i in range(len(idx)):
            c = columns[idx[i]]
            df = df0[[target, c]]
            df = df.dropna()
            df = tools_DF.hash_categoricals(df)

            fig = plt.figure()
            fig = self.turn_light_mode(fig)
            plt.grid(color=self.clr_grid)
            J = seaborn.histplot(data=df, x=c, hue=target, palette=pal,element='poly',legend=True)

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
                J = seaborn.jointplot(data=df, x=c1, y=c2, hue=target,palette=pal,edgecolor=None)
                J.ax_joint.grid(color=self.clr_grid)
                J.ax_joint.set_facecolor(self.clr_bg)
                J.ax_marg_x.set_facecolor(self.clr_bg)
                J.ax_marg_y.set_facecolor(self.clr_bg)
                J.ax_joint.xaxis.label.set_color(self.clr_font)
                J.ax_joint.yaxis.label.set_color(self.clr_font)

                legend = J.ax_joint.legend()
                self.recolor_legend_plt(legend)

                plt.savefig(self.folder_out + 'pairplot_%02d_%02d_%s_%s.png'%(i,j,c1,c2),facecolor=fig.get_facecolor())
                plt.close(fig)


        return
# ----------------------------------------------------------------------------------------------------------------------
    def pairplots_df(self,df0, idx_target=0,cumul_mode=False,add_noise=True):

        f_handle = open(self.folder_out + "descript.ion", "w+")
        f_handle.close()

        columns = df0.columns.to_numpy()
        target = columns[idx_target]
        idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
        transparency = 0.95 if add_noise else 0

        for i in range(len(idx)-1):
            for j in range(i+1,len(idx)):
                c1, c2 = columns[idx[i]], columns[idx[j]]
                df = df0[[target,c1,c2]]
                df = df.dropna()
                df = tools_DF.hash_categoricals(df)
                I = int(100*mutual_info_classif(df.iloc[:,[1, 2]], df.iloc[:,0]).sum())
                file_out = 'pairplot_%02d_%02d_%s_%s_%02d.png' % (i, j, c1, c2, I)
                if cumul_mode:
                    self.plot_2D_features_cumul(df, remove_legend=True,filename_out=file_out)
                else:
                    self.plot_2D_features_v3(df, add_noice=add_noise,transparency=transparency,remove_legend=True,filename_out=file_out)
                f_handle = open(self.folder_out + "descript.ion", "a+")
                f_handle.write("%s %s\n" % (file_out, '%03d'%I))
                f_handle.close()


        for i in range(len(idx)):
            c1 = columns[idx[i]]
            df = df0[[target, c1]]
            df = df.dropna()
            df = tools_DF.hash_categoricals(df)
            bins = numpy.arange(-0.5, df[[c1]].max() + 0.5, 0.25)
            self.plot_1D_features_pos_neg(df[[c1]].to_numpy(), df[target].to_numpy(), labels=c1, bins=bins,filename_out='plot_%02d_%02d_%s.png' % (i, i,c1))


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
    def TS_matplotlib(self, df, idxs_target,idx_feature,idxs_fill=None,remove_legend=False,x_range=None,y_range=None,figsize=(22, 3),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)
        ax = plt.gca()
        XX = numpy.linspace(0,df.shape[0]-1,df.shape[0])
        X = None if idx_feature is None else df.columns[idx_feature]
        if not isinstance(idxs_target, list): idxs_target = [idxs_target]
        colors = seaborn.color_palette(palette='tab10', n_colors=len(idxs_target))


        for i, idx_target in enumerate(idxs_target):
            df.plot(x=X, y=df.columns[idx_target], ax=ax, color=colors[i])
            #plt.plot(XX, df.iloc[:,idx_target],color = colors[i])

        if idxs_fill is not None:
            plt.fill_between(XX,df.iloc[:,idxs_fill[0]],df.iloc[:,idxs_fill[1]],color=self.clr_grid)

        if remove_legend:
            plt.legend([], [], frameon=False)
        else:
            legend = plt.legend(loc="upper left")
            self.recolor_legend_plt(legend)

        if x_range is not None:plt.xlim(x_range)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out, facecolor=fig.get_facecolor())
            plt.close()

        plt.close(fig)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def TS_seaborn(self,df,idxs_target,idx_feature,mode='pointplot',remove_legend=False,figsize=(12, 4),filename_out=None):

        fig = plt.figure(figsize=figsize)
        fig = self.turn_light_mode(fig)
        X = df.index if idx_feature is None else df.columns[idx_feature]
        if not isinstance(idxs_target, list):idxs_target = [idxs_target]
        colors = seaborn.color_palette(palette='tab10', n_colors=len(idxs_target))
        patches = []
        for i,idx_target in enumerate(idxs_target):
            if mode=='pointplot':
                g = seaborn.pointplot(data=df, x=X, y=df.columns[idx_target], scale=0.25,color=colors[i],markers='o',label=df.columns[idx_target])
                patches.append(mpatches.Patch(color=colors[i], label=df.columns[idx_target]))
            else:
                g = seaborn.lineplot(data=df, x=X, y=df.columns[idx_target],color=colors[i],label=df.columns[idx_target])

        if remove_legend:
            plt.legend([], [], frameon=False)
        else:
            legend = plt.legend(handles=patches)
            self.recolor_legend_plt(legend)

        g.set_ylabel('')
        g.set(xticks=[])
        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out+filename_out,facecolor=fig.get_facecolor())

        #plt.close(fig)
        return fig
# ----------------------------------------------------------------------------------------------------------------------
    def TS_plotly(self,df,idx_target,idx_feature,filename_out=None):
        import plotly.express as px
        import plotly.io as pio

        fig = px.line(data_frame=df,x=df.columns[idx_feature], y=df.columns[idx_target])
        if filename_out is not None:
            pio.write_image(fig, self.folder_out+filename_out, width=1080, height=720)
        #fig.show()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_TS_separatly(self, df, idx_target):
        df = tools_DF.hash_categoricals(df)

        for i, feature in enumerate(df.columns):
            color = seaborn.color_palette(palette='Dark2')[0] if i == idx_target else None
            self.TS_matplotlib(df, i, None, color=color, filename_out='%s.png' % feature)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_tSNE(self,df, idx_target,filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_TSNE = TSNE(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_TSNE), axis=1), columns=['tSNE', 'x0', 'x1'])
        df = df.astype({'tSNE': 'int32'})
        self.plot_2D_features_v3(df, remove_legend=True, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_PCA(self,df, idx_target,filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_PCA = PCA(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_PCA), axis=1), columns=['PCA', 'x0', 'x1'])
        df = df.astype({'PCA': 'int32'})
        self.plot_2D_features_v3(df, remove_legend=True, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_UMAP(self,df, idx_target,filename_out=None):
        import umap
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_UMAP = umap.UMAP(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_UMAP), axis=1), columns=['UMAP', 'x0', 'x1'])
        df = df.astype({'UMAP': 'int32'})
        self.plot_2D_features_v3(df, remove_legend=True, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_SVD(self,df, idx_target,filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_SVD = TruncatedSVD(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_SVD),axis=1),columns=['SVD','x0','x1'])
        df = df.astype({'SVD': 'int32'})
        self.plot_2D_features_v3(df, remove_legend=True,filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_LLE(self,df, idx_target,filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_LLE = LocallyLinearEmbedding(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_LLE),axis=1),columns=['LLE','x0','x1'])
        df = df.astype({'LLE': 'int32'})
        self.plot_2D_features_v3(df, remove_legend=True, filename_out=filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def plot_ISOMAP(self,df, idx_target,filename_out=None):
        X, Y = tools_DF.df_to_XY(df, idx_target)
        X_ = Isomap(n_components=2).fit_transform(X)
        df = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X_), axis=1), columns=['ISOMAP', 'x0', 'x1'])
        df = df.astype({'ISOMAP': 'int32'})
        self.plot_2D_features_v3(df, remove_legend=True, filename_out=filename_out)
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
    def plot_squarify(self,weights,labels,filename_out=None):
        colors2 = (tools_draw_numpy.get_colors(1 + len(labels), colormap='tab10', alpha_blend=0.25,shuffle=True) / 255)[1:]
        colors2 = numpy.hstack((colors2, numpy.full((len(labels), 1), 1)))

        fig = plt.figure(figsize=(12, 4))
        fig = self.turn_light_mode(fig)

        squarify.plot(sizes=weights, label=labels, color=colors2,ax=plt.gca())
        plt.axis('off')
        plt.tight_layout()
        if filename_out is not None:
            plt.savefig(self.folder_out + filename_out)


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
