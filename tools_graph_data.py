#http://moviegraphs.cs.toronto.edu/
#https://www.data-to-viz.com/story/AdjacencyMatrix.html
#https://github.com/briatte/awesome-network-analysis
#https://www.toptal.com/data-science/graph-data-science-python-networkx
#https://github.com/topics/graph-anomaly-detection
#https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/tree/main/graph%20data
#https://towardsdatascience.com/pyvis-visualize-interactive-network-graphs-in-python-77e059791f01
# ----------------------------------------------------------------------------------------------------------------------
import cv2
import networkx
import numpy
import pandas as pd
import inspect

import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.data import Data
from sknetwork.clustering import Louvain,PropagationClustering
from pyvis.network import Network
import matplotlib.pyplot as plt
from scipy import stats
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_Hyptest
import tools_time_profiler
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Graph_Processor(object):
    def __init__(self,folder_out):
        self.folder_out=folder_out
        self.TP = tools_time_profiler.Time_Profiler()
        self.HypTest = tools_Hyptest.HypTest()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_data_karate(self):
        data = KarateClub()[0]

        #remove some connections
        edge_index = numpy.array(data.edge_index).T
        for node in [4,5,6,11,16,21,10,15]:
            edge_index = numpy.delete(edge_index, numpy.where(edge_index[:,0]==node)[0], 0)
            edge_index = numpy.delete(edge_index, numpy.where(edge_index[:,1]==node)[0], 0)

        data.edge_index = torch.from_numpy(edge_index.T)

        return data
# ----------------------------------------------------------------------------------------------------------------------
    def get_data_facebook(self,filename_in):
        df =  pd.read_csv(filename_in,sep=' ')
        edge_index = df.values
        data = Data(x=numpy.arange(1+numpy.max(edge_index)), edge_index=torch.from_numpy(numpy.array(edge_index).T))
        return data
# ----------------------------------------------------------------------------------------------------------------------
    def split_items(self, A):
        A = numpy.concatenate([a.split(',') for a in A])
        A = numpy.unique([a[1:] if a[0] == ' ' else a for a in A])
        return A
# ----------------------------------------------------------------------------------------------------------------------
    def enrich_with_means(self, df_items, df_features0, do_split=False):

        df_features = df_features0.copy()
        key = df_features.columns[0]
        if do_split:
            df_features[key] = df_features[key].apply(lambda k:k.split(','))
            df_features = df_features.explode(key)
            df_features[key] = df_features[key].apply(lambda a:a[1:] if a[0] == ' ' else a)

        for col_value in df_features.columns[1:]:
            df_agg = tools_DF.my_agg(df_features,[key],cols_value=[col_value],aggs=['mean'],list_res_names=[col_value+'_mean'])
            df_items = tools_DF.fetch(df_items,df_items.columns[1],df_agg,key,df_agg.columns[-1])

        return df_items
# ----------------------------------------------------------------------------------------------------------------------
    def enrich_with_KS(self, df_items, df_features0, do_split=False):

        df_features = df_features0.copy()
        key = df_features.columns[0]
        if do_split:
            df_features[key] = df_features[key].apply(lambda k:k.split(','))
            df_features = df_features.explode(key)
            df_features[key] = df_features[key].apply(lambda a:a[1:] if a[0] == ' ' else a)

        col_value = df_features.columns[1]
        df_features[col_value] = df_features[col_value].fillna(0)
        df_agg = tools_DF.my_agg(df_features, [key], cols_value=[col_value], aggs=['count','mean',numpy.std],list_res_names=['#','mean','std'])
        df_agg['std'] = df_agg['std'].fillna(0)
        df_items = tools_DF.fetch(df_items, df_items.columns[1], df_agg, key, df_agg.columns[-3:].values.tolist())

        for col_value in df_features.columns[1:]:
            values1 = df_features[col_value].dropna().values
            p_values = []
            for item in df_items.iloc[:,1]:
                values2 = df_features[df_features[key]==item][col_value].dropna().values
                res1 = stats.ks_2samp(values1, values2).pvalue if len(values2)>0 else 0
                p_values.append(res1)
            df_items[col_value+'_KS'] = p_values

        return df_items
# ----------------------------------------------------------------------------------------------------------------------
    def calc_edge_index(self,df_pairs):
        edge_index = []
        for item2 in df_pairs.iloc[:, 1].unique():
            itms1 = df_pairs[df_pairs.iloc[:, 1] == item2].iloc[:, 0].values
            for i1 in range(len(itms1) - 1):
                for i2 in range(i1 + 1, len(itms1)):
                    edge_index.append((itms1[i1], itms1[i2]))
                    edge_index.append((itms1[i2], itms1[i1]))

        # M = numpy.zeros(1+df_pairs.max().values)
        # M[df_pairs.values[:, 0], df_pairs.values[:, 1]] = 1
        # for r1 in range(M.shape[0]-1):
        #     for r2 in range(r1+1,M.shape[0]):
        #         if numpy.dot(M[r1],M[r2])>0:
        #             edge_index.append((r1,r2))

        return numpy.array(edge_index)
# ----------------------------------------------------------------------------------------------------------------------
    def calc_pairs(self,df,Items1,Items2,idx_entity1,idx_entity2,do_split1,do_split2):
        idx_items1 = dict(zip(Items1, numpy.arange(Items1.shape[0])))
        idx_items2 = dict(zip(Items2, numpy.arange(Items2.shape[0])))
        pairs = []

        for r in range(df.shape[0]):
            items1 = [df.iloc[r, idx_entity1]]
            if do_split1:items1 = self.split_items(items1)
            items2 = df.iloc[r, idx_entity2]
            if do_split2:items2 = self.split_items([items2])
            else:items2 = [items2]

            for item1 in items1:
                for item2 in items2:
                    pairs.append((idx_items1[item1],idx_items2[item2]))

        pairs=numpy.unique(pairs,axis=0)
        return pairs
# ----------------------------------------------------------------------------------------------------------------------
    def feature_engineering(self, dataframe, idx_entity1, idx_entity2, idx_features1=[], idx_features2=[], do_split1=False, do_split2=False):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)

        df = pd.read_csv(dataframe) if isinstance(dataframe, str) else dataframe

        Items1 = df.iloc[:, idx_entity1].unique()
        if do_split1:Items1 = self.split_items(Items1)

        Items2 = df.iloc[:, idx_entity2].unique()
        if do_split2:Items2 = self.split_items(Items2)

        df_items1 = pd.DataFrame({'id':numpy.arange(Items1.shape[0]), df.columns[idx_entity1]:Items1})
        df_items2 = pd.DataFrame({'id':numpy.arange(Items2.shape[0]), df.columns[idx_entity2]:Items2})

        pairs = self.calc_pairs(df,Items1,Items2,idx_entity1,idx_entity2,do_split1,do_split2)
        df_pairs = pd.DataFrame({df.columns[idx_entity1]+'_id':pairs[:,0],df.columns[idx_entity2]+'_id':pairs[:,1]})

        if len(idx_features1)>0:df_items1 = self.enrich_with_KS(df_items1, df.iloc[:, [idx_entity1] + idx_features1], do_split1)
        if len(idx_features2)>0:df_items2 = self.enrich_with_KS(df_items2, df.iloc[:, [idx_entity2] + idx_features2], do_split2)

        data = Data(x=df_items1, edge_index=torch.from_numpy((self.calc_edge_index(df_pairs)).T))
        df_items1['cluster_id'] = self.get_cluster_id_Louvain(data)

        df_items1.to_csv(self.folder_out + df.columns[idx_entity1] + '.csv', index=False,float_format='%.2f')
        df_items2.to_csv(self.folder_out + df.columns[idx_entity2] + '.csv', index=False,float_format='%.2f')
        #df_pairs.to_csv(self.folder_out + df.columns[idx_entity1] + '_'+df.columns[idx_entity2] + '.csv', index=False)

        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return data
# ----------------------------------------------------------------------------------------------------------------------
    def export_data_pandas(self,data,filename_out):
        if data.y is not None:
            df = pd.concat([pd.DataFrame(data.y),pd.DataFrame(data.x)],axis=1)
        else:
            df = pd.DataFrame(data.x)
        df.to_csv(self.folder_out+filename_out,index=False)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_graph_pyvis(self,data,filename_out):    #.html
        net = Network()
        edges = data.edge_index.numpy().T
        X = data.x.numpy()
        if data.y is None:
            net.add_nodes([i for i in range(X.shape[0])])
        else:
            colors = [(10, 10, 10) if y == 0 else (100, 100, 100) for y in data.y]
            net.add_nodes([i for i in range(X.shape[0])],color=colors)
        net.save_graph(self.folder_out+filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_graph_gephi(self,data,filename_out):   #.gexf
        edges = data.edge_index.numpy().T
        g = networkx.Graph([tuple(e) for e in edges])
        networkx.write_gexf(g, self.folder_out + filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_colors(self,labels):
        labels_unique = [e for e in numpy.unique(labels)]
        colors_pal = tools_draw_numpy.get_colors(len(labels_unique), colormap='jet')[:, [2, 1, 0]] / 255.0
        colors = [tuple(colors_pal[labels_unique.index(e)]) for e in labels]
        return colors
# ----------------------------------------------------------------------------------------------------------------------
    def get_cluster_id_Louvain(self, data):
        #self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        edges = data.edge_index.numpy().T
        if edges.shape[0]==0:
            return numpy.zeros(data.x.shape[0]).astype(int)
        adjacency_matrix = numpy.full((data.x.shape[0], data.x.shape[0]), 0)
        adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
        labels = Louvain().fit_predict(adjacency_matrix)
        #self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return labels
# ----------------------------------------------------------------------------------------------------------------------
    def get_cluster_id_Propagation(self, data):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)

        edges = data.edge_index.numpy().T
        adjacency_matrix = numpy.full((data.x.shape[0], data.x.shape[0]), 0)
        adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
        labels = PropagationClustering().fit_predict(adjacency_matrix)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return labels
# ----------------------------------------------------------------------------------------------------------------------
    def pos_to_centers(self,dict_pos,H=1000,W=1000):
        pad = 85

        xy = numpy.array([(xy[0],xy[1]) for xy in dict_pos.values()])
        xy[:,0]=pad+( xy[:,0]+1)*(W-2*pad)/2
        xy[:,1]=pad+(-xy[:,1]+1)*(H-2*pad)/2

        dct_centers = dict(zip(dict_pos.keys(),xy.astype(int)))

        return dct_centers
# ----------------------------------------------------------------------------------------------------------------------
    def export_graph_v1(self,G,pos,colors,filename_out,node_size=300,alpha=1.0):
        #labels=dict(zip(G.nodes, [labels[k] for k in G.nodes]))

        plt.figure(figsize=(10, 10))
        plt.clf()

        networkx.draw(G,node_color=[colors[g] for g in G.nodes] if colors is not None else None,alpha=alpha,node_size=node_size,pos=pos,with_labels=True)
        plt.savefig(self.folder_out+filename_out)
        plt.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_graph_v2(self,edges,pos,colors,filename_out):
        W, H = 1000, 1000
        image = numpy.full((H, W, 3), 255, dtype=numpy.uint8)
        dct_centers = self.pos_to_centers(pos, H, W)
        labels = [k for k in pos.keys()]
        colors = (255 * numpy.array(colors)).astype(int)[numpy.array(labels)][:,[2,1,0]]

        idx = numpy.argsort(labels)
        for e in edges:
            pos1,pos2 =  dct_centers[e[0]],dct_centers[e[1]]
            image = tools_draw_numpy.draw_line(image, pos1[1],pos1[0],pos2[1],pos2[0],color_bgr=(0,0,0), antialiasing=True)


        image = tools_draw_numpy.draw_points(image, [dct_centers[l] for l in labels],
                                             color=colors,
                                             #labels=labels,
                                             w = 40)
        cv2.imwrite(self.folder_out + filename_out,image)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_graph(self,data, filename_out,layout='shell',node_size=300,alpha=1.0):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)

        cluster_id = data.x['cluster_id'].values if (isinstance(data.x, (pd.DataFrame)) and ('cluster_id' in data.x.columns)) else self.get_cluster_id_Louvain(data)
        sizes = numpy.array(list(zip(numpy.unique(cluster_id), [sum(cluster_id == cl_id) for cl_id in numpy.unique(cluster_id)])))
        good_clusters = sizes[sizes[:, 1] > 1, 0]
        empty_clusters = sizes[sizes[:, 1] <= 1, 0]


        N = sizes[sizes[:, 1] > 1][:, 0].max()
        tail_size = sizes[sizes[:, 1] == 1][:, 1].sum()
        colors = numpy.array(self.get_colors(numpy.arange(1+N))+[(0,0,0)]*tail_size)

        G = networkx.Graph([tuple(e) for e in data.edge_index.numpy().T])

        if layout=='shell':
            pos = networkx.drawing.shell_layout(G)
        elif layout == 'shell_sorted':
            pos0 = networkx.drawing.shell_layout(G)
            if len(pos0.keys())>0:
                keys0 = numpy.array(list(pos0.keys()))
                keys = keys0[numpy.argsort(cluster_id[keys0])]
                pos = dict(zip(keys, pos0.values()))
            else:
                pos = pos0

        elif layout=='random':
            pos = networkx.drawing.random_layout(G)
        else:
            pos = networkx.drawing.spring_layout(G)

        self.export_graph_v1(G,pos,colors[cluster_id],filename_out,node_size=node_size,alpha=alpha)
        #self.export_graph_v2(data.edge_index.numpy().T,pos,colors[cluster_id],filename_out)

        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_mat(self, data, filename_out,layout='shell'):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        cluster_id = data.x['cluster_id'].values if (isinstance(data.x, (pd.DataFrame)) and ('cluster_id' in data.x.columns)) else self.get_cluster_id_Louvain(data)
        sizes = numpy.array(list(zip(numpy.unique(cluster_id),[sum(cluster_id==cl_id) for cl_id in numpy.unique(cluster_id)])))
        N = sizes[sizes[:, 1] > 1][:, 0].max()
        tail_size = sizes[sizes[:, 1] == 1][:, 1].sum()

        adjacency_matrix = numpy.full((data.x.shape[0],data.x.shape[0],3),255)
        if layout == 'shell_sorted':
            adjacency_matrix[numpy.arange(-tail_size, 0), numpy.arange(-tail_size, 0)] = 0

        colors = self.get_colors(numpy.arange(1 + N))+[(0, 0, 0)] * tail_size
        colors = (255 * numpy.array(colors)).astype(int)[:, [2, 1, 0]]

        idx_sort = numpy.argsort(cluster_id)
        idx_sort_inv = numpy.argsort(idx_sort)

        for e in data.edge_index.numpy().T:
            if layout == 'shell_sorted':
                adjacency_matrix[idx_sort_inv[e[0]],idx_sort_inv[e[1]]] = colors[cluster_id[e[0]]]
                adjacency_matrix[idx_sort_inv[e[1]],idx_sort_inv[e[0]]] = colors[cluster_id[e[1]]]
            else:
                adjacency_matrix[e[0], e[1]] = colors[cluster_id[e[0]]]
                adjacency_matrix[e[1], e[0]] = colors[cluster_id[e[1]]]

        cv2.imwrite(self.folder_out + filename_out, adjacency_matrix)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
