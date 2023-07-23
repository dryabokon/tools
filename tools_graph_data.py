#http://moviegraphs.cs.toronto.edu/
#https://www.data-to-viz.com/story/AdjacencyMatrix.html
#https://github.com/briatte/awesome-network-analysis
#https://www.toptal.com/data-science/graph-data-science-python-networkx
#https://github.com/topics/graph-anomaly-detection
#https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/tree/main/graph%20data
#https://towardsdatascience.com/pyvis-visualize-interactive-network-graphs-in-python-77e059791f01
# ----------------------------------------------------------------------------------------------------------------------
import math
import cv2
import networkx
import numpy
import pandas as pd
import inspect

import scipy.io as sio
import scipy.sparse as sp

import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid,TUDataset, KarateClub,GNNBenchmarkDataset
from torch_geometric.data import Data

from pygod.utils import load_data
import pygod.detector

from sknetwork.clustering import Louvain,PropagationClustering

from pyvis.network import Network
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
import tools_time_profiler
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Graph_Processor(object):
    def __init__(self,folder_out):
        self.folder_out=folder_out
        self.detector = self.init_detector()
        self.TP = tools_time_profiler.Time_Profiler()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_data_karate(self):
        data = KarateClub()[0]
        return data
# ----------------------------------------------------------------------------------------------------------------------
    def get_data_facebook(self,filename_in):
        df =  pd.read_csv(filename_in,sep=' ')
        edge_index = df.values
        data = Data(x=numpy.arange(1+numpy.max(edge_index)), edge_index=torch.from_numpy(numpy.array(edge_index).T))
        return data
# ----------------------------------------------------------------------------------------------------------------------
    def get_data_pygod(self,dataset = 'inj_cora'):# see more at https://github.com/pygod-team/data
        data = load_data(dataset)
        return data
# ----------------------------------------------------------------------------------------------------------------------
    def get_data_IMDB(self,filename_in):
        df =  pd.read_csv(filename_in)#.iloc[:10]

        idx1,idx2 = 1,5
        E1 = df.iloc[:,idx1].unique().tolist()
        E2 = [e for e in (numpy.unique(numpy.concatenate([df.iloc[r, idx2].split(',') for r in range(df.shape[0])])))]

        edge_index = []
        for r in range(df.shape[0]):
            i1 = E1.index(df.iloc[r, idx1])
            for e in df.iloc[r, idx2].split(','):
                i2 = E2.index(e)
                edge_index.append((i1,i2+len(E1)))

        part1 = numpy.concatenate([numpy.array([df.columns[idx1]]*len(E1)).reshape((-1,1)),numpy.array(E1).reshape((-1,1))],axis=1)
        part2 = numpy.concatenate([numpy.array([df.columns[idx2]]*len(E2)).reshape((-1,1)),numpy.array(E2).reshape((-1,1))],axis=1)
        KV = numpy.concatenate([part1, part2], axis=0)
        data = Data(x=KV, edge_index=torch.from_numpy(numpy.array(edge_index).T))
        return data
# ----------------------------------------------------------------------------------------------------------------------
    def get_data_GNNBenchmark(self,dataset='MNIST'):
        data = GNNBenchmarkDataset(root=self.folder_out,name=dataset)
        return data
# ----------------------------------------------------------------------------------------------------------------------
    def get_data_TUDataset(self,dataset='MUTAG'):
        data = TUDataset(root=self.folder_out,name=dataset)
        #data = Data(x=data.x, edge_index=data.edge_index)
        return data
# ----------------------------------------------------------------------------------------------------------------------# ----------------------------------------------------------------------------------------------------------------------
    def get_data_planetoid(self,dataset='Cora'):# seee more 'https://github.com/kimiyoung/planetoid/raw/master/data'
        data = Planetoid(root=self.folder_out, name=dataset)[0]
        return data
# ----------------------------------------------------------------------------------------------------------------------
    def load_data_mat(self,filename):

        data = sio.loadmat(filename)
        data = {"features": sp.lil_matrix(data['Attributes']),"adj": sp.csr_matrix(data['Network']),"ad_labels": numpy.squeeze(numpy.array(data['Label']))}
        features = torch.from_numpy(data["features"].todense()).float()
        adj = data["adj"]
        #ad_labels = data['ad_labels']
        edge_index, _ = from_scipy_sparse_matrix(adj)
        data = Data(x=features, edge_index=edge_index)
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
    def init_detector(self):
        # "AdONE", "ANOMALOUS", "AnomalyDAE", "CoLA","CONAD", "DOMINANT", "DONE", "GAAN", "GAE", "GUIDE", "OCGNN", "ONE","Radar", "SCAN"
        detector = pygod.detector.SCAN()
        return detector
# ----------------------------------------------------------------------------------------------------------------------
    def get_labels_anomalies(self,data):

        self.detector.fit(data)
        pred, score, prob, conf = self.detector.predict(data,return_pred=True,return_score=True,return_prob=True,return_conf=True)
        #pd.DataFrame({'Y':data.y,'pred':pred,'score':score,'prob':prob,'conf':conf}).to_csv('./data/output/xxx.csv',index=False)
        return score
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
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)

        edges = data.edge_index.numpy().T
        adjacency_matrix = numpy.full((data.x.shape[0], data.x.shape[0]), 0)
        adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
        labels = Louvain().fit_predict(adjacency_matrix)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
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
    def get_labels_entities(self,data):
        value = data.x[:, 1]
        labels = dict(zip(numpy.arange(data.x.shape[0]), [l for l in value]))
        return labels
# ----------------------------------------------------------------------------------------------------------------------
    def pos_to_centers(self,dict_pos,H=1000,W=1000):
        pad = 85

        xy = numpy.array([(xy[0],xy[1]) for xy in dict_pos.values()])
        xy[:,0]=pad+( xy[:,0]+1)*(W-2*pad)/2
        xy[:,1]=pad+(-xy[:,1]+1)*(H-2*pad)/2

        return numpy.array(xy).astype(int)
# ----------------------------------------------------------------------------------------------------------------------
    def export_graph_v1(self,G,pos,colors,filename_out,node_size=300,alpha=1.0):
        #labels=dict(zip(G.nodes, [labels[k] for k in G.nodes]))

        plt.figure(figsize=(10, 10))
        plt.clf()

        networkx.draw(G,node_color=[colors[g] for g in G.nodes] if colors is not None else None,alpha=alpha,node_size=node_size,pos=pos,with_labels=False)
        plt.savefig(self.folder_out+filename_out)
        plt.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_graph_v2(self,edges,pos,colors,filename_out):
        W, H = 1000, 1000
        image = numpy.full((H, W, 3), 255, dtype=numpy.uint8)
        xy_centers = self.pos_to_centers(pos, H, W)
        labels = [k for k in pos.keys()]
        colors = (255 * numpy.array(colors)).astype(int)[numpy.array([k for k in pos.keys()])][:,[2,1,0]]

        idx = numpy.argsort(labels)
        for e in edges:
            pos1,pos2 =  xy_centers[idx[e[0]]],xy_centers[idx[e[1]]]
            image = tools_draw_numpy.draw_line(image, pos1[1],pos1[0],pos2[1],pos2[0],color_bgr=(0,0,0), antialiasing=True)


        image = tools_draw_numpy.draw_points(image, xy_centers,
                                             color=colors,
                                             #labels=labels,
                                             w = 40)
        cv2.imwrite(self.folder_out + filename_out,image)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def construct_animation(self,data,node_size=300,alpha=1.0):

        G = networkx.Graph([tuple(e) for e in data.edge_index.numpy().T])
        cluster_id = self.get_cluster_id_Louvain(data)
        colors = self.get_colors(cluster_id)

        pos_start = networkx.drawing.shell_layout(G)
        # idx_orig = [k for k in pos_start.keys()]
        # idx_sort = numpy.argsort(cluster_id)
        # pos_stop = dict(zip([idx_sort[i] for i in range(len(cluster_id))], [pos_start[idx_orig[i]] for i in range(len(cluster_id))]))

        pos_stop = networkx.drawing.spring_layout(G)

        N = 2*20
        for n in range(N)[::-1]:
            a = float((N-1-n)/(N-1))
            pos = dict(zip(pos_start.keys(),[pos_start[k] * a + pos_stop[k] * (1.0 - a) for k in pos_start.keys()]))
            if n<=N-2:
                colors = [(0,0,0) for e in range(len(colors))]
            self.export_graph_v1(G, pos, colors, '%04d.png'%n, node_size,alpha)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_graph(self,data, filename_out,layout='shell',node_size=300,alpha=1.0):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        G = networkx.Graph([tuple(e) for e in data.edge_index.numpy().T])
        #cluster_id = self.get_cluster_id_Louvain(data)
        cluster_id = self.get_cluster_id_Propagation(data)

        colors = self.get_colors(cluster_id)

        if layout=='shell':
            pos = networkx.drawing.shell_layout(G)#colors = None
        elif  layout == 'shell_sorted':
            pos = networkx.drawing.shell_layout(G)
            idx_orig = [k for k in pos.keys()]
            idx_sort = numpy.argsort(cluster_id)
            pos = dict(zip([idx_sort[i] for i in range(len(cluster_id))],[pos[idx_orig[i]] for i in range(len(cluster_id))]))
        elif layout=='random':
            pos = networkx.drawing.random_layout(G)
        else:
            pos = networkx.drawing.spring_layout(G)

        self.export_graph_v1(G,pos,colors,filename_out,node_size=node_size,alpha=alpha)
        #self.export_graph_v2(data.edge_index.numpy().T,pos,colors,filename_out)

        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_mat(self, data, filename_out,layout='shell'):

        #cluster_id = self.get_cluster_id_Louvain(data)
        cluster_id = self.get_cluster_id_Propagation(data)
        colors = self.get_colors(cluster_id)

        G = networkx.Graph([tuple(e) for e in data.edge_index.numpy().T])
        pos = networkx.drawing.shell_layout(G)
        if layout=='shell_sorted':
            idx_orig = [k for k in pos.keys()]
            idx_sort = numpy.argsort(cluster_id)
            pos = dict(zip([idx_sort[i] for i in range(len(cluster_id))], [pos[idx_orig[i]] for i in range(len(cluster_id))]))
            idx_sort_inv=numpy.argsort(idx_sort)

        colors = (255 * numpy.array(colors)).astype(int)[numpy.array([k for k in pos.keys()])][:, [2, 1, 0]]
        N = data.x.shape[0]
        adjacency_matrix = numpy.full((N,N,3),255)
        for e in data.edge_index.numpy().T:
            if layout == 'shell_sorted':
                ii1 = idx_sort_inv[e[0]]
                ii2 = idx_sort_inv[e[1]]
                adjacency_matrix[ii1,ii2] = colors[ii1]
                adjacency_matrix[ii2,ii1] = colors[ii2]

            else:
                adjacency_matrix[e[0], e[1]] = colors[e[0]]
                adjacency_matrix[e[1], e[0]] = colors[e[1]]

        # for i in range(N):
        #     adjacency_matrix[i,i]=numpy.array(colors[i])

        cv2.imwrite(self.folder_out + filename_out, adjacency_matrix)
        return
# ----------------------------------------------------------------------------------------------------------------------
